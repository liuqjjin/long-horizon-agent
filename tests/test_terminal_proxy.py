"""Security and protocol tests for the Terminal-Bench credential broker."""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import signal
import socket
import ssl
import subprocess
import sys
import threading
from collections.abc import Mapping
from dataclasses import replace
from datetime import timedelta
from http.client import HTTPConnection, HTTPSConnection
from typing import Any

import pytest

import lha.bench.terminal_proxy as proxy
import lha.bench.terminal_proxy_server as proxy_server
from lha.bench.terminal_proxy import (
    BROKER_MAX_REQUESTS,
    BROKER_STREAM_RETRY_LIMIT,
    BROKER_STREAM_RETRY_LIMIT_PER_REQUEST,
    BROKER_TTL_S,
    BrokerSecrets,
    BrokerStartup,
    HttpUpstreamTransport,
    ProxyRequest,
    TerminalProxyController,
    TerminalProxyError,
    TokenBroker,
    UpstreamResponse,
)
from lha.bench.terminal_proxy_server import _BrokerServer, _ShutdownCoordinator

_SOURCE_CONTAINER = "a" * 64
_IMAGE_ID = f"sha256:{'b' * 64}"
_CAPABILITY = "capability_" + "c" * 48
_ACCESS_TOKEN = "real-access-token-that-must-not-leak"
_ACCOUNT_ID = "real-account-id-that-must-not-leak"


def _startup(**overrides: Any) -> BrokerStartup:
    certificate_pem, private_key_pem, certificate_sha256 = proxy._generate_broker_tls_material()
    values: dict[str, Any] = {
        "evaluation_id": "tb21-fixed-20",
        "attempt_id": "regex-log-01",
        "source_container_id": _SOURCE_CONTAINER,
        "source_ip": "172.30.0.4",
        "model": "gpt-5.6-sol",
        "reasoning_effort": "max",
        "capability": _CAPABILITY,
        "credentials": BrokerSecrets(
            access_token=_ACCESS_TOKEN,
            account_id=_ACCOUNT_ID,
        ),
        "tls_certificate_pem": certificate_pem,
        "tls_certificate_sha256": certificate_sha256,
        "tls_private_key_pem": private_key_pem,
    }
    values.update(overrides)
    return BrokerStartup(**values)


def _headers(startup: BrokerStartup | None = None) -> dict[str, str]:
    startup = startup or _startup()
    return {
        "Authorization": f"Bearer {startup.capability}",
        "Content-Type": "application/json",
        "X-LHA-Evaluation-ID": startup.evaluation_id,
        "X-LHA-Attempt-ID": startup.attempt_id,
        "X-LHA-Container-ID": startup.source_container_id,
    }


def _body(*, model: str = "gpt-5.6-sol", effort: str = "max") -> bytes:
    return json.dumps(
        {
            "model": model,
            "reasoning": {"effort": effort, "summary": "auto"},
            "input": "test",
            "stream": True,
        }
    ).encode()


def _completed_sse(response_id: str = "resp_test") -> bytes:
    payload = json.dumps(
        {
            "type": "response.completed",
            "response": {"id": response_id},
        },
        separators=(",", ":"),
    )
    return f"event: response.completed\ndata: {payload}\n\n".encode()


def _data_only_completed_sse(response_id: str = "resp_test") -> bytes:
    payload = json.dumps(
        {
            "type": "response.completed",
            "response": {"id": response_id},
        },
        separators=(",", ":"),
    )
    return f"data: {payload}\n\n".encode()


class _CaptureTransport:
    def __init__(
        self,
        chunks: tuple[bytes, ...] = (
            b"event: response.output_text.delta\ndata: {\"type\":\"response."
            b"output_text.delta\",\"delta\":\"one\"}\n\n",
            _completed_sse(),
        ),
    ):
        self.calls: list[dict[str, Any]] = []
        self.chunks = chunks
        self.close_count = 0

    def __call__(
        self,
        *,
        body: bytes,
        headers: Mapping[str, str],
        access_token: str,
        account_id: str,
        timeout_s: float,
    ) -> UpstreamResponse:
        self.calls.append(
            {
                "body": body,
                "headers": dict(headers),
                "access_token": access_token,
                "account_id": account_id,
                "timeout_s": timeout_s,
            }
        )

        def close() -> None:
            self.close_count += 1

        return UpstreamResponse(
            status=200,
            headers=(("Content-Type", "text/event-stream"), ("X-Upstream", "yes")),
            body=self.chunks,
            close=close,
        )


def _request(
    startup: BrokerStartup | None = None,
    *,
    path: str = "/responses",
    headers: Mapping[str, str] | None = None,
    body: bytes | None = None,
    source_ip: str | None = None,
) -> ProxyRequest:
    startup = startup or _startup()
    return ProxyRequest(
        method="POST",
        path=path,
        headers=headers or _headers(startup),
        body=body or _body(model=startup.model, effort=startup.reasoning_effort),
        source_ip=source_ip or startup.source_ip,
    )


def _consume(decision: proxy.ProxyDecision) -> bytes:
    return b"".join(decision.body)


def test_startup_stdin_round_trip_and_repr_redact_every_secret():
    startup = _startup()
    private_key = bytes(startup.tls_private_key_pem).decode("ascii")
    serialized = startup.stdin_json()
    restored = BrokerStartup.from_stdin_json(serialized)

    assert restored == startup
    assert _ACCESS_TOKEN in serialized
    assert _ACCOUNT_ID in serialized
    assert _CAPABILITY in serialized
    startup_record = json.loads(serialized)
    assert startup_record["schema_version"] == 4
    assert startup_record["stream_retry_limit"] == BROKER_STREAM_RETRY_LIMIT
    assert (
        startup_record["stream_retry_limit_per_request"]
        == BROKER_STREAM_RETRY_LIMIT_PER_REQUEST
    )
    assert startup_record["tls_private_key_pem"] == private_key
    for rendered in (repr(startup), repr(startup.credentials)):
        assert _ACCESS_TOKEN not in rendered
        assert _ACCOUNT_ID not in rendered
        assert _CAPABILITY not in rendered
        assert private_key not in rendered

    restored.clear_tls_private_key()
    assert restored.tls_private_key_pem == bytearray()
    with pytest.raises(ValueError, match="already consumed"):
        restored.stdin_json()


def test_generated_certificate_is_attempt_local_and_valid_for_only_the_broker_alias():
    from cryptography import x509

    certificate_pem, private_key_pem, certificate_sha256 = proxy._generate_broker_tls_material()
    certificate = x509.load_pem_x509_certificate(certificate_pem)
    alternative_names = certificate.extensions.get_extension_for_class(
        x509.SubjectAlternativeName
    ).value

    assert alternative_names.get_values_for_type(x509.DNSName) == [proxy.BROKER_ALIAS]
    assert certificate.not_valid_after_utc - certificate.not_valid_before_utc < timedelta(hours=1)
    assert certificate_sha256 == hashlib.sha256(certificate_pem).hexdigest()
    assert private_key_pem.startswith(b"-----BEGIN PRIVATE KEY-----")


@pytest.mark.parametrize(
    ("request_change", "expected_status"),
    [
        ({"path": "/responses/compact"}, 404),
        ({"path": "/v1/responses"}, 404),
        ({"source_ip": "172.30.0.5"}, 403),
        ({"headers": {**_headers(), "Authorization": "Bearer wrong-capability"}}, 401),
        (
            {
                "headers": {
                    **_headers(),
                    "X-LHA-Evaluation-ID": "different-evaluation",
                }
            },
            403,
        ),
        (
            {
                "headers": {
                    **_headers(),
                    "X-LHA-Attempt-ID": "different-attempt",
                }
            },
            403,
        ),
        (
            {
                "headers": {
                    **_headers(),
                    "X-LHA-Container-ID": "d" * 64,
                }
            },
            403,
        ),
        ({"body": _body(model="different-model")}, 403),
        ({"body": _body(effort="low")}, 403),
    ],
)
def test_broker_rejects_wrong_route_capability_binding_model_and_effort(
    request_change, expected_status
):
    transport = _CaptureTransport()
    broker = TokenBroker(_startup(), transport=transport)

    decision = broker.handle(_request(**request_change))

    assert decision.status == expected_status
    assert transport.calls == []
    assert _ACCESS_TOKEN.encode() not in _consume(decision)
    assert _ACCOUNT_ID.encode() not in _consume(decision)
    assert _CAPABILITY.encode() not in _consume(decision)


def test_broker_accepts_at_most_sixty_upstream_requests_without_retry():
    transport = _CaptureTransport(chunks=(_completed_sse(),))
    broker = TokenBroker(_startup(), transport=transport)

    statuses = []
    for _ in range(BROKER_MAX_REQUESTS + 1):
        decision = broker.handle(_request())
        statuses.append(decision.status)
        _consume(decision)

    assert statuses == [200] * BROKER_MAX_REQUESTS + [429]
    assert len(transport.calls) == BROKER_MAX_REQUESTS
    receipt = broker.receipt()
    assert receipt["downstream_accepted_requests"] == BROKER_MAX_REQUESTS
    assert receipt["upstream_attempts"] == BROKER_MAX_REQUESTS
    assert receipt["stream_retries_used"] == 0
    assert receipt["rejected_requests"] == 1


def test_buffering_expires_without_emitting_partial_response():
    clock = [100.0]
    close_count = 0

    def transport(**_kwargs: Any) -> UpstreamResponse:
        def chunks():
            yield b"first"
            clock[0] += BROKER_TTL_S
            yield b"must-not-pass"

        def close() -> None:
            nonlocal close_count
            close_count += 1

        return UpstreamResponse(
            status=200,
            headers=(("Content-Type", "text/event-stream"),),
            body=chunks(),
            close=close,
        )

    broker = TokenBroker(_startup(), transport=transport, monotonic=lambda: clock[0])
    decision = broker.handle(_request())

    assert decision.status == 502
    assert b"first" not in _consume(decision)
    decision.close()
    assert close_count == 1
    assert broker.receipt()["revoked"] is True
    assert broker.receipt()["rejection_reasons"] == {"upstream_timeout": 1}


def test_broker_expires_at_registered_ttl_and_revoke_closes_active_stream():
    clock = [100.0]
    transport = _CaptureTransport()
    broker = TokenBroker(_startup(), transport=transport, monotonic=lambda: clock[0])

    active = broker.handle(_request())
    assert active.status == 200
    broker.revoke()
    assert transport.close_count == 1
    assert broker.handle(_request()).status == 410

    clock[0] = 100.0
    expiring = TokenBroker(_startup(), transport=transport, monotonic=lambda: clock[0])
    clock[0] += BROKER_TTL_S
    assert expiring.handle(_request()).status == 410


def test_broker_receipt_contains_no_token_account_or_capability():
    broker = TokenBroker(_startup(), transport=_CaptureTransport())
    decision = broker.handle(_request())
    _consume(decision)
    broker.revoke()

    rendered = json.dumps(broker.receipt(outcome="sigterm"), sort_keys=True)

    assert _ACCESS_TOKEN not in rendered
    assert _ACCOUNT_ID not in rendered
    assert _CAPABILITY not in rendered
    assert '"revoked": true' in rendered


def test_broker_records_a_bounded_secret_free_upstream_error():
    error_body = json.dumps(
        {
            "error": {
                "code": "invalid_request",
                "type": "invalid_request_error",
                "param": "tools",
                "message": "A required tool field is missing.",
            }
        },
        separators=(",", ":"),
    ).encode()
    close_count = 0

    def transport(**_kwargs: Any) -> UpstreamResponse:
        def close() -> None:
            nonlocal close_count
            close_count += 1

        return UpstreamResponse(
            status=400,
            headers=(("Content-Type", "application/json"),),
            body=(error_body,),
            close=close,
        )

    broker = TokenBroker(_startup(), transport=transport)
    decision = broker.handle(_request())

    assert decision.status == 400
    assert _consume(decision) == error_body
    assert close_count == 1
    diagnostic = broker.receipt()["upstream_error"]
    assert diagnostic == {
        "status": 400,
        "body_sha256": hashlib.sha256(error_body).hexdigest(),
        "body_bytes": len(error_body),
        "error_code": "invalid_request",
        "error_type": "invalid_request_error",
        "error_param": "tools",
        "message": "A required tool field is missing.",
    }
    rendered = json.dumps(diagnostic, sort_keys=True)
    assert _ACCESS_TOKEN not in rendered
    assert _ACCOUNT_ID not in rendered
    assert _CAPABILITY not in rendered


def test_broker_allows_one_registered_upstream_stream_retry():
    class RemoteProtocolError(OSError):
        pass

    calls = 0
    close_counts = [0, 0]
    second_response = b"data: second-attempt-only\n\n" + _completed_sse("resp_second")

    def failing_body():
        yield b"data: partial\n\n"
        raise RemoteProtocolError("transport details must not enter the receipt")

    def transport(**_kwargs: Any) -> UpstreamResponse:
        nonlocal calls
        attempt = calls
        calls += 1

        def close() -> None:
            close_counts[attempt] += 1

        return UpstreamResponse(
            status=200,
            headers=(
                ("Content-Type", "text/event-stream; charset=utf-8"),
                ("Content-Length", "999999"),
                ("Connection", "keep-alive"),
            ),
            body=failing_body() if attempt == 0 else (second_response,),
            close=close,
        )

    broker = TokenBroker(_startup(), transport=transport)
    decision = broker.handle(_request())
    rendered = _consume(decision)

    assert decision.status == 200
    assert rendered == second_response
    assert b"partial" not in rendered
    assert calls == 2
    assert close_counts == [1, 1]
    assert [value for name, value in decision.headers if name.lower() == "content-type"] == [
        "text/event-stream"
    ]
    assert [
        value for name, value in decision.headers if name.lower() == "content-length"
    ] == [str(len(second_response))]
    assert [value for name, value in decision.headers if name.lower() == "connection"] == [
        "close"
    ]
    receipt = broker.receipt()
    assert receipt["upstream_stream_errors"] == {"RemoteProtocolError": 1}
    assert receipt["downstream_accepted_requests"] == 1
    assert receipt["upstream_attempts"] == 2
    assert receipt["upstream_statuses"] == {"200": 2}
    assert receipt["stream_retries_used"] == 1
    assert receipt["stream_retried_requests"] == 1
    assert receipt["max_stream_retries_on_request"] == 1
    assert receipt["revoked"] is False
    assert "transport details" not in json.dumps(receipt, sort_keys=True)


def test_broker_recovers_after_four_stream_failures_on_one_request():
    class RemoteProtocolError(OSError):
        pass

    close_counts = [0] * 5
    calls = 0
    completed = _completed_sse("resp_fifth_attempt")

    def transport(**_kwargs: Any) -> UpstreamResponse:
        nonlocal calls
        attempt = calls
        calls += 1

        def chunks():
            if attempt < 4:
                yield f"data: partial-{attempt}\n\n".encode()
                raise RemoteProtocolError("private transport detail")
            yield completed

        def close() -> None:
            close_counts[attempt] += 1

        return UpstreamResponse(
            status=200,
            headers=(("Content-Type", "text/event-stream"),),
            body=chunks(),
            close=close,
        )

    broker = TokenBroker(_startup(), transport=transport)
    decision = broker.handle(_request())
    rendered = _consume(decision)

    assert decision.status == 200
    assert rendered == completed
    assert b"partial" not in rendered
    assert calls == 5
    assert close_counts == [1] * 5
    receipt = broker.receipt()
    assert receipt["downstream_accepted_requests"] == 1
    assert receipt["upstream_attempts"] == 5
    assert receipt["upstream_statuses"] == {"200": 5}
    assert receipt["stream_retries_used"] == 4
    assert receipt["stream_retried_requests"] == 1
    assert receipt["max_stream_retries_on_request"] == 4
    assert receipt["upstream_stream_errors"] == {"RemoteProtocolError": 4}
    assert receipt["rejection_reasons"] == {}
    assert receipt["revoked"] is False


def test_broker_revokes_after_per_request_stream_retry_budget_is_exceeded():
    class RemoteProtocolError(OSError):
        pass

    calls = 0
    close_counts = [0] * 5

    def transport(**_kwargs: Any) -> UpstreamResponse:
        nonlocal calls
        attempt = calls
        calls += 1

        def chunks():
            yield f"data: hidden-{attempt}\n\n".encode()
            raise RemoteProtocolError("private transport detail")

        def close() -> None:
            close_counts[attempt] += 1

        return UpstreamResponse(
            status=200,
            headers=(("Content-Type", "text/event-stream"),),
            body=chunks(),
            close=close,
        )

    broker = TokenBroker(_startup(), transport=transport)
    decision = broker.handle(_request())

    assert decision.status == 502
    assert b"hidden" not in _consume(decision)
    assert calls == 5
    assert close_counts == [1] * 5
    receipt = broker.receipt()
    assert receipt["downstream_accepted_requests"] == 1
    assert receipt["upstream_attempts"] == 5
    assert receipt["stream_retries_used"] == 4
    assert receipt["stream_retried_requests"] == 1
    assert receipt["max_stream_retries_on_request"] == 4
    assert receipt["upstream_stream_errors"] == {"RemoteProtocolError": 5}
    assert receipt["rejection_reasons"] == {"upstream_stream_failure": 1}
    assert receipt["revoked"] is True


def test_broker_global_stream_retry_budget_is_shared_across_requests():
    class RemoteProtocolError(OSError):
        pass

    calls = 0
    close_counts = [0] * 16

    def transport(**_kwargs: Any) -> UpstreamResponse:
        nonlocal calls
        attempt = calls
        calls += 1

        def chunks():
            if attempt % 5 < 4:
                yield f"data: hidden-{attempt}\n\n".encode()
                raise RemoteProtocolError("private transport detail")
            yield _completed_sse(f"resp_{attempt}")

        def close() -> None:
            close_counts[attempt] += 1

        return UpstreamResponse(
            status=200,
            headers=(("Content-Type", "text/event-stream"),),
            body=chunks(),
            close=close,
        )

    broker = TokenBroker(_startup(), transport=transport)
    decisions = [broker.handle(_request()) for _ in range(4)]

    assert [decision.status for decision in decisions] == [200, 200, 200, 502]
    assert all(b"hidden" not in _consume(decision) for decision in decisions)
    assert calls == 16
    assert close_counts == [1] * 16
    receipt = broker.receipt()
    assert receipt["downstream_accepted_requests"] == 4
    assert receipt["upstream_attempts"] == 16
    assert receipt["upstream_statuses"] == {"200": 16}
    assert receipt["stream_retries_used"] == BROKER_STREAM_RETRY_LIMIT
    assert receipt["stream_retried_requests"] == 3
    assert receipt["max_stream_retries_on_request"] == 4
    assert receipt["upstream_stream_errors"] == {"RemoteProtocolError": 13}
    assert receipt["rejection_reasons"] == {"upstream_stream_failure": 1}
    assert receipt["revoked"] is True


def test_broker_rejects_non_success_status_from_internal_stream_retry():
    class RemoteProtocolError(OSError):
        pass

    calls = 0
    close_counts = [0, 0]
    error_body_read = False

    def transport(**_kwargs: Any) -> UpstreamResponse:
        nonlocal calls
        attempt = calls
        calls += 1

        def first_body():
            yield b"data: hidden-first-attempt\n\n"
            raise RemoteProtocolError("private transport detail")

        def second_body():
            nonlocal error_body_read
            error_body_read = True
            yield b"upstream error details"

        def close() -> None:
            close_counts[attempt] += 1

        return UpstreamResponse(
            status=200 if attempt == 0 else 503,
            headers=(("Content-Type", "text/event-stream"),),
            body=first_body() if attempt == 0 else second_body(),
            close=close,
        )

    broker = TokenBroker(_startup(), transport=transport)
    decision = broker.handle(_request())

    assert decision.status == 502
    assert b"hidden-first-attempt" not in _consume(decision)
    assert calls == 2
    assert close_counts == [1, 1]
    assert error_body_read is False
    receipt = broker.receipt()
    assert receipt["upstream_statuses"] == {"200": 1, "503": 1}
    assert receipt["stream_retries_used"] == 1
    assert receipt["upstream_stream_errors"] == {"RemoteProtocolError": 1}
    assert receipt["rejection_reasons"] == {"upstream_retry_failed": 1}
    assert receipt["upstream_error"] is None
    assert receipt["revoked"] is True


def test_broker_does_not_retry_an_unregistered_stream_exception():
    def body():
        yield b"data: hidden-prefix\n\n"
        raise OSError("private filesystem or transport detail")

    transport = _CaptureTransport(chunks=())

    def failing_transport(**kwargs: Any) -> UpstreamResponse:
        response = transport(**kwargs)
        response.body = body()
        return response

    broker = TokenBroker(_startup(), transport=failing_transport)
    decision = broker.handle(_request())

    assert decision.status == 502
    assert b"hidden-prefix" not in _consume(decision)
    assert len(transport.calls) == 1
    assert transport.close_count == 1
    receipt = broker.receipt()
    assert receipt["stream_retries_used"] == 0
    assert receipt["upstream_stream_errors"] == {"OSError": 1}
    assert receipt["rejection_reasons"] == {"upstream_stream_failure": 1}
    assert receipt["revoked"] is True


def test_broker_revokes_immediately_for_unregistered_transport_exception():
    class SensitiveConnectError(OSError):
        pass

    def transport(**_kwargs: Any) -> UpstreamResponse:
        raise SensitiveConnectError(
            f"credential={_ACCESS_TOKEN} path=/private/credential/location"
        )

    broker = TokenBroker(_startup(), transport=transport)

    decision = broker.handle(_request())
    rendered_receipt = json.dumps(broker.receipt(), sort_keys=True)

    assert decision.status == 502
    assert broker.receipt()["upstream_transport_errors"] == {
        "SensitiveConnectError": 1
    }
    assert broker.receipt()["rejection_reasons"] == {
        "upstream_transport_exception": 1
    }
    assert broker.receipt()["revoked"] is True
    assert _ACCESS_TOKEN not in rendered_receipt
    assert "/private/credential/location" not in rendered_receipt


def test_broker_allows_one_registered_transport_retry():
    class RemoteProtocolError(OSError):
        pass

    def transport(**_kwargs: Any) -> UpstreamResponse:
        raise RemoteProtocolError("transport details must not enter the receipt")

    broker = TokenBroker(_startup(), transport=transport)

    assert broker.handle(_request()).status == 502
    assert broker.receipt()["upstream_transport_errors"] == {
        "RemoteProtocolError": 1
    }
    assert broker.receipt()["rejection_reasons"] == {
        "upstream_transport_exception": 1
    }
    assert broker.receipt()["revoked"] is False

    assert broker.handle(_request()).status == 502
    assert broker.receipt()["upstream_transport_errors"] == {
        "RemoteProtocolError": 2
    }
    assert broker.receipt()["revoked"] is True


@pytest.mark.parametrize(
    ("response", "expected_reason"),
    [
        (UpstreamResponse(status=99, headers=(), body=()), "upstream_invalid_status"),
        (
            UpstreamResponse(
                status=200,
                headers=(("Content-Type", "text/event-stream", "extra"),),  # type: ignore[arg-type]
                body=(_completed_sse(),),
            ),
            "upstream_header_shape",
        ),
        (
            UpstreamResponse(
                status=200,
                headers=(("Content-Type", "text/event-stream\t"),),
                body=(_completed_sse(),),
            ),
            "upstream_header_shape",
        ),
        (
            UpstreamResponse(
                status=200,
                headers=(("Bad Header", "text/event-stream"),),
                body=(_completed_sse(),),
            ),
            "upstream_header_shape",
        ),
        (
            UpstreamResponse(
                status=400,
                headers=(("Content-Type", "application/json"),),
                body=("not-bytes",),
            ),
            "upstream_invalid_error_body",
        ),
        (
            UpstreamResponse(
                status=400,
                headers=(("Content-Type", "application/json"),),
                body=(_CAPABILITY.encode(),),
            ),
            "upstream_secret_in_error_body",
        ),
    ],
)
def test_broker_assigns_distinct_fixed_reasons_to_upstream_502_failures(
    response,
    expected_reason,
):
    broker = TokenBroker(
        _startup(),
        transport=lambda **_kwargs: response,
    )

    decision = broker.handle(_request())

    assert decision.status == 502
    assert broker.receipt()["rejection_reasons"] == {expected_reason: 1}
    if expected_reason == "upstream_header_shape":
        assert broker.receipt()["observed_content_types"] == []


def test_broker_stops_at_structurally_valid_completed_event_before_transport_eof():
    class RemoteProtocolError(OSError):
        pass

    close_count = 0
    requested_after_completion = False
    completed = _completed_sse()

    def chunks():
        nonlocal requested_after_completion
        yield b"event: response.output_text.delta\ndata: {\"type\":\"response."
        yield b"output_text.delta\",\"delta\":\"ok\"}\n\n" + completed[:17]
        yield completed[17:] + b"must-not-be-forwarded"
        requested_after_completion = True
        raise http.client.IncompleteRead(b"")

    def transport(**_kwargs: Any) -> UpstreamResponse:
        def close() -> None:
            nonlocal close_count
            close_count += 1
            raise RemoteProtocolError("HTTP/2 close failed after response.completed")

        return UpstreamResponse(
            status=200,
            headers=(("Content-Type", "text/event-stream"),),
            body=chunks(),
            close=close,
        )

    broker = TokenBroker(_startup(), transport=transport)
    decision = broker.handle(_request())
    rendered = _consume(decision)

    assert rendered.endswith(completed)
    assert b"must-not-be-forwarded" not in rendered
    assert requested_after_completion is False
    assert close_count == 1
    assert broker.receipt()["upstream_stream_errors"] == {}


def test_broker_accepts_data_only_completed_sse_like_codex_0141():
    completed = _data_only_completed_sse()
    broker = TokenBroker(
        _startup(),
        transport=_CaptureTransport(
            chunks=(completed[:31], completed[31:]),
        ),
    )

    rendered = _consume(broker.handle(_request()))

    assert rendered == completed
    assert broker.receipt()["upstream_stream_errors"] == {}


@pytest.mark.parametrize(
    ("headers", "expected_observed"),
    [
        ((), []),
        (
            (
                ("Content-Type", "text/event-stream"),
                ("Content-Type", "TEXT/EVENT-STREAM; charset=utf-8"),
            ),
            ["text/event-stream", "text/event-stream; charset=utf-8"],
        ),
        (
            (
                (
                    "Content-Type",
                    'text/event-stream; profile="one,two", text/event-stream',
                ),
            ),
            ['text/event-stream; profile="one,two", text/event-stream'],
        ),
    ],
)
def test_broker_accepts_repeated_or_coalesced_identical_sse_content_types(
    headers,
    expected_observed,
):
    close_count = 0
    response_body = _completed_sse()
    if not headers:
        response_body = (
            b"event: response.output_text.delta\n"
            b'data: {"type":"response.output_text.delta","delta":"ok"}\n\n'
            + response_body
        )

    def transport(**_kwargs: Any) -> UpstreamResponse:
        def close() -> None:
            nonlocal close_count
            close_count += 1

        return UpstreamResponse(
            status=200,
            headers=headers,
            body=(response_body,),
            close=close,
        )

    broker = TokenBroker(_startup(), transport=transport)
    decision = broker.handle(_request())

    assert decision.status == 200
    assert _consume(decision) == response_body
    assert close_count == 1
    assert broker.receipt()["rejection_reasons"] == {}
    assert broker.receipt()["observed_content_types"] == expected_observed
    assert [
        value for name, value in decision.headers if name.lower() == "content-type"
    ] == ["text/event-stream"]


@pytest.mark.parametrize(
    ("headers", "expected_observed"),
    [
        (
            (("Content-Type", "application/json"),),
            ["application/json"],
        ),
        (
            (
                ("Content-Type", "text/event-stream"),
                ("Content-Type", "application/json"),
            ),
            ["text/event-stream", "application/json"],
        ),
        (
            (("Content-Type", "text/event-stream, application/json"),),
            ["text/event-stream, application/json"],
        ),
        (
            (("Content-Type", 'text/event-stream; profile="unterminated'),),
            ['text/event-stream; profile="unterminated'],
        ),
        (
            (("Content-Type", "text/event-stream,,text/event-stream"),),
            ["text/event-stream,,text/event-stream"],
        ),
    ],
)
def test_broker_rejects_present_mixed_or_malformed_sse_content_types(
    headers,
    expected_observed,
):
    close_count = 0

    def transport(**_kwargs: Any) -> UpstreamResponse:
        def close() -> None:
            nonlocal close_count
            close_count += 1

        return UpstreamResponse(
            status=200,
            headers=headers,
            body=(_completed_sse(),),
            close=close,
        )

    broker = TokenBroker(_startup(), transport=transport)
    decision = broker.handle(_request())

    assert decision.status == 502
    assert _completed_sse() not in _consume(decision)
    assert close_count == 1
    receipt = broker.receipt()
    assert receipt["rejection_reasons"] == {"upstream_content_type_invalid": 1}
    assert receipt["upstream_stream_errors"] == {}
    assert receipt["observed_content_types"] == expected_observed
    assert receipt["revoked"] is True


def test_broker_requires_valid_responses_events_when_content_type_is_missing():
    malformed = b"data: not-json\n\n" + _completed_sse()
    broker = TokenBroker(
        _startup(),
        transport=lambda **_kwargs: UpstreamResponse(
            status=200,
            headers=(),
            body=(malformed,),
        ),
    )

    decision = broker.handle(_request())

    assert decision.status == 502
    assert malformed not in _consume(decision)
    receipt = broker.receipt()
    assert receipt["observed_content_types"] == []
    assert receipt["upstream_stream_errors"] == {"ValueError": 1}
    assert receipt["rejection_reasons"] == {"upstream_invalid_sse": 1}
    assert receipt["revoked"] is True


def test_broker_retries_then_rejects_truncated_body_without_content_type():
    calls = 0
    close_count = 0

    def transport(**_kwargs: Any) -> UpstreamResponse:
        nonlocal calls
        calls += 1

        def close() -> None:
            nonlocal close_count
            close_count += 1

        return UpstreamResponse(
            status=200,
            headers=(),
            body=(b'data: {"type":"response.output_text.delta"',),
            close=close,
        )

    broker = TokenBroker(_startup(), transport=transport)
    decision = broker.handle(_request())

    assert decision.status == 502
    assert b"response.output_text.delta" not in _consume(decision)
    assert calls == 5
    assert close_count == 5
    receipt = broker.receipt()
    assert receipt["observed_content_types"] == []
    assert receipt["stream_retries_used"] == 4
    assert receipt["stream_retried_requests"] == 1
    assert receipt["max_stream_retries_on_request"] == 4
    assert receipt["upstream_stream_errors"] == {"MissingResponseCompleted": 5}
    assert receipt["rejection_reasons"] == {"upstream_stream_failure": 1}
    assert receipt["revoked"] is True


def test_broker_bounds_observed_content_type_diagnostics():
    long_value = "text/event-stream; profile=" + "a" * 300
    broker = TokenBroker(
        _startup(),
        transport=lambda **_kwargs: UpstreamResponse(
            status=200,
            headers=(("Content-Type", long_value),),
            body=(_completed_sse(),),
        ),
    )

    assert broker.handle(_request()).status == 200
    observed = broker.receipt()["observed_content_types"]
    assert isinstance(observed, list)
    assert len(observed) == 1
    assert observed[0].startswith("sha256:")
    assert observed[0].endswith(f":bytes={len(long_value)}")
    assert long_value not in json.dumps(broker.receipt())

    headers = tuple(
        ("Content-Type", f"text/event-stream; profile={index}")
        for index in range(5)
    )
    broker = TokenBroker(
        _startup(),
        transport=lambda **_kwargs: UpstreamResponse(
            status=200,
            headers=headers,
            body=(_completed_sse(),),
        ),
    )

    assert broker.handle(_request()).status == 200
    assert broker.receipt()["observed_content_types"] == [
        "text/event-stream; profile=0",
        "text/event-stream; profile=1",
        "text/event-stream; profile=2",
        "<additional-content-types-omitted>",
    ]


def test_broker_enforces_exact_buffer_bound_without_leaking_prefix(monkeypatch):
    response = b"data: filler\n\n" + _completed_sse("resp_exact_limit")
    monkeypatch.setattr(proxy, "BROKER_MAX_BUFFERED_RESPONSE_BYTES", len(response))
    exact_transport = _CaptureTransport(chunks=(response,))
    exact_broker = TokenBroker(_startup(), transport=exact_transport)

    exact_decision = exact_broker.handle(_request())

    assert exact_decision.status == 200
    assert _consume(exact_decision) == response
    assert exact_transport.close_count == 1

    monkeypatch.setattr(proxy, "BROKER_MAX_BUFFERED_RESPONSE_BYTES", len(response) - 1)
    oversized_transport = _CaptureTransport(chunks=(response,))
    oversized_broker = TokenBroker(_startup(), transport=oversized_transport)

    oversized_decision = oversized_broker.handle(_request())
    rendered = _consume(oversized_decision)

    assert oversized_decision.status == 502
    assert response not in rendered
    assert b"filler" not in rendered
    assert oversized_transport.close_count == 1
    receipt = oversized_broker.receipt()
    assert receipt["upstream_attempts"] == 1
    assert receipt["stream_retries_used"] == 0
    assert receipt["upstream_stream_errors"] == {"UpstreamResponseTooLarge": 1}
    assert receipt["rejection_reasons"] == {"upstream_response_too_large": 1}
    assert receipt["revoked"] is True


@pytest.mark.parametrize(
    "invalid_terminal",
    [
        b"event: response.completed\ndata: not-json\n\n",
        b"event: response.completed\ndata: {\"type\":\"response.completed\"}",
        (
            b"event: response.completed\n"
            b"data: {\"type\":\"response.completed\",\"response\":{}}\n\n"
        ),
    ],
)
def test_broker_rejects_invalid_or_incomplete_completed_sse(invalid_terminal):
    broker = TokenBroker(
        _startup(),
        transport=_CaptureTransport(chunks=(invalid_terminal,)),
    )

    rendered = _consume(broker.handle(_request()))

    assert _completed_sse() not in rendered
    receipt = broker.receipt()
    assert set(receipt["upstream_stream_errors"]) <= {
        "ValueError",
        "MissingResponseCompleted",
    }
    if receipt["upstream_stream_errors"] == {"ValueError": 1}:
        assert receipt["revoked"] is True
        assert receipt["stream_retries_used"] == 0
    else:
        assert receipt["upstream_stream_errors"] == {"MissingResponseCompleted": 5}
        assert receipt["stream_retries_used"] == 4
        assert receipt["stream_retried_requests"] == 1
        assert receipt["max_stream_retries_on_request"] == 4
        assert receipt["revoked"] is True


@pytest.mark.parametrize("leak_location", ["header", "body", "compressed"])
def test_broker_never_forwards_credentials_returned_by_upstream(leak_location):
    def transport(**_kwargs: Any) -> UpstreamResponse:
        headers: tuple[tuple[str, str], ...] = (("Content-Type", "text/event-stream"),)
        chunks = (b"data: safe\n\n",)
        if leak_location == "header":
            headers += (("X-Reflected-Account", _ACCOUNT_ID),)
        elif leak_location == "body":
            split = len(_ACCESS_TOKEN) // 2
            chunks = (
                b"data: prefix ",
                _ACCESS_TOKEN[:split].encode(),
                _ACCESS_TOKEN[split:].encode(),
            )
        else:
            headers += (("Content-Encoding", "gzip"),)
        return UpstreamResponse(status=200, headers=headers, body=chunks)

    broker = TokenBroker(_startup(), transport=transport)
    decision = broker.handle(_request())
    rendered = _consume(decision)

    assert _ACCESS_TOKEN.encode() not in rendered
    assert _ACCOUNT_ID.encode() not in rendered
    if leak_location == "body":
        assert broker.receipt()["revoked"] is True
    else:
        assert decision.status == 502
    expected_reason = {
        "header": "upstream_secret_in_headers",
        "body": "upstream_secret_in_body",
        "compressed": "upstream_content_encoding",
    }[leak_location]
    receipt = broker.receipt()
    assert receipt["rejection_reasons"] == {expected_reason: 1}
    assert receipt["observed_content_types"] == (
        [] if leak_location == "header" else ["text/event-stream"]
    )


@pytest.mark.parametrize("secret", [_ACCESS_TOKEN, _ACCOUNT_ID, _CAPABILITY])
def test_broker_detects_each_secret_split_across_three_chunks_without_retry(secret):
    split_one = max(1, len(secret) // 3)
    split_two = max(split_one + 1, (2 * len(secret)) // 3)
    transport = _CaptureTransport(
        chunks=(
            b"data: harmless-prefix ",
            secret[:split_one].encode(),
            secret[split_one:split_two].encode(),
            secret[split_two:].encode(),
            _completed_sse(),
        )
    )
    broker = TokenBroker(_startup(), transport=transport)

    decision = broker.handle(_request())
    rendered = _consume(decision)

    assert decision.status == 502
    assert secret.encode() not in rendered
    assert b"harmless-prefix" not in rendered
    assert len(transport.calls) == 1
    assert transport.close_count == 1
    receipt = broker.receipt()
    assert receipt["upstream_attempts"] == 1
    assert receipt["stream_retries_used"] == 0
    assert receipt["upstream_stream_errors"] == {"UpstreamSecretInBody": 1}
    assert receipt["rejection_reasons"] == {"upstream_secret_in_body": 1}
    assert receipt["revoked"] is True


def test_real_transport_replaces_authority_headers_and_streams_http2_once():
    requests: list[dict[str, Any]] = []
    constructed = 0
    response_close_count = 0
    client_close_count = 0

    class FakeHeaders:
        def multi_items(self) -> list[tuple[str, str]]:
            return [("Content-Type", "text/event-stream")]

    class FakeResponse:
        status_code = 200
        http_version = "HTTP/2"
        headers = FakeHeaders()

        def iter_raw(self, *, chunk_size: int):
            assert chunk_size == 64 * 1024
            yield b"data: first\n\n"
            yield b"data: second\n\n"

        def close(self):
            nonlocal response_close_count
            response_close_count += 1

    class FakeClient:
        def __init__(self, timeout: float):
            nonlocal constructed
            constructed += 1
            assert timeout == 17

        def build_request(
            self,
            method: str,
            url: str,
            *,
            content: bytes,
            headers: Mapping[str, str],
        ) -> object:
            request = {
                "method": method,
                "url": url,
                "body": content,
                "headers": dict(headers),
            }
            requests.append(request)
            return request

        def send(self, _request: object, *, stream: bool) -> FakeResponse:
            assert stream is True
            return FakeResponse()

        def close(self):
            nonlocal client_close_count
            client_close_count += 1

    response = HttpUpstreamTransport(client_factory=FakeClient)(
        body=_body(),
        headers={
            "authorization": f"Bearer {_CAPABILITY}",
            "chatgpt-account-id": "spoofed-account",
            "x-lha-evaluation-id": "should-not-forward",
            "content-type": "application/json",
            "accept": "text/event-stream",
        },
        access_token=_ACCESS_TOKEN,
        account_id=_ACCOUNT_ID,
        timeout_s=17,
    )

    assert b"".join(response.body) == b"data: first\n\ndata: second\n\n"
    response.close()
    assert constructed == 1
    assert response_close_count == 1
    assert client_close_count == 1
    assert len(requests) == 1
    sent = requests[0]
    assert sent["method"] == "POST"
    assert sent["url"] == (
        f"https://{proxy.CHATGPT_UPSTREAM_HOST}{proxy.CHATGPT_RESPONSES_PATH}"
    )
    assert sent["headers"]["Authorization"] == f"Bearer {_ACCESS_TOKEN}"
    assert sent["headers"]["ChatGPT-Account-ID"] == _ACCOUNT_ID
    assert sent["headers"]["Content-Type"] == "application/json"
    assert sum(name.lower() == "content-type" for name in sent["headers"]) == 1
    assert sent["headers"]["Accept-Encoding"] == "identity"
    assert all(name.lower() != "connection" for name in sent["headers"])
    rendered = json.dumps(sent["headers"], sort_keys=True)
    assert _CAPABILITY not in rendered
    assert "spoofed-account" not in rendered
    assert "should-not-forward" not in rendered


def test_real_transport_does_not_retry_an_upstream_connection_error():
    attempts = 0
    close_count = 0

    class FailingClient:
        def build_request(self, *_args: Any, **_kwargs: Any) -> object:
            return object()

        def send(self, *_args: Any, **_kwargs: Any) -> None:
            nonlocal attempts
            attempts += 1
            raise OSError("simulated upstream failure")

        def close(self) -> None:
            nonlocal close_count
            close_count += 1
            raise OSError("cleanup failure must not mask send failure")

    with pytest.raises(OSError, match="simulated"):
        HttpUpstreamTransport(client_factory=lambda _timeout: FailingClient())(
            body=_body(),
            headers={"content-type": "application/json"},
            access_token=_ACCESS_TOKEN,
            account_id=_ACCOUNT_ID,
            timeout_s=17,
        )
    assert attempts == 1
    assert close_count == 1


def test_real_transport_configures_http2_without_environment_or_retries(monkeypatch):
    observed: dict[str, Any] = {}

    def transport_factory(**kwargs: Any) -> object:
        observed["transport"] = kwargs
        return object()

    def client_factory(**kwargs: Any) -> object:
        observed["client"] = kwargs
        return object()

    monkeypatch.setitem(
        sys.modules,
        "httpx",
        type(
            "FakeHttpx",
            (),
            {
                "HTTPTransport": staticmethod(transport_factory),
                "Client": staticmethod(client_factory),
            },
        )(),
    )

    client = HttpUpstreamTransport()._new_client(17)

    assert client is not None
    assert observed["transport"] == {
        "http1": False,
        "http2": True,
        "retries": 0,
        "trust_env": False,
    }
    assert observed["client"]["http2"] is True
    assert observed["client"]["trust_env"] is False
    assert observed["client"]["follow_redirects"] is False
    assert observed["client"]["timeout"] == 17
    assert observed["client"]["transport"] is not None


def test_http1_downgrade_is_a_fixed_transport_failure():
    response_close_count = 0
    client_close_count = 0

    class FakeHeaders:
        def multi_items(self) -> list[tuple[str, str]]:
            return []

    class Http1Response:
        status_code = 200
        http_version = "HTTP/1.1"
        headers = FakeHeaders()

        def close(self) -> None:
            nonlocal response_close_count
            response_close_count += 1

    class Http1Client:
        def build_request(self, *_args: Any, **_kwargs: Any) -> object:
            return object()

        def send(self, *_args: Any, **_kwargs: Any) -> Http1Response:
            return Http1Response()

        def close(self) -> None:
            nonlocal client_close_count
            client_close_count += 1

    broker = TokenBroker(
        _startup(),
        transport=HttpUpstreamTransport(
            client_factory=lambda _timeout: Http1Client(),
        ),
    )

    decision = broker.handle(_request())
    receipt = broker.receipt()

    assert decision.status == 502
    assert receipt["upstream_transport_errors"] == {
        "UpstreamHttpVersionError": 1
    }
    assert receipt["rejection_reasons"] == {"upstream_transport_exception": 1}
    assert receipt["revoked"] is True
    assert response_close_count == 1
    assert client_close_count == 1


class _LoopbackHTTPSConnection(HTTPSConnection):
    """Verify the broker DNS name while connecting to the local test socket."""

    def connect(self) -> None:
        raw_socket = socket.create_connection(("127.0.0.1", self.port), self.timeout)
        self.sock = self._context.wrap_socket(  # type: ignore[union-attr]
            raw_socket,
            server_hostname=proxy.BROKER_ALIAS,
        )


def _client_tls_context(certificate_pem: bytes) -> ssl.SSLContext:
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cadata=certificate_pem.decode("ascii"))
    return context


def test_https_server_preserves_sse_bytes_and_clears_startup_private_key():
    startup = replace(_startup(), source_ip="127.0.0.1")
    transport = _CaptureTransport()
    expected = b"".join(transport.chunks)
    broker = TokenBroker(startup, transport=transport)
    context = proxy_server._load_tls_context(startup)
    assert startup.tls_private_key_pem == bytearray()
    server = _BrokerServer(("127.0.0.1", 0), broker, context)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = _LoopbackHTTPSConnection(
            proxy.BROKER_ALIAS,
            server.server_port,
            timeout=5,
            context=_client_tls_context(startup.tls_certificate_pem),
        )
        connection.request(
            "POST",
            "/responses",
            body=_body(),
            headers={
                **_headers(startup),
                "Content-Length": str(len(_body())),
            },
        )
        response = connection.getresponse()
        payload = response.read()
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)

    assert response.status == 200
    assert response.getheader("Content-Type") == "text/event-stream"
    assert payload == expected


def test_loading_tls_context_removes_temporary_certificate_and_key(
    monkeypatch,
    tmp_path,
):
    startup = _startup()
    real_mkstemp = proxy_server.tempfile.mkstemp
    created_paths: list[str] = []

    def tracked_mkstemp(*, prefix: str, dir: str) -> tuple[int, str]:
        del dir
        descriptor, path = real_mkstemp(prefix=prefix, dir=tmp_path)
        created_paths.append(path)
        return descriptor, path

    monkeypatch.setattr(proxy_server.tempfile, "mkstemp", tracked_mkstemp)

    context = proxy_server._load_tls_context(startup)

    assert isinstance(context, ssl.SSLContext)
    assert startup.tls_private_key_pem == bytearray()
    assert len(created_paths) == 2
    assert list(tmp_path.iterdir()) == []


def test_tls_server_rejects_plaintext_and_a_different_self_signed_certificate():
    startup = replace(_startup(), source_ip="127.0.0.1")
    broker = TokenBroker(startup, transport=_CaptureTransport())
    context = proxy_server._load_tls_context(startup)
    server = _BrokerServer(("127.0.0.1", 0), broker, context)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        plaintext = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        with pytest.raises((ConnectionError, OSError)):
            plaintext.request("POST", "/responses", body=b"{}")
            plaintext.getresponse()
        plaintext.close()

        other_certificate, _other_key, _other_digest = proxy._generate_broker_tls_material()
        wrong_certificate = _LoopbackHTTPSConnection(
            proxy.BROKER_ALIAS,
            server.server_port,
            timeout=5,
            context=_client_tls_context(other_certificate),
        )
        with pytest.raises(ssl.SSLCertVerificationError):
            wrong_certificate.request("POST", "/responses", body=b"{}")
        wrong_certificate.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


class _ShutdownTestServer:
    def __init__(self) -> None:
        self.shutdown_calls = 0
        self.shutdown_called = threading.Event()

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.shutdown_called.set()


class _ShutdownTestBroker:
    def __init__(self) -> None:
        self.revoke_calls = 0

    def revoke(self) -> None:
        self.revoke_calls += 1


def test_shutdown_coordinator_allows_exactly_one_racing_outcome():
    server = _ShutdownTestServer()
    broker = _ShutdownTestBroker()
    coordinator = _ShutdownCoordinator(server, broker)  # type: ignore[arg-type]
    barrier = threading.Barrier(3)
    results: list[bool] = []

    def request(outcome: str) -> None:
        barrier.wait()
        results.append(coordinator.request(outcome))

    watchdog = threading.Thread(target=request, args=("ttl_expired",))
    sigterm = threading.Thread(target=request, args=("sigterm",))
    watchdog.start()
    sigterm.start()
    barrier.wait()
    watchdog.join(5)
    sigterm.join(5)

    assert sorted(results) == [False, True]
    assert coordinator.outcome in {"ttl_expired", "sigterm"}
    assert server.shutdown_called.wait(5)
    assert server.shutdown_calls == 1
    assert broker.revoke_calls == 1


def _exercise_server_main(
    monkeypatch: pytest.MonkeyPatch,
    *,
    watchdog_fires: bool,
    stop_signal: int | None,
) -> tuple[list[dict[str, Any]], Any, Any]:
    startup = _startup()
    handlers: dict[int, Any] = {}
    instances: dict[str, Any] = {}

    class FakeBroker:
        def __init__(self, received: BrokerStartup) -> None:
            assert received.evaluation_id == startup.evaluation_id
            assert received.attempt_id == startup.attempt_id
            assert received.tls_certificate_pem == startup.tls_certificate_pem
            assert received.tls_certificate_sha256 == startup.tls_certificate_sha256
            assert received.tls_private_key_pem == bytearray()
            self.revoke_calls = 0
            instances["broker"] = self

        def revoke(self) -> None:
            self.revoke_calls += 1

        def receipt(self, *, outcome: str) -> dict[str, object]:
            return {
                "schema_version": 1,
                "type": "terminal_proxy_receipt",
                "outcome": outcome,
                "revoked": self.revoke_calls > 0,
            }

    class FakeServer:
        def __init__(
            self,
            address: tuple[str, int],
            broker: Any,
            tls_context: ssl.SSLContext,
        ) -> None:
            assert address == ("0.0.0.0", proxy.BROKER_PORT)
            assert broker is instances["broker"]
            assert isinstance(tls_context, ssl.SSLContext)
            self.certificate_sha256 = startup.tls_certificate_sha256
            self.shutdown_calls = 0
            self.closed = False
            self.shutdown_called = threading.Event()
            instances["server"] = self

        def serve_forever(self, *, poll_interval: float) -> None:
            assert poll_interval == 0.1
            if stop_signal is not None:
                handlers[stop_signal](stop_signal, None)
            assert self.shutdown_called.wait(5)

        def shutdown(self) -> None:
            self.shutdown_calls += 1
            self.shutdown_called.set()

        def server_close(self) -> None:
            self.closed = True

    class FakeTimer:
        def __init__(self, interval: float, function: Any, args: tuple[str, ...]) -> None:
            self.interval = interval
            self.function = function
            self.args = args
            self.daemon = False
            self.cancelled = False
            self.joined = False
            instances["timer"] = self

        def start(self) -> None:
            if watchdog_fires:
                self.function(*self.args)

        def cancel(self) -> None:
            self.cancelled = True

        def join(self) -> None:
            self.joined = True

    monkeypatch.setattr(proxy_server, "TokenBroker", FakeBroker)
    monkeypatch.setattr(proxy_server, "_BrokerServer", FakeServer)
    monkeypatch.setattr(proxy_server.threading, "Timer", FakeTimer)
    monkeypatch.setattr(proxy_server.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(
        proxy_server.signal,
        "signal",
        lambda signum, handler: handlers.__setitem__(signum, handler),
    )
    monkeypatch.setattr(proxy_server.sys, "stdin", io.StringIO(startup.stdin_json() + "\n"))

    assert proxy_server.main() == 0
    output = [
        json.loads(line)
        for line in proxy_server.sys.stdout.getvalue().splitlines()  # type: ignore[attr-defined]
    ]
    return output, instances["server"], instances["timer"]


def test_watchdog_expires_at_startup_ttl_and_emits_ttl_expired(monkeypatch, capsys):
    monkeypatch.setattr(proxy_server.sys, "stdout", io.StringIO())

    output, server, timer = _exercise_server_main(
        monkeypatch,
        watchdog_fires=True,
        stop_signal=None,
    )

    assert timer.interval == BROKER_TTL_S
    assert timer.cancelled is True
    assert timer.joined is True
    assert server.shutdown_calls == 1
    assert server.closed is True
    assert output[0]["schema_version"] == 4
    assert output[0]["request_retry_limit"] == 1
    assert output[0]["stream_retry_limit"] == BROKER_STREAM_RETRY_LIMIT
    assert (
        output[0]["stream_retry_limit_per_request"]
        == BROKER_STREAM_RETRY_LIMIT_PER_REQUEST
    )
    assert output[0]["tls_certificate_sha256"] == server.certificate_sha256
    assert output[1]["outcome"] == "ttl_expired"
    assert output[1]["revoked"] is True
    capsys.readouterr()


def test_sigterm_wins_over_idle_watchdog_and_receipt_remains_sigterm(monkeypatch, capsys):
    monkeypatch.setattr(proxy_server.sys, "stdout", io.StringIO())

    output, server, _timer = _exercise_server_main(
        monkeypatch,
        watchdog_fires=False,
        stop_signal=signal.SIGTERM,
    )

    assert server.shutdown_calls == 1
    assert output[1]["outcome"] == "sigterm"
    capsys.readouterr()


class _InputCapture:
    def __init__(self) -> None:
        self.payload = ""

    def write(self, value: str) -> int:
        self.payload += value
        return len(value)

    def flush(self) -> None:
        return

    def close(self) -> None:
        return


class _FakeProcess:
    def __init__(self, argv: list[str], stdout_text: str):
        self.args = argv
        self.stdin = _InputCapture()
        self.stdout = io.StringIO(stdout_text)
        self.stderr = io.StringIO("")
        self.returncode: int | None = None
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def _container_inspection(
    *,
    container_id: str,
    name: str,
    image_id: str,
    network: str,
    ip: str,
    broker: bool,
    evaluation_id: str = "tb21-fixed-20",
    attempt_id: str = "regex-log-01",
) -> dict[str, Any]:
    host: dict[str, Any] = {
        "NetworkMode": network,
        "ReadonlyRootfs": broker,
        "Privileged": False,
        "PortBindings": None,
        "Binds": None,
        "CapDrop": ["ALL"] if broker else None,
        "SecurityOpt": ["no-new-privileges"] if broker else None,
    }
    labels = (
        {
            "lha.terminal.role": "broker",
            "lha.terminal.evaluation_id": evaluation_id,
            "lha.terminal.attempt_id": attempt_id,
        }
        if broker
        else {
            "com.docker.compose.service": "main",
            "com.docker.compose.project": "harbor_trial",
        }
    )
    return {
        "Id": container_id,
        "Name": f"/{name}",
        "Image": image_id,
        "State": {"Running": True},
        "Config": {
            "Env": ["PATH=/usr/local/bin:/usr/bin"],
            "Image": image_id,
            "Labels": labels,
        },
        "HostConfig": host,
        "Mounts": [],
        "NetworkSettings": {
            "Networks": {
                network: {
                    "IPAddress": ip,
                }
            }
        },
    }


@pytest.mark.parametrize("receipt_outcome", ["sigterm", "ttl_expired"])
def test_controller_uses_stdin_only_attests_network_and_confirms_deletion(
    monkeypatch,
    receipt_outcome,
):
    evaluation_id = "tb21-fixed-20"
    attempt_id = "regex-log-01"
    network = "harbor_trial_default"
    broker_name = TerminalProxyController._name(evaluation_id, attempt_id)
    source = _container_inspection(
        container_id=_SOURCE_CONTAINER,
        name="main",
        image_id=f"sha256:{'e' * 64}",
        network=network,
        ip="172.30.0.4",
        broker=False,
    )
    broker_container_id = "f" * 64
    broker_inspection = _container_inspection(
        container_id=broker_container_id,
        name=broker_name,
        image_id=_IMAGE_ID,
        network=network,
        ip="172.30.0.5",
        broker=True,
    )
    removed = False
    commands: list[list[str]] = []
    certificate_pem, private_key_pem, certificate_sha256 = proxy._generate_broker_tls_material()
    private_key_text = bytes(private_key_pem).decode("ascii")
    monkeypatch.setattr(
        proxy,
        "_generate_broker_tls_material",
        lambda: (certificate_pem, bytearray(private_key_pem), certificate_sha256),
    )

    def completed(
        argv: list[str], returncode: int, stdout: str = "", stderr: str = ""
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    def run_command(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal removed
        commands.append(argv)
        if argv[1:4] == ["inspect", "--type", "image"]:
            return completed(
                argv,
                0,
                json.dumps(
                    [{"Id": _IMAGE_ID, "Os": "linux", "Architecture": "amd64"}]
                ),
            )
        if argv[1:4] == ["inspect", "--type", "container"]:
            target = argv[4]
            if target == _SOURCE_CONTAINER:
                return completed(argv, 0, json.dumps([source]))
            if target == broker_name:
                if removed:
                    return completed(argv, 1, stderr=f"Error: No such object: {target}")
                # The first probe proves the deterministic name was unused.
                previous = [item for item in commands[:-1] if item[-1] == broker_name]
                if not previous:
                    return completed(argv, 1, stderr=f"Error: No such object: {target}")
                return completed(argv, 0, json.dumps([broker_inspection]))
        if argv[1] == "stop":
            return completed(argv, 0, broker_name)
        if argv[1:3] == ["rm", "-f"]:
            removed = True
            return completed(argv, 0, broker_name)
        raise AssertionError(f"unexpected Docker command: {argv}")

    ready = {
        "schema_version": 4,
        "type": "terminal_proxy_ready",
        "evaluation_id": evaluation_id,
        "attempt_id": attempt_id,
        "port": proxy.BROKER_PORT,
        "ttl_s": BROKER_TTL_S,
        "max_requests": BROKER_MAX_REQUESTS,
        "request_retry_limit": 1,
        "stream_retry_limit": BROKER_STREAM_RETRY_LIMIT,
        "stream_retry_limit_per_request": BROKER_STREAM_RETRY_LIMIT_PER_REQUEST,
        "tls_certificate_sha256": certificate_sha256,
    }
    receipt = {
        "schema_version": 5,
        "type": "terminal_proxy_receipt",
        "evaluation_id": evaluation_id,
        "attempt_id": attempt_id,
        "source_container_id": _SOURCE_CONTAINER,
        "started_at": "2026-07-27T00:00:00+00:00",
        "stopped_at": "2026-07-27T00:01:00+00:00",
        "ttl_s": BROKER_TTL_S,
        "max_requests": BROKER_MAX_REQUESTS,
        "max_buffered_response_bytes": proxy.BROKER_MAX_BUFFERED_RESPONSE_BYTES,
        "request_retry_limit": 1,
        "stream_retry_limit": BROKER_STREAM_RETRY_LIMIT,
        "stream_retry_limit_per_request": BROKER_STREAM_RETRY_LIMIT_PER_REQUEST,
        "downstream_accepted_requests": 1,
        "rejected_requests": 0,
        "rejection_reasons": {},
        "upstream_attempts": 1,
        "upstream_statuses": {"200": 1},
        "stream_retries_used": 0,
        "stream_retried_requests": 0,
        "max_stream_retries_on_request": 0,
        "upstream_error": None,
        "upstream_transport_errors": {},
        "upstream_stream_errors": {},
        "observed_content_types": ["text/event-stream"],
        "revoked": True,
        "outcome": receipt_outcome,
    }
    process_holder: list[_FakeProcess] = []

    def popen_factory(argv: list[str], **_kwargs: Any) -> Any:
        process = _FakeProcess(
            argv,
            json.dumps(ready, separators=(",", ":"))
            + "\n"
            + json.dumps(receipt, separators=(",", ":"))
            + "\n",
        )
        process_holder.append(process)
        return process

    controller = TerminalProxyController(
        image_id=_IMAGE_ID,
        run_command=run_command,
        popen_factory=popen_factory,
    )
    credentials = BrokerSecrets(_ACCESS_TOKEN, _ACCOUNT_ID)
    handle = controller.start(
        evaluation_id=evaluation_id,
        attempt_id=attempt_id,
        source_container_id=_SOURCE_CONTAINER,
        network=network,
        model="gpt-5.6-sol",
        reasoning_effort="max",
        credentials=credentials,
    )

    process = process_holder[0]
    argv_text = "\n".join(process.args)
    assert process.args[-1] == _IMAGE_ID
    assert process.args[process.args.index("--platform") + 1] == "linux/amd64"
    assert "--network" in process.args
    assert process.args[process.args.index("--network") + 1] == network
    assert "--read-only" in process.args
    assert "--cap-drop" in process.args
    assert "--security-opt" in process.args
    assert not {"-v", "--volume", "-p", "--publish", "--env", "-e"} & set(process.args)
    assert "/var/run/docker.sock" not in argv_text
    assert _ACCESS_TOKEN not in argv_text
    assert _ACCOUNT_ID not in argv_text
    assert _ACCESS_TOKEN in process.stdin.payload
    assert _ACCOUNT_ID in process.stdin.payload
    assert json.loads(process.stdin.payload)["tls_private_key_pem"] == private_key_text
    assert private_key_text not in argv_text
    assert private_key_text not in json.dumps(broker_inspection)
    assert private_key_text not in json.dumps(ready)
    assert private_key_text not in json.dumps(receipt)
    assert _ACCESS_TOKEN not in repr(handle)
    assert _ACCOUNT_ID not in repr(handle)
    assert private_key_text not in repr(handle)
    assert handle.base_url == f"https://{proxy.BROKER_ALIAS}:{proxy.BROKER_PORT}"
    assert handle.tls_certificate_pem == certificate_pem
    assert handle.tls_certificate_sha256 == certificate_sha256
    client_headers = handle.client_headers()
    assert client_headers["X-LHA-Container-ID"] == _SOURCE_CONTAINER
    capability = client_headers["Authorization"].removeprefix("Bearer ")
    assert capability not in repr(handle)
    assert handle.binding_headers() == {
        "X-LHA-Evaluation-ID": evaluation_id,
        "X-LHA-Attempt-ID": attempt_id,
        "X-LHA-Container-ID": _SOURCE_CONTAINER,
    }
    assert handle.capability_environment() == {
        proxy.CAPABILITY_ENV: capability,
    }

    if receipt_outcome == "sigterm":
        final_receipt = controller.stop(handle)
        assert final_receipt == receipt
    else:
        with pytest.raises(TerminalProxyError, match="shutdown receipt"):
            controller.stop(handle)
    assert removed
    assert capability not in json.dumps(receipt)
    assert _ACCESS_TOKEN not in json.dumps(receipt)
    assert _ACCOUNT_ID not in json.dumps(receipt)
    with pytest.raises(TerminalProxyError, match="no longer available"):
        handle.client_headers()
    with pytest.raises(TerminalProxyError, match="no longer available"):
        handle.capability_environment()
    assert commands[-1][1:4] == ["inspect", "--type", "container"]


def test_controller_fails_closed_when_source_has_an_extra_network():
    network = "harbor_trial_default"
    source = _container_inspection(
        container_id=_SOURCE_CONTAINER,
        name="main",
        image_id=f"sha256:{'e' * 64}",
        network=network,
        ip="172.30.0.4",
        broker=False,
    )
    source["NetworkSettings"]["Networks"]["unexpected"] = {"IPAddress": "172.31.0.4"}

    def run_command(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if argv[1:4] == ["inspect", "--type", "image"]:
            return subprocess.CompletedProcess(
                argv,
                0,
                json.dumps(
                    [{"Id": _IMAGE_ID, "Os": "linux", "Architecture": "amd64"}]
                ),
                "",
            )
        return subprocess.CompletedProcess(argv, 0, json.dumps([source]), "")

    controller = TerminalProxyController(image_id=_IMAGE_ID, run_command=run_command)
    with pytest.raises(TerminalProxyError, match="additional network"):
        controller.start(
            evaluation_id="tb21-fixed-20",
            attempt_id="regex-log-01",
            source_container_id=_SOURCE_CONTAINER,
            network=network,
            model="gpt-5.6-sol",
            reasoning_effort="max",
            credentials=BrokerSecrets(_ACCESS_TOKEN, _ACCOUNT_ID),
        )


@pytest.mark.parametrize(
    ("returncode", "stderr"),
    [
        (0, ""),
        (1, "Cannot connect to the Docker daemon"),
    ],
)
def test_controller_does_not_mistake_a_live_or_uninspectable_container_for_deleted(
    returncode, stderr
):
    def run_command(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, returncode, "", stderr)

    controller = TerminalProxyController(image_id=_IMAGE_ID, run_command=run_command)
    with pytest.raises(TerminalProxyError, match="deletion could not be confirmed"):
        controller._confirm_removed("lha-terminal-proxy-test")


def test_abandoned_cleanup_attests_and_deletes_exact_broker_and_task_containers():
    evaluation_id = "tb21-fixed-20"
    attempt_id = "regex-log-01"
    network = "harbor_trial_default"
    broker_name = TerminalProxyController._name(evaluation_id, attempt_id)
    broker_id = "f" * 64
    task_id = _SOURCE_CONTAINER
    task_image_id = f"sha256:{'e' * 64}"
    task_digest = f"sha256:{'a' * 64}"
    broker = _container_inspection(
        container_id=broker_id,
        name=broker_name,
        image_id=_IMAGE_ID,
        network=network,
        ip="172.30.0.5",
        broker=True,
        evaluation_id=evaluation_id,
        attempt_id=attempt_id,
    )
    task = _container_inspection(
        container_id=task_id,
        name="main",
        image_id=task_image_id,
        network=network,
        ip="172.30.0.4",
        broker=False,
    )
    task["Config"]["Image"] = f"registry.example/task@{task_digest}"
    present = {broker_id, task_id}
    commands: list[list[str]] = []

    def completed(
        argv: list[str],
        returncode: int,
        stdout: str = "",
        stderr: str = "",
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, returncode, stdout, stderr)

    def run_command(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(argv)
        if argv[1:4] == ["inspect", "--type", "container"]:
            target = argv[4]
            if target in {broker_name, broker_id}:
                value = broker if broker_id in present else None
            elif target == task_id:
                value = task if task_id in present else None
            else:
                value = None
            if value is None:
                return completed(argv, 1, stderr=f"Error: No such object: {target}")
            return completed(argv, 0, json.dumps([value]))
        if argv[1:4] == ["inspect", "--type", "image"]:
            assert argv[4] == task_image_id
            return completed(
                argv,
                0,
                json.dumps(
                    [
                        {
                            "Id": task_image_id,
                            "RepoDigests": [
                                f"registry.example/task@{task_digest}"
                            ],
                        }
                    ]
                ),
            )
        if argv[1:3] == ["rm", "-f"]:
            present.discard(argv[3])
            return completed(argv, 0, argv[3])
        raise AssertionError(f"unexpected Docker command: {argv}")

    TerminalProxyController(
        image_id=_IMAGE_ID,
        run_command=run_command,
    ).cleanup_abandoned(
        evaluation_id=evaluation_id,
        attempt_id=attempt_id,
        source_container_id=task_id,
        expected_task_image_digest=task_digest,
    )

    assert present == set()
    assert [command for command in commands if command[1:3] == ["rm", "-f"]] == [
        ["docker", "rm", "-f", broker_id],
        ["docker", "rm", "-f", task_id],
    ]
    assert commands[-1] == [
        "docker",
        "inspect",
        "--type",
        "container",
        task_id,
    ]


def test_abandoned_cleanup_refuses_unattested_task_without_deleting_it():
    task_id = _SOURCE_CONTAINER
    task = _container_inspection(
        container_id=task_id,
        name="main",
        image_id=f"sha256:{'e' * 64}",
        network="harbor_trial_default",
        ip="172.30.0.4",
        broker=False,
    )
    task["Config"]["Labels"]["com.docker.compose.service"] = "sidecar"
    removed: list[str] = []

    def run_command(argv: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        if argv[1:4] == ["inspect", "--type", "container"]:
            if argv[4].startswith("lha-terminal-proxy-"):
                return subprocess.CompletedProcess(
                    argv,
                    1,
                    "",
                    "Error: No such object",
                )
            return subprocess.CompletedProcess(argv, 0, json.dumps([task]), "")
        if argv[1:3] == ["rm", "-f"]:
            removed.append(argv[3])
            return subprocess.CompletedProcess(argv, 0, argv[3], "")
        raise AssertionError(f"unexpected Docker command: {argv}")

    with pytest.raises(TerminalProxyError, match="cleanup could not be proven"):
        TerminalProxyController(
            image_id=_IMAGE_ID,
            run_command=run_command,
        ).cleanup_abandoned(
            evaluation_id="tb21-fixed-20",
            attempt_id="regex-log-01",
            source_container_id=task_id,
            expected_task_image_digest=f"sha256:{'a' * 64}",
        )
    assert removed == []


@pytest.mark.parametrize(
    "image_id",
    [
        "lha-terminal-proxy:latest",
        "sha256:short",
        f"sha256:{'A' * 64}",
    ],
)
def test_controller_requires_a_full_lowercase_image_id(image_id):
    with pytest.raises(ValueError, match="pinned"):
        TerminalProxyController(image_id=image_id)


def test_broker_image_command_does_not_pull_or_use_an_unpinned_runtime_tag():
    command = proxy.broker_image_build_command(tag="lha-terminal-proxy:formal")
    assert command == [
        "docker",
        "build",
        "--pull=false",
        "--platform",
        "linux/amd64",
        "--file",
        "docker/terminal-bench-proxy.Dockerfile",
        "--tag",
        "lha-terminal-proxy:formal",
        ".",
    ]
