"""Credential broker used by the direct Harbor Terminal-Bench adapter.

The benchmark container receives a short-lived capability, never the real
ChatGPT credentials.  This module has two deliberately separate parts:

* :class:`TokenBroker` validates and forwards Responses API requests.
* :class:`TerminalProxyController` starts that broker as a locked-down sibling
  container on the Harbor trial network.

The broker image has no host port, mount, Docker socket, or credential-bearing
environment variable.  Its startup record, including the upstream bearer token
and account id, is delivered once over the container's standard input.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import queue
import re
import secrets
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import IO, Any, Protocol, cast

from lha.clock import now

BROKER_PORT = 8080
BROKER_TTL_S = 2100
BROKER_MAX_REQUESTS = 60
BROKER_REQUEST_RETRY_LIMIT = 1
BROKER_STREAM_RETRY_LIMIT = 12
BROKER_STREAM_RETRY_LIMIT_PER_REQUEST = 4
BROKER_MAX_BUFFERED_RESPONSE_BYTES = 16 * 1024 * 1024
BROKER_MAX_OBSERVED_CONTENT_TYPES = 4
BROKER_MAX_OBSERVED_CONTENT_TYPE_CHARS = 256
BROKER_ALIAS = "lha-terminal-proxy"
CAPABILITY_ENV = "LHA_TERMINAL_PROXY_CAPABILITY"
CHATGPT_UPSTREAM_HOST = "chatgpt.com"
CHATGPT_RESPONSES_PATH = "/backend-api/codex/responses"

_SHA256_IMAGE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONTAINER_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_NETWORK_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_CAPABILITY_RE = re.compile(r"^[A-Za-z0-9_-]{32,256}$")
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_ERROR_TYPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.]{0,127}$")
_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
BROKER_REJECTION_REASONS = frozenset(
    {
        "binding_mismatch",
        "capability_revoked",
        "content_type_not_allowed",
        "effort_mismatch",
        "invalid_capability",
        "invalid_json",
        "invalid_request",
        "model_mismatch",
        "request_limit_reached",
        "request_too_large",
        "response_after_revocation",
        "route_not_allowed",
        "source_mismatch",
        "upstream_content_encoding",
        "upstream_content_type_invalid",
        "upstream_header_shape",
        "upstream_invalid_error_body",
        "upstream_invalid_status",
        "upstream_invalid_sse",
        "upstream_non_bytes_chunk",
        "upstream_response_too_large",
        "upstream_retry_failed",
        "upstream_secret_in_body",
        "upstream_secret_in_error_body",
        "upstream_secret_in_headers",
        "upstream_stream_failure",
        "upstream_attempt_timeout",
        "upstream_timeout",
        "upstream_transport_exception",
    }
)
BROKER_RECOVERABLE_TRANSPORT_ERRORS = frozenset(
    {
        "ConnectError",
        "ConnectTimeout",
        "NetworkError",
        "PoolTimeout",
        "ProxyError",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutException",
        "WriteError",
        "WriteTimeout",
    }
)
BROKER_RECOVERABLE_STREAM_ERRORS = frozenset(
    {
        "MissingResponseCompleted",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
    }
)
_FINAL_STREAM_REJECTION_REASONS = frozenset(
    {
        "upstream_invalid_sse",
        "upstream_non_bytes_chunk",
        "upstream_response_too_large",
        "upstream_secret_in_body",
        "upstream_stream_failure",
        "upstream_timeout",
    }
)
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_UPSTREAM_REPLACED_HEADERS = frozenset(
    {
        "accept-encoding",
        "authorization",
        "chatgpt-account-id",
        "content-type",
        "set-cookie",
    }
)
_CLIENT_STRIPPED_HEADERS = frozenset(
    {
        "authorization",
        "chatgpt-account-id",
        "set-cookie",
    }
)
_BINDING_HEADERS = frozenset(
    {
        "x-lha-evaluation-id",
        "x-lha-attempt-id",
        "x-lha-container-id",
    }
)
_MAX_REQUEST_BYTES = 32 * 1024 * 1024
_MAX_ERROR_RESPONSE_BYTES = 64 * 1024
_MAX_ERROR_TEXT_CHARS = 512
_MAX_SSE_FRAME_BYTES = 1024 * 1024
_OBSERVED_CONTENT_TYPES_OMITTED = "<additional-content-types-omitted>"
_DOCKER_TIMEOUT_S = 30.0
_STOP_TIMEOUT_S = 15.0


class TerminalProxyError(RuntimeError):
    """Fail-closed proxy or container-controller error."""


class UpstreamHttpVersionError(RuntimeError):
    """The upstream connection did not negotiate the required HTTP version."""


class MissingResponseCompleted(RuntimeError):
    """A successful upstream stream ended without its terminal SSE event."""


class UpstreamResponseTooLarge(RuntimeError):
    """A successful upstream stream exceeded the fixed in-memory response bound."""


class UpstreamSecretInBody(RuntimeError):
    """An upstream response contained broker-held secret material."""


class UpstreamNonBytesChunk(RuntimeError):
    """An upstream transport yielded a value outside its bytes contract."""


class UpstreamDeadlineExceeded(RuntimeError):
    """The attempt deadline expired while buffering an upstream response."""


def _safe_identifier(value: str, *, label: str) -> str:
    checked = value.strip()
    if _SAFE_ID_RE.fullmatch(checked) is None:
        raise ValueError(f"{label} has an invalid format")
    return checked


def _canonical_ip(value: str) -> str:
    try:
        return str(ipaddress.ip_address(value))
    except ValueError as exc:
        raise ValueError("source_ip must be a canonical IP address") from exc


@dataclass(frozen=True, repr=False)
class BrokerSecrets:
    """Secrets supplied only through broker-container stdin."""

    access_token: str
    account_id: str

    def __post_init__(self) -> None:
        for label, value in (
            ("access_token", self.access_token),
            ("account_id", self.account_id),
        ):
            if not value or len(value) > 16_384 or any(ord(char) < 32 for char in value):
                raise ValueError(f"{label} is invalid")

    def __repr__(self) -> str:
        return "BrokerSecrets(**redacted**)"


def _generate_broker_tls_material() -> tuple[bytes, bytearray, str]:
    """Create one short-lived certificate for the broker's private-network alias."""
    # Imported here because the broker image only needs the stdlib TLS server. Certificate
    # generation happens on the controller and is an explicit LHA runtime dependency.
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    private_key = ec.generate_private_key(ec.SECP256R1())
    issued_at = now()
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, BROKER_ALIAS)])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(issued_at - timedelta(minutes=5))
        .not_valid_after(issued_at + timedelta(seconds=BROKER_TTL_S + 300))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName(BROKER_ALIAS)]),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )
    certificate_pem = certificate.public_bytes(serialization.Encoding.PEM)
    private_key_pem = bytearray(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return (
        certificate_pem,
        private_key_pem,
        hashlib.sha256(certificate_pem).hexdigest(),
    )


@dataclass(frozen=True, repr=False)
class BrokerStartup:
    """One attempt's broker policy, credentials, and consumable TLS key."""

    evaluation_id: str
    attempt_id: str
    source_container_id: str
    source_ip: str
    model: str
    reasoning_effort: str
    capability: str
    credentials: BrokerSecrets
    tls_certificate_pem: bytes
    tls_certificate_sha256: str
    tls_private_key_pem: bytearray = field(repr=False)
    ttl_s: int = BROKER_TTL_S
    max_requests: int = BROKER_MAX_REQUESTS
    request_retry_limit: int = BROKER_REQUEST_RETRY_LIMIT
    stream_retry_limit: int = BROKER_STREAM_RETRY_LIMIT
    stream_retry_limit_per_request: int = BROKER_STREAM_RETRY_LIMIT_PER_REQUEST

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluation_id",
            _safe_identifier(self.evaluation_id, label="evaluation_id"),
        )
        object.__setattr__(
            self,
            "attempt_id",
            _safe_identifier(self.attempt_id, label="attempt_id"),
        )
        if _CONTAINER_ID_RE.fullmatch(self.source_container_id) is None:
            raise ValueError("source_container_id must be a full Docker container id")
        object.__setattr__(self, "source_ip", _canonical_ip(self.source_ip))
        if not self.model.strip() or len(self.model) > 128:
            raise ValueError("model is invalid")
        if not self.reasoning_effort.strip() or len(self.reasoning_effort) > 32:
            raise ValueError("reasoning_effort is invalid")
        object.__setattr__(self, "model", self.model.strip())
        object.__setattr__(self, "reasoning_effort", self.reasoning_effort.strip())
        if _CAPABILITY_RE.fullmatch(self.capability) is None:
            raise ValueError("capability has an invalid format")
        if (
            not isinstance(self.tls_certificate_pem, bytes)
            or not self.tls_certificate_pem.startswith(b"-----BEGIN CERTIFICATE-----\n")
            or not self.tls_certificate_pem.endswith(b"-----END CERTIFICATE-----\n")
            or len(self.tls_certificate_pem) > 64 * 1024
        ):
            raise ValueError("TLS certificate is invalid")
        if (
            not isinstance(self.tls_private_key_pem, bytearray)
            or not self.tls_private_key_pem.startswith(b"-----BEGIN PRIVATE KEY-----\n")
            or not self.tls_private_key_pem.endswith(b"-----END PRIVATE KEY-----\n")
            or len(self.tls_private_key_pem) > 64 * 1024
        ):
            raise ValueError("TLS private key is invalid")
        if _SHA256_HEX_RE.fullmatch(
            self.tls_certificate_sha256
        ) is None or not secrets.compare_digest(
            self.tls_certificate_sha256,
            hashlib.sha256(self.tls_certificate_pem).hexdigest(),
        ):
            raise ValueError("TLS certificate digest does not match")
        if self.ttl_s != BROKER_TTL_S:
            raise ValueError(f"ttl_s must be exactly {BROKER_TTL_S}")
        if self.max_requests != BROKER_MAX_REQUESTS:
            raise ValueError(f"max_requests must be exactly {BROKER_MAX_REQUESTS}")
        if self.request_retry_limit != BROKER_REQUEST_RETRY_LIMIT:
            raise ValueError(
                f"request_retry_limit must be exactly {BROKER_REQUEST_RETRY_LIMIT}"
            )
        if self.stream_retry_limit != BROKER_STREAM_RETRY_LIMIT:
            raise ValueError(
                f"stream_retry_limit must be exactly {BROKER_STREAM_RETRY_LIMIT}"
            )
        if (
            self.stream_retry_limit_per_request
            != BROKER_STREAM_RETRY_LIMIT_PER_REQUEST
        ):
            raise ValueError(
                "stream_retry_limit_per_request must be exactly "
                f"{BROKER_STREAM_RETRY_LIMIT_PER_REQUEST}"
            )

    def __repr__(self) -> str:
        return (
            "BrokerStartup("
            f"evaluation_id={self.evaluation_id!r}, "
            f"attempt_id={self.attempt_id!r}, "
            f"source_container_id={self.source_container_id!r}, "
            f"source_ip={self.source_ip!r}, "
            f"model={self.model!r}, "
            f"reasoning_effort={self.reasoning_effort!r}, "
            "capability=<redacted>, credentials=<redacted>, tls_private_key=<redacted>, "
            f"tls_certificate_sha256={self.tls_certificate_sha256!r}, "
            f"ttl_s={self.ttl_s}, max_requests={self.max_requests}, "
            f"request_retry_limit={self.request_retry_limit}, "
            f"stream_retry_limit={self.stream_retry_limit}, "
            "stream_retry_limit_per_request="
            f"{self.stream_retry_limit_per_request})"
        )

    def clear_tls_private_key(self) -> None:
        """Overwrite and release the startup key after the TLS context has loaded it."""
        for index in range(len(self.tls_private_key_pem)):
            self.tls_private_key_pem[index] = 0
        self.tls_private_key_pem.clear()

    def stdin_json(self) -> str:
        """Serialize the one record that is written to broker stdin."""
        if not self.tls_private_key_pem:
            raise ValueError("TLS private key was already consumed")
        return json.dumps(
            {
                "schema_version": 4,
                "evaluation_id": self.evaluation_id,
                "attempt_id": self.attempt_id,
                "source_container_id": self.source_container_id,
                "source_ip": self.source_ip,
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "capability": self.capability,
                "access_token": self.credentials.access_token,
                "account_id": self.credentials.account_id,
                "tls_certificate_pem": self.tls_certificate_pem.decode("ascii"),
                "tls_certificate_sha256": self.tls_certificate_sha256,
                "tls_private_key_pem": bytes(self.tls_private_key_pem).decode("ascii"),
                "ttl_s": self.ttl_s,
                "max_requests": self.max_requests,
                "request_retry_limit": self.request_retry_limit,
                "stream_retry_limit": self.stream_retry_limit,
                "stream_retry_limit_per_request": (
                    self.stream_retry_limit_per_request
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_stdin_json(cls, payload: str) -> "BrokerStartup":
        """Parse stdin without accepting misspelled or extra policy fields."""
        try:
            raw = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("broker startup record is not valid JSON") from exc
        if not isinstance(raw, dict):
            raise ValueError("broker startup record must be an object")
        expected = {
            "schema_version",
            "evaluation_id",
            "attempt_id",
            "source_container_id",
            "source_ip",
            "model",
            "reasoning_effort",
            "capability",
            "access_token",
            "account_id",
            "tls_certificate_pem",
            "tls_certificate_sha256",
            "tls_private_key_pem",
            "ttl_s",
            "max_requests",
            "request_retry_limit",
            "stream_retry_limit",
            "stream_retry_limit_per_request",
        }
        if set(raw) != expected or raw.get("schema_version") != 4:
            raise ValueError("broker startup record has an unsupported schema")
        string_fields = (
            "evaluation_id",
            "attempt_id",
            "source_container_id",
            "source_ip",
            "model",
            "reasoning_effort",
            "capability",
            "access_token",
            "account_id",
            "tls_certificate_pem",
            "tls_certificate_sha256",
            "tls_private_key_pem",
        )
        if any(not isinstance(raw.get(field_name), str) for field_name in string_fields):
            raise ValueError("broker startup record contains a non-string field")
        if (
            not isinstance(raw.get("ttl_s"), int)
            or not isinstance(raw.get("max_requests"), int)
            or not isinstance(raw.get("request_retry_limit"), int)
            or not isinstance(raw.get("stream_retry_limit"), int)
            or not isinstance(raw.get("stream_retry_limit_per_request"), int)
        ):
            raise ValueError("broker startup limits must be integers")
        return cls(
            evaluation_id=raw["evaluation_id"],
            attempt_id=raw["attempt_id"],
            source_container_id=raw["source_container_id"],
            source_ip=raw["source_ip"],
            model=raw["model"],
            reasoning_effort=raw["reasoning_effort"],
            capability=raw["capability"],
            credentials=BrokerSecrets(
                access_token=raw["access_token"],
                account_id=raw["account_id"],
            ),
            tls_certificate_pem=raw["tls_certificate_pem"].encode("ascii"),
            tls_certificate_sha256=raw["tls_certificate_sha256"],
            tls_private_key_pem=bytearray(raw["tls_private_key_pem"], "ascii"),
            ttl_s=raw["ttl_s"],
            max_requests=raw["max_requests"],
            request_retry_limit=raw["request_retry_limit"],
            stream_retry_limit=raw["stream_retry_limit"],
            stream_retry_limit_per_request=raw[
                "stream_retry_limit_per_request"
            ],
        )


@dataclass(frozen=True)
class ProxyRequest:
    method: str
    path: str
    headers: Mapping[str, str]
    body: bytes
    source_ip: str


@dataclass
class UpstreamResponse:
    status: int
    headers: tuple[tuple[str, str], ...]
    body: Iterable[bytes]
    close: Callable[[], None] = lambda: None


class UpstreamTransport(Protocol):
    def __call__(
        self,
        *,
        body: bytes,
        headers: Mapping[str, str],
        access_token: str,
        account_id: str,
        timeout_s: float,
    ) -> UpstreamResponse: ...


@dataclass(frozen=True)
class ProxyDecision:
    status: int
    headers: tuple[tuple[str, str], ...]
    body: Iterable[bytes]
    close: Callable[[], None] = field(default=lambda: None, repr=False)


class HttpUpstreamTransport:
    """One-shot HTTP/2 transport available only in the broker image."""

    def __init__(
        self,
        *,
        client_factory: Callable[[float], Any] | None = None,
    ) -> None:
        self._client_factory = client_factory

    def _new_client(self, timeout_s: float) -> Any:
        if self._client_factory is not None:
            return self._client_factory(timeout_s)
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError(
                "the Terminal-Bench broker image is missing its HTTP/2 transport"
            ) from exc
        transport = httpx.HTTPTransport(
            http1=False,
            http2=True,
            retries=0,
            trust_env=False,
        )
        return httpx.Client(
            transport=transport,
            http2=True,
            trust_env=False,
            follow_redirects=False,
            timeout=timeout_s,
        )

    def __call__(
        self,
        *,
        body: bytes,
        headers: Mapping[str, str],
        access_token: str,
        account_id: str,
        timeout_s: float,
    ) -> UpstreamResponse:
        client = self._new_client(max(1.0, timeout_s))
        client_close_lock = threading.Lock()
        client_closed = False

        def close_client_once() -> None:
            nonlocal client_closed
            with client_close_lock:
                if client_closed:
                    return
                client_closed = True
            client.close()

        def close_client_without_masking() -> None:
            try:
                close_client_once()
            except BaseException:
                pass

        forwarded = _upstream_headers(
            headers,
            access_token=access_token,
            account_id=account_id,
            body_length=len(body),
        )
        try:
            request = client.build_request(
                "POST",
                f"https://{CHATGPT_UPSTREAM_HOST}{CHATGPT_RESPONSES_PATH}",
                content=body,
                headers=forwarded,
            )
            response = client.send(request, stream=True)
        except BaseException:
            close_client_without_masking()
            raise
        if response.http_version != "HTTP/2":
            try:
                response.close()
            except BaseException:
                pass
            close_client_without_masking()
            raise UpstreamHttpVersionError("upstream did not negotiate HTTP/2")

        close_lock = threading.Lock()
        closed = False

        def close() -> None:
            nonlocal closed
            with close_lock:
                if closed:
                    return
                closed = True
            try:
                response.close()
            finally:
                close_client_once()

        def chunks() -> Iterable[bytes]:
            try:
                yield from response.iter_raw(chunk_size=64 * 1024)
            finally:
                close()

        try:
            status = response.status_code
            response_headers = tuple(response.headers.multi_items())
        except BaseException:
            try:
                close()
            except BaseException:
                pass
            raise
        return UpstreamResponse(
            status=status,
            headers=response_headers,
            body=chunks(),
            close=close,
        )


def _lower_headers(headers: Mapping[str, str]) -> dict[str, str]:
    lowered: dict[str, str] = {}
    for name, value in headers.items():
        key = name.strip().lower()
        if not key or "\n" in value or "\r" in value:
            raise ValueError("request contains an invalid header")
        if key in lowered:
            raise ValueError("request contains a duplicate header")
        lowered[key] = value.strip()
    return lowered


def _upstream_headers(
    incoming: Mapping[str, str],
    *,
    access_token: str,
    account_id: str,
    body_length: int,
) -> dict[str, str]:
    """Remove client authority and replace it with broker-held credentials."""
    forwarded: dict[str, str] = {}
    for name, value in incoming.items():
        lowered = name.lower()
        if (
            lowered in _HOP_BY_HOP_HEADERS
            or lowered in _UPSTREAM_REPLACED_HEADERS
            or lowered in _BINDING_HEADERS
            or lowered.startswith("x-lha-")
        ):
            continue
        forwarded[name] = value
    forwarded["Authorization"] = f"Bearer {access_token}"
    forwarded["ChatGPT-Account-ID"] = account_id
    forwarded["Content-Type"] = "application/json"
    forwarded["Content-Length"] = str(body_length)
    # Secret scanning is meaningful only over the uncompressed response bytes.
    forwarded["Accept-Encoding"] = "identity"
    return forwarded


def _client_response_headers(
    headers: Sequence[tuple[str, str]],
    *,
    body_length: int | None = None,
    content_type: str | None = None,
) -> tuple[tuple[str, str], ...]:
    safe: list[tuple[str, str]] = []
    for name, value in headers:
        lowered = name.lower()
        if (
            lowered in _HOP_BY_HOP_HEADERS
            or lowered in _CLIENT_STRIPPED_HEADERS
            or (content_type is not None and lowered == "content-type")
        ):
            continue
        if "\r" in name or "\n" in name or "\r" in value or "\n" in value:
            continue
        safe.append((name, value))
    if body_length is not None:
        safe.append(("Content-Length", str(body_length)))
    if content_type is not None:
        safe.append(("Content-Type", content_type))
    safe.append(("Connection", "close"))
    return tuple(safe)


def content_types_are_sse(raw_values: Sequence[str]) -> bool:
    """Accept repeated or safely coalesced copies of the SSE media type."""
    if not raw_values:
        return False

    media_types: list[str] = []
    for raw_value in raw_values:
        members: list[str] = []
        start = 0
        quoted = False
        escaped = False
        for index, character in enumerate(raw_value):
            if escaped:
                escaped = False
            elif quoted and character == "\\":
                escaped = True
            elif character == '"':
                quoted = not quoted
            elif character == "," and not quoted:
                members.append(raw_value[start:index].strip())
                start = index + 1
        if quoted or escaped:
            return False
        members.append(raw_value[start:].strip())
        if any(not member for member in members):
            return False
        media_types.extend(
            member.partition(";")[0].strip().lower() for member in members
        )

    return bool(media_types) and all(
        media_type == "text/event-stream" for media_type in media_types
    )


def _bounded_content_type_observation(value: str) -> str:
    normalized = " ".join(value.strip().lower().split())
    if (
        normalized
        and len(normalized) <= BROKER_MAX_OBSERVED_CONTENT_TYPE_CHARS
        and normalized.isascii()
        and normalized.isprintable()
    ):
        return normalized
    encoded = value.encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}:bytes={len(encoded)}"


def _valid_observed_content_types(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) <= BROKER_MAX_OBSERVED_CONTENT_TYPES
        and all(
            isinstance(item, str)
            and 0 < len(item) <= BROKER_MAX_OBSERVED_CONTENT_TYPE_CHARS
            and item.isascii()
            and item.isprintable()
            for item in value
        )
    )


def _json_error(status: int, code: str) -> ProxyDecision:
    body = json.dumps(
        {"error": {"code": code, "message": "request rejected by evaluation broker"}},
        separators=(",", ":"),
    ).encode()
    return ProxyDecision(
        status=status,
        headers=(
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(body))),
            ("Connection", "close"),
        ),
        body=(body,),
    )


def _diagnostic_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > _MAX_ERROR_TEXT_CHARS
        or any(ord(character) < 32 or ord(character) == 127 for character in cleaned)
    ):
        return None
    return cleaned


def _upstream_error_diagnostic(status: int, body: bytes) -> dict[str, object]:
    error_code: str | None = None
    error_type: str | None = None
    error_param: str | None = None
    message: str | None = None
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = None
    if isinstance(payload, dict):
        raw_error = payload.get("error")
        error = raw_error if isinstance(raw_error, dict) else payload
        error_code = _diagnostic_text(error.get("code"))
        error_type = _diagnostic_text(error.get("type"))
        error_param = _diagnostic_text(error.get("param"))
        message = _diagnostic_text(error.get("message"))
        if message is None:
            message = _diagnostic_text(payload.get("detail"))
    return {
        "status": status,
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "body_bytes": len(body),
        "error_code": error_code,
        "error_type": error_type,
        "error_param": error_param,
        "message": message,
    }


class _ResponsesSseTracker:
    """Recognize one structurally complete Responses API terminal SSE frame."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self.completed = False

    @staticmethod
    def _frame_end(payload: bytes | bytearray, start: int) -> tuple[int, int] | None:
        lf = payload.find(b"\n\n", start)
        crlf = payload.find(b"\r\n\r\n", start)
        candidates = (
            (lf, 2),
            (crlf, 4),
        )
        present = tuple(candidate for candidate in candidates if candidate[0] >= 0)
        if not present:
            return None
        offset, delimiter_length = min(present, key=lambda candidate: candidate[0])
        return offset, offset + delimiter_length

    @staticmethod
    def _is_completed_frame(
        frame: bytes,
        *,
        require_valid_event: bool,
    ) -> bool:
        event: str | None = None
        data_lines: list[bytes] = []
        for raw_line in frame.replace(b"\r\n", b"\n").split(b"\n"):
            if not raw_line or raw_line.startswith(b":"):
                continue
            name, separator, raw_value = raw_line.partition(b":")
            value = raw_value[1:] if separator and raw_value.startswith(b" ") else raw_value
            if name == b"event":
                try:
                    event = value.decode("utf-8")
                except UnicodeDecodeError:
                    return False
            elif name == b"data":
                data_lines.append(value)
        if not data_lines:
            if event == "response.completed":
                raise ValueError("response.completed SSE frame omitted data")
            return False
        try:
            payload = json.loads(b"\n".join(data_lines))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            if event == "response.completed" or require_valid_event:
                raise ValueError("Responses SSE data was invalid") from exc
            return False
        if not isinstance(payload, dict) or not isinstance(payload.get("type"), str):
            if event == "response.completed" or require_valid_event:
                raise ValueError("Responses SSE data omitted its event type")
            return False
        payload_type = payload["type"]
        if require_valid_event and event not in (None, payload_type):
            raise ValueError("Responses SSE event and data types disagree")
        if payload_type != "response.completed":
            if event == "response.completed":
                raise ValueError("response.completed SSE data changed type")
            return False
        if event not in (None, "response.completed"):
            raise ValueError("response.completed SSE event changed type")
        response = payload.get("response")
        if not (
            isinstance(response, dict)
            and isinstance(response.get("id"), str)
            and bool(response["id"].strip())
        ):
            raise ValueError("response.completed SSE data omitted the response id")
        return True

    def feed(self, chunk: bytes, *, require_valid_events: bool = False) -> int | None:
        """Return the chunk prefix ending at response.completed, if present."""
        previous_length = len(self._buffer)
        combined = self._buffer + chunk
        cursor = 0
        while frame_boundary := self._frame_end(combined, cursor):
            frame_end, next_cursor = frame_boundary
            frame = bytes(combined[cursor:frame_end])
            if len(frame) > _MAX_SSE_FRAME_BYTES:
                raise ValueError("upstream SSE frame exceeded the broker limit")
            if self._is_completed_frame(
                frame,
                require_valid_event=require_valid_events,
            ):
                self.completed = True
                self._buffer.clear()
                return max(0, next_cursor - previous_length)
            cursor = next_cursor
        remainder = combined[cursor:]
        if len(remainder) > _MAX_SSE_FRAME_BYTES:
            raise ValueError("upstream SSE frame exceeded the broker limit")
        self._buffer = bytearray(remainder)
        return None


class TokenBroker:
    """Validate an attempt-scoped capability before forwarding one request."""

    def __init__(
        self,
        startup: BrokerStartup,
        *,
        transport: UpstreamTransport | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._startup = startup
        self._transport = transport or HttpUpstreamTransport()
        self._monotonic = monotonic
        self._started_monotonic = monotonic()
        self._started_at = now()
        self._lock = threading.Lock()
        self._revoked = False
        self._accepted = 0
        self._rejected = 0
        self._rejection_reasons: dict[str, int] = {}
        self._upstream_attempts = 0
        self._upstream_statuses: dict[int, int] = {}
        self._stream_retries_used = 0
        self._stream_retried_requests = 0
        self._max_stream_retries_on_request = 0
        self._upstream_error: dict[str, object] | None = None
        self._upstream_transport_errors: dict[str, int] = {}
        self._upstream_stream_errors: dict[str, int] = {}
        self._observed_content_types: list[str] = []
        self._active: set[Callable[[], None]] = set()

    @staticmethod
    def _safe_error_type(exc: Exception) -> str:
        error_type = type(exc).__name__
        return (
            error_type
            if _ERROR_TYPE_RE.fullmatch(error_type) is not None
            else "UnknownError"
        )

    def _record_transport_error(self, error_type: str) -> int:
        with self._lock:
            self._upstream_transport_errors[error_type] = (
                self._upstream_transport_errors.get(error_type, 0) + 1
            )
            return sum(self._upstream_transport_errors.values())

    def _record_stream_error(self, error_type: str) -> int:
        with self._lock:
            self._upstream_stream_errors[error_type] = (
                self._upstream_stream_errors.get(error_type, 0) + 1
            )
            return sum(self._upstream_stream_errors.values())

    def _record_upstream_attempt(self) -> None:
        with self._lock:
            self._upstream_attempts += 1

    def _record_upstream_status(self, status: int) -> None:
        with self._lock:
            self._upstream_statuses[status] = (
                self._upstream_statuses.get(status, 0) + 1
            )

    def _record_observed_content_types(
        self,
        headers: Sequence[tuple[str, str]],
    ) -> tuple[str, ...]:
        raw_values = tuple(
            value for name, value in headers if name.lower() == "content-type"
        )
        observations = tuple(
            _bounded_content_type_observation(value) for value in raw_values
        )
        with self._lock:
            for observation in observations:
                if (
                    observation in self._observed_content_types
                    or _OBSERVED_CONTENT_TYPES_OMITTED
                    in self._observed_content_types
                ):
                    continue
                if (
                    len(self._observed_content_types)
                    < BROKER_MAX_OBSERVED_CONTENT_TYPES
                ):
                    self._observed_content_types.append(observation)
                else:
                    self._observed_content_types[-1] = (
                        _OBSERVED_CONTENT_TYPES_OMITTED
                    )
        return raw_values

    def _claim_stream_retry(self, retries_for_request: int) -> bool:
        with self._lock:
            if (
                self._revoked
                or self._stream_retries_used >= self._startup.stream_retry_limit
            ):
                return False
            self._stream_retries_used += 1
            if retries_for_request == 0:
                self._stream_retried_requests += 1
            self._max_stream_retries_on_request = max(
                self._max_stream_retries_on_request,
                retries_for_request + 1,
            )
            return True

    def _record_rejection(self, reason: str) -> None:
        if reason not in BROKER_REJECTION_REASONS:
            raise ValueError("broker rejection reason is not registered")
        with self._lock:
            self._rejected += 1
            self._rejection_reasons[reason] = self._rejection_reasons.get(reason, 0) + 1

    def _reject(
        self,
        status: int,
        code: str,
        *,
        reason: str | None = None,
    ) -> ProxyDecision:
        self._record_rejection(reason or code)
        return _json_error(status, code)

    def _reserve(self) -> float | None:
        with self._lock:
            elapsed = self._monotonic() - self._started_monotonic
            if self._revoked or elapsed >= self._startup.ttl_s:
                return None
            if self._accepted >= self._startup.max_requests:
                return -1.0
            self._accepted += 1
            return max(1.0, self._startup.ttl_s - elapsed)

    def _remaining_timeout(self) -> float | None:
        with self._lock:
            elapsed = self._monotonic() - self._started_monotonic
            if self._revoked or elapsed >= self._startup.ttl_s:
                return None
            return max(1.0, self._startup.ttl_s - elapsed)

    def _tracked_close(
        self,
        upstream: UpstreamResponse,
    ) -> tuple[Callable[[], None], bool]:
        close_lock = threading.Lock()
        closed = False

        def close() -> None:
            nonlocal closed
            with close_lock:
                if closed:
                    return
                closed = True
            try:
                upstream.close()
            finally:
                with self._lock:
                    self._active.discard(close)

        with self._lock:
            revoked = self._revoked
            if not revoked:
                self._active.add(close)
        return close, revoked

    @staticmethod
    def _close_without_masking(close: Callable[[], None]) -> None:
        # Once a validated response.completed frame is buffered, an HTTP/2
        # shutdown error cannot change its contents or be allowed to discard it.
        try:
            close()
        except Exception:
            pass

    def _buffer_success_body(
        self,
        upstream: UpstreamResponse,
        *,
        protected: tuple[bytes, ...],
        require_valid_events: bool,
    ) -> bytes:
        buffered = bytearray()
        tracker = _ResponsesSseTracker()
        overlap = max(len(secret) for secret in protected) - 1
        scan_tail = b""
        for chunk in upstream.body:
            if self._remaining_timeout() is None:
                raise UpstreamDeadlineExceeded
            if not isinstance(chunk, bytes):
                raise UpstreamNonBytesChunk
            terminal_prefix = tracker.feed(
                chunk,
                require_valid_events=require_valid_events,
            )
            accepted_chunk = chunk if terminal_prefix is None else chunk[:terminal_prefix]
            if (
                len(buffered) + len(accepted_chunk)
                > BROKER_MAX_BUFFERED_RESPONSE_BYTES
            ):
                raise UpstreamResponseTooLarge
            scanned = scan_tail + accepted_chunk
            if any(secret in scanned for secret in protected):
                raise UpstreamSecretInBody
            buffered.extend(accepted_chunk)
            scan_tail = scanned[-overlap:] if overlap else b""
            if terminal_prefix is not None:
                # Do not request another HTTP/2 DATA frame or wait for EOF. The
                # terminal SSE frame is the application-level completeness proof.
                return bytes(buffered)
        raise MissingResponseCompleted

    def handle(self, request: ProxyRequest) -> ProxyDecision:
        """Return a response without ever reflecting a secret or capability."""
        if request.method != "POST" or request.path != "/responses":
            return self._reject(404, "route_not_allowed")
        if len(request.body) > _MAX_REQUEST_BYTES:
            return self._reject(413, "request_too_large")
        try:
            source_ip = _canonical_ip(request.source_ip)
            headers = _lower_headers(request.headers)
        except ValueError:
            return self._reject(400, "invalid_request")
        if source_ip != self._startup.source_ip:
            return self._reject(403, "source_mismatch")
        if headers.get("authorization") != f"Bearer {self._startup.capability}":
            return self._reject(401, "invalid_capability")
        if (
            headers.get("x-lha-evaluation-id") != self._startup.evaluation_id
            or headers.get("x-lha-attempt-id") != self._startup.attempt_id
            or headers.get("x-lha-container-id") != self._startup.source_container_id
        ):
            return self._reject(403, "binding_mismatch")
        if headers.get("content-type", "").partition(";")[0].strip() != "application/json":
            return self._reject(415, "content_type_not_allowed")
        try:
            payload = json.loads(request.body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._reject(400, "invalid_json")
        if not isinstance(payload, dict):
            return self._reject(400, "invalid_json")
        reasoning = payload.get("reasoning")
        if payload.get("model") != self._startup.model:
            return self._reject(403, "model_mismatch")
        if (
            not isinstance(reasoning, dict)
            or reasoning.get("effort") != self._startup.reasoning_effort
        ):
            return self._reject(403, "effort_mismatch")

        timeout_s = self._reserve()
        if timeout_s is None:
            return self._reject(410, "capability_revoked")
        if timeout_s < 0:
            return self._reject(429, "request_limit_reached")

        protected_text = (
            self._startup.credentials.access_token,
            self._startup.credentials.account_id,
            self._startup.capability,
        )
        protected = tuple(secret.encode() for secret in protected_text)
        stream_retries_for_request = 0

        while True:
            # Re-check the shared deadline and revocation state immediately
            # before every physical request, including the first one. Another
            # request may have revoked this broker after reservation.
            remaining_timeout = self._remaining_timeout()
            if remaining_timeout is None:
                self.revoke()
                return self._reject(
                    502,
                    "upstream_failed",
                    reason="upstream_attempt_timeout",
                )
            self._record_upstream_attempt()
            try:
                upstream = self._transport(
                    body=request.body,
                    headers=headers,
                    access_token=self._startup.credentials.access_token,
                    account_id=self._startup.credentials.account_id,
                    timeout_s=remaining_timeout,
                )
            except Exception as exc:
                error_type = self._safe_error_type(exc)
                failures = self._record_transport_error(error_type)
                decision = self._reject(
                    502,
                    "upstream_failed",
                    reason="upstream_transport_exception",
                )
                if (
                    stream_retries_for_request > 0
                    or error_type not in BROKER_RECOVERABLE_TRANSPORT_ERRORS
                    or failures > self._startup.request_retry_limit
                ):
                    self.revoke()
                return decision

            if type(upstream.status) is not int or not 100 <= upstream.status <= 599:
                self._close_without_masking(upstream.close)
                self.revoke()
                return self._reject(
                    502,
                    "upstream_failed",
                    reason="upstream_invalid_status",
                )
            self._record_upstream_status(upstream.status)
            close, revoked = self._tracked_close(upstream)
            if revoked:
                self._record_rejection("response_after_revocation")
                self._close_without_masking(close)
                return _json_error(410, "capability_revoked")

            try:
                response_headers = tuple(upstream.headers)
                valid_headers = all(
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], str)
                    and isinstance(item[1], str)
                    and _HEADER_NAME_RE.fullmatch(item[0]) is not None
                    and not any(
                        ord(character) < 32 or ord(character) == 127
                        for character in item[0]
                    )
                    and not any(
                        ord(character) < 32 or ord(character) == 127
                        for character in item[1]
                    )
                    for item in response_headers
                )
            except Exception:
                valid_headers = False
                response_headers = ()
            if not valid_headers:
                self._close_without_masking(close)
                self.revoke()
                return self._reject(
                    502,
                    "upstream_failed",
                    reason="upstream_header_shape",
                )
            if any(
                secret in name or secret in value
                for name, value in response_headers
                for secret in protected_text
            ):
                self._close_without_masking(close)
                self.revoke()
                return self._reject(
                    502,
                    "upstream_failed",
                    reason="upstream_secret_in_headers",
                )
            observed_content_types = self._record_observed_content_types(
                response_headers
            )
            content_encodings = [
                value.strip().lower()
                for name, value in response_headers
                if name.lower() == "content-encoding"
            ]
            if any(value not in {"", "identity"} for value in content_encodings):
                self._close_without_masking(close)
                self.revoke()
                return self._reject(
                    502,
                    "upstream_failed",
                    reason="upstream_content_encoding",
                )

            if upstream.status >= 400:
                if stream_retries_for_request > 0:
                    self._close_without_masking(close)
                    self.revoke()
                    return self._reject(
                        502,
                        "upstream_failed",
                        reason="upstream_retry_failed",
                    )
                error_body = bytearray()
                valid_body = True
                try:
                    for chunk in upstream.body:
                        if not isinstance(chunk, bytes):
                            valid_body = False
                            break
                        if len(error_body) + len(chunk) > _MAX_ERROR_RESPONSE_BYTES:
                            valid_body = False
                            break
                        error_body.extend(chunk)
                except Exception:
                    valid_body = False
                finally:
                    self._close_without_masking(close)
                leaked_secret = any(secret in error_body for secret in protected)
                if not valid_body or leaked_secret:
                    self.revoke()
                    return self._reject(
                        502,
                        "upstream_failed",
                        reason=(
                            "upstream_secret_in_error_body"
                            if leaked_secret
                            else "upstream_invalid_error_body"
                        ),
                    )
                rendered_error = bytes(error_body)
                with self._lock:
                    self._upstream_error = _upstream_error_diagnostic(
                        upstream.status,
                        rendered_error,
                    )
                return ProxyDecision(
                    status=upstream.status,
                    headers=_client_response_headers(
                        response_headers,
                        body_length=len(rendered_error),
                    ),
                    body=(rendered_error,),
                )

            if observed_content_types and not content_types_are_sse(
                observed_content_types
            ):
                self._close_without_masking(close)
                self.revoke()
                return self._reject(
                    502,
                    "upstream_failed",
                    reason="upstream_content_type_invalid",
                )

            try:
                rendered = self._buffer_success_body(
                    upstream,
                    protected=protected,
                    require_valid_events=not observed_content_types,
                )
            except Exception as exc:
                self._close_without_masking(close)
                error_type = self._safe_error_type(exc)
                self._record_stream_error(error_type)
                if (
                    stream_retries_for_request
                    < self._startup.stream_retry_limit_per_request
                    and error_type in BROKER_RECOVERABLE_STREAM_ERRORS
                    and self._claim_stream_retry(stream_retries_for_request)
                ):
                    stream_retries_for_request += 1
                    continue
                if isinstance(exc, UpstreamSecretInBody):
                    reason = "upstream_secret_in_body"
                elif isinstance(exc, UpstreamResponseTooLarge):
                    reason = "upstream_response_too_large"
                elif isinstance(exc, UpstreamNonBytesChunk):
                    reason = "upstream_non_bytes_chunk"
                elif isinstance(exc, UpstreamDeadlineExceeded):
                    reason = "upstream_timeout"
                elif isinstance(exc, ValueError):
                    reason = "upstream_invalid_sse"
                else:
                    reason = "upstream_stream_failure"
                self.revoke()
                return self._reject(
                    502,
                    "upstream_failed",
                    reason=reason,
                )
            else:
                self._close_without_masking(close)
                return ProxyDecision(
                    status=upstream.status,
                    headers=_client_response_headers(
                        response_headers,
                        body_length=len(rendered),
                        content_type="text/event-stream",
                    ),
                    body=(rendered,),
                )

    def revoke(self) -> None:
        """Make future requests fail and close active upstream streams."""
        with self._lock:
            if self._revoked:
                return
            self._revoked = True
            active = tuple(self._active)
            self._active.clear()
        for close in active:
            try:
                close()
            except Exception:
                pass

    def receipt(self, *, outcome: str = "revoked") -> dict[str, object]:
        """Return the intentionally secret-free shutdown receipt."""
        with self._lock:
            accepted = self._accepted
            rejected = self._rejected
            upstream_attempts = self._upstream_attempts
            statuses = {
                str(status): count for status, count in sorted(self._upstream_statuses.items())
            }
            stream_retries_used = self._stream_retries_used
            stream_retried_requests = self._stream_retried_requests
            max_stream_retries_on_request = self._max_stream_retries_on_request
            upstream_error = (
                dict(self._upstream_error) if self._upstream_error is not None else None
            )
            rejection_reasons = dict(sorted(self._rejection_reasons.items()))
            upstream_transport_errors = dict(
                sorted(self._upstream_transport_errors.items())
            )
            upstream_stream_errors = dict(sorted(self._upstream_stream_errors.items()))
            observed_content_types = list(self._observed_content_types)
            revoked = self._revoked
        return {
            "schema_version": 5,
            "type": "terminal_proxy_receipt",
            "evaluation_id": self._startup.evaluation_id,
            "attempt_id": self._startup.attempt_id,
            "source_container_id": self._startup.source_container_id,
            "started_at": self._started_at.isoformat(),
            "stopped_at": now().isoformat(),
            "ttl_s": self._startup.ttl_s,
            "max_requests": self._startup.max_requests,
            "max_buffered_response_bytes": BROKER_MAX_BUFFERED_RESPONSE_BYTES,
            "request_retry_limit": self._startup.request_retry_limit,
            "stream_retry_limit": self._startup.stream_retry_limit,
            "stream_retry_limit_per_request": (
                self._startup.stream_retry_limit_per_request
            ),
            "downstream_accepted_requests": accepted,
            "rejected_requests": rejected,
            "rejection_reasons": rejection_reasons,
            "upstream_attempts": upstream_attempts,
            "upstream_statuses": statuses,
            "stream_retries_used": stream_retries_used,
            "stream_retried_requests": stream_retried_requests,
            "max_stream_retries_on_request": max_stream_retries_on_request,
            "upstream_error": upstream_error,
            "upstream_transport_errors": upstream_transport_errors,
            "upstream_stream_errors": upstream_stream_errors,
            "observed_content_types": observed_content_types,
            "revoked": revoked,
            "outcome": outcome,
        }


class _LinePump:
    def __init__(self, stream: IO[str]) -> None:
        self._stream = stream
        self._queue: queue.Queue[str] = queue.Queue()
        self._lines: list[str] = []
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        for line in self._stream:
            stripped = line.rstrip("\r\n")
            self._lines.append(stripped)
            self._queue.put(stripped)

    def next(self, timeout_s: float) -> str:
        try:
            return self._queue.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise TerminalProxyError("broker did not report readiness") from exc

    def join(self, timeout_s: float) -> None:
        self._thread.join(timeout_s)

    @property
    def lines(self) -> tuple[str, ...]:
        return tuple(self._lines)


@dataclass(repr=False)
class TerminalProxyHandle:
    """A running broker.  Repr deliberately excludes the capability."""

    name: str
    evaluation_id: str
    attempt_id: str
    source_container_id: str
    source_ip: str
    network: str
    broker_container_id: str
    broker_ip: str
    image_id: str
    tls_certificate_pem: bytes
    tls_certificate_sha256: str
    process: subprocess.Popen[str]
    stdout: _LinePump
    stderr: _LinePump
    _capability: str = field(repr=False)
    _credentials: BrokerSecrets | None = field(repr=False)
    _stopped: bool = field(default=False, repr=False)

    def __repr__(self) -> str:
        state = "stopped" if self._stopped else "running"
        return (
            "TerminalProxyHandle("
            f"name={self.name!r}, evaluation_id={self.evaluation_id!r}, "
            f"attempt_id={self.attempt_id!r}, source_container_id={self.source_container_id!r}, "
            f"source_ip={self.source_ip!r}, network={self.network!r}, "
            f"broker_container_id={self.broker_container_id!r}, "
            f"broker_ip={self.broker_ip!r}, image_id={self.image_id!r}, "
            f"tls_certificate_sha256={self.tls_certificate_sha256!r}, state={state!r}, "
            "capability=<redacted>, credentials=<redacted>)"
        )

    @property
    def base_url(self) -> str:
        return f"https://{BROKER_ALIAS}:{BROKER_PORT}"

    def client_headers(self) -> dict[str, str]:
        """Return direct-client headers.

        Codex config should use :meth:`binding_headers` as ``http_headers`` and
        :meth:`capability_environment` as the provider ``env_key``.  Keeping
        Authorization out of config prevents the capability from being saved
        as a literal TOML value.
        """
        if self._stopped or not self._capability:
            raise TerminalProxyError("broker capability is no longer available")
        return {
            "Authorization": f"Bearer {self._capability}",
            **self.binding_headers(),
        }

    def binding_headers(self) -> dict[str, str]:
        """Static, non-secret headers for a Codex custom model provider."""
        return {
            "X-LHA-Evaluation-ID": self.evaluation_id,
            "X-LHA-Attempt-ID": self.attempt_id,
            "X-LHA-Container-ID": self.source_container_id,
        }

    def capability_environment(self) -> dict[str, str]:
        """The one process-local variable named by the provider ``env_key``."""
        if self._stopped or not self._capability:
            raise TerminalProxyError("broker capability is no longer available")
        return {CAPABILITY_ENV: self._capability}


RunCommand = Callable[..., subprocess.CompletedProcess[str]]
PopenFactory = Callable[..., subprocess.Popen[str]]


class TerminalProxyController:
    """Start and attest one broker container from the host."""

    def __init__(
        self,
        *,
        image_id: str,
        docker: str = "docker",
        run_command: RunCommand = subprocess.run,
        popen_factory: PopenFactory = subprocess.Popen,
    ) -> None:
        if _SHA256_IMAGE_RE.fullmatch(image_id) is None:
            raise ValueError("broker image must be pinned by full sha256 image ID")
        self.image_id = image_id
        self.docker = str(Path(docker)) if "/" in docker else docker
        self._run_command = run_command
        self._popen_factory = popen_factory

    def _run(
        self, argv: list[str], *, timeout_s: float = _DOCKER_TIMEOUT_S
    ) -> subprocess.CompletedProcess[str]:
        try:
            return self._run_command(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise TerminalProxyError("Docker control command failed") from exc

    def _inspect(self, target: str, *, kind: str = "container") -> Mapping[str, Any]:
        process = self._run([self.docker, "inspect", "--type", kind, target])
        if process.returncode != 0:
            raise TerminalProxyError("required Docker object is unavailable")
        try:
            payload = json.loads(process.stdout)
        except json.JSONDecodeError as exc:
            raise TerminalProxyError("Docker inspection returned invalid JSON") from exc
        if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
            raise TerminalProxyError("Docker inspection returned an unexpected shape")
        return cast(Mapping[str, Any], payload[0])

    @staticmethod
    def _network_binding(
        inspection: Mapping[str, Any],
        network: str,
        *,
        require_only_network: bool,
    ) -> tuple[str, str]:
        container_id = inspection.get("Id")
        state = inspection.get("State")
        settings = inspection.get("NetworkSettings")
        networks = settings.get("Networks") if isinstance(settings, Mapping) else None
        if (
            not isinstance(container_id, str)
            or _CONTAINER_ID_RE.fullmatch(container_id) is None
            or not isinstance(state, Mapping)
            or state.get("Running") is not True
            or not isinstance(networks, Mapping)
            or network not in networks
        ):
            raise TerminalProxyError("Docker container is not running on the required network")
        if require_only_network and set(networks) != {network}:
            raise TerminalProxyError("Docker container has an unexpected additional network")
        binding = networks[network]
        ip_value = binding.get("IPAddress") if isinstance(binding, Mapping) else None
        if not isinstance(ip_value, str) or not ip_value:
            raise TerminalProxyError("Docker network did not assign a source IP")
        try:
            canonical = _canonical_ip(ip_value)
        except ValueError as exc:
            raise TerminalProxyError("Docker network returned an invalid source IP") from exc
        return container_id, canonical

    def _attest_image(self) -> None:
        image = self._inspect(self.image_id, kind="image")
        if (
            image.get("Id") != self.image_id
            or image.get("Os") != "linux"
            or image.get("Architecture") != "amd64"
        ):
            raise TerminalProxyError("broker image identity changed during inspection")

    def _attest_broker(
        self,
        inspection: Mapping[str, Any],
        *,
        name: str,
        network: str,
        evaluation_id: str,
        attempt_id: str,
        forbidden_values: Sequence[str],
    ) -> tuple[str, str]:
        container_id, broker_ip = self._network_binding(
            inspection,
            network,
            require_only_network=True,
        )
        config = inspection.get("Config")
        host = inspection.get("HostConfig")
        mounts = inspection.get("Mounts")
        labels = config.get("Labels") if isinstance(config, Mapping) else None
        if (
            inspection.get("Name") not in {name, f"/{name}"}
            or inspection.get("Image") != self.image_id
            or not isinstance(config, Mapping)
            or not isinstance(labels, Mapping)
            or labels.get("lha.terminal.role") != "broker"
            or labels.get("lha.terminal.evaluation_id") != evaluation_id
            or labels.get("lha.terminal.attempt_id") != attempt_id
            or not isinstance(host, Mapping)
            or mounts != []
            or host.get("NetworkMode") != network
            or host.get("ReadonlyRootfs") is not True
            or host.get("Privileged") is True
            or host.get("PortBindings") not in (None, {})
            or host.get("Binds") not in (None, [])
            or host.get("PidMode") not in (None, "")
            or host.get("IpcMode") in {"host", "container"}
        ):
            raise TerminalProxyError("broker container isolation attestation failed")
        cap_drop = host.get("CapDrop")
        security = host.get("SecurityOpt")
        if (
            not isinstance(cap_drop, list)
            or "ALL" not in cap_drop
            or not isinstance(security, list)
            or not any("no-new-privileges" in str(value) for value in security)
        ):
            raise TerminalProxyError("broker container privilege attestation failed")
        serialized = json.dumps(inspection, sort_keys=True)
        if any(value and value in serialized for value in forbidden_values):
            raise TerminalProxyError("broker secret appeared in Docker inspection")
        env = config.get("Env")
        if isinstance(env, list) and any(
            str(item).partition("=")[0]
            in {
                "OPENAI_API_KEY",
                "CODEX_ACCESS_TOKEN",
                "LHA_PROXY_CAPABILITY",
                "LHA_PROXY_ACCOUNT_ID",
                CAPABILITY_ENV,
            }
            for item in env
        ):
            raise TerminalProxyError("broker received a credential-bearing environment variable")
        return container_id, broker_ip

    def _build_run_argv(
        self,
        *,
        name: str,
        network: str,
        evaluation_id: str,
        attempt_id: str,
    ) -> list[str]:
        return [
            self.docker,
            "run",
            "-i",
            "--platform",
            "linux/amd64",
            "--name",
            name,
            "--network",
            network,
            "--network-alias",
            BROKER_ALIAS,
            "--label",
            "lha.terminal.role=broker",
            "--label",
            f"lha.terminal.evaluation_id={evaluation_id}",
            "--label",
            f"lha.terminal.attempt_id={attempt_id}",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            "128m",
            "--cpus",
            "0.5",
            "--user",
            "65532:65532",
            self.image_id,
        ]

    @staticmethod
    def _name(evaluation_id: str, attempt_id: str) -> str:
        digest = hashlib.sha256(f"{evaluation_id}\0{attempt_id}".encode()).hexdigest()[:20]
        return f"lha-terminal-proxy-{digest}"

    def _remove(self, name: str) -> None:
        result = self._run([self.docker, "rm", "-f", name])
        if result.returncode != 0:
            raise TerminalProxyError("Docker did not remove the broker")

    def _remove_if_present(self, name: str) -> None:
        result = self._run([self.docker, "rm", "-f", name])
        if result.returncode != 0 and not self._reports_absent(result):
            raise TerminalProxyError("Docker did not remove the broker")
        self._confirm_removed(name)

    @staticmethod
    def _reports_absent(result: subprocess.CompletedProcess[str]) -> bool:
        if result.returncode == 0:
            return False
        detail = f"{result.stdout}\n{result.stderr}".lower()
        return "no such object" in detail or "no such container" in detail

    def _confirm_removed(self, name: str) -> None:
        result = self._run([self.docker, "inspect", "--type", "container", name])
        if not self._reports_absent(result):
            raise TerminalProxyError("broker container deletion could not be confirmed")

    def _require_absent(self, name: str) -> None:
        result = self._run([self.docker, "inspect", "--type", "container", name])
        if not self._reports_absent(result):
            raise TerminalProxyError("broker container name is not provably unused")

    def _inspect_if_present(self, target: str) -> Mapping[str, Any] | None:
        result = self._run(
            [self.docker, "inspect", "--type", "container", target]
        )
        if self._reports_absent(result):
            return None
        if result.returncode != 0:
            raise TerminalProxyError("Docker container presence could not be determined")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise TerminalProxyError("Docker inspection returned invalid JSON") from exc
        if (
            not isinstance(payload, list)
            or len(payload) != 1
            or not isinstance(payload[0], dict)
        ):
            raise TerminalProxyError("Docker inspection returned an unexpected shape")
        return cast(Mapping[str, Any], payload[0])

    def _remove_container_and_confirm(
        self,
        container_id: str,
        *,
        additional_absence_target: str | None = None,
    ) -> None:
        result = self._run([self.docker, "rm", "-f", container_id])
        if result.returncode != 0 and not self._reports_absent(result):
            raise TerminalProxyError("Docker did not remove an abandoned container")
        self._confirm_removed(container_id)
        if additional_absence_target is not None:
            self._confirm_removed(additional_absence_target)

    def _attest_abandoned_broker(
        self,
        inspection: Mapping[str, Any],
        *,
        name: str,
        evaluation_id: str,
        attempt_id: str,
    ) -> str:
        container_id = inspection.get("Id")
        config = inspection.get("Config")
        labels = config.get("Labels") if isinstance(config, Mapping) else None
        host = inspection.get("HostConfig")
        mounts = inspection.get("Mounts")
        cap_drop = host.get("CapDrop") if isinstance(host, Mapping) else None
        security = host.get("SecurityOpt") if isinstance(host, Mapping) else None
        if (
            not isinstance(container_id, str)
            or _CONTAINER_ID_RE.fullmatch(container_id) is None
            or inspection.get("Name") not in {name, f"/{name}"}
            or inspection.get("Image") != self.image_id
            or not isinstance(config, Mapping)
            or config.get("Image") != self.image_id
            or not isinstance(labels, Mapping)
            or labels.get("lha.terminal.role") != "broker"
            or labels.get("lha.terminal.evaluation_id") != evaluation_id
            or labels.get("lha.terminal.attempt_id") != attempt_id
            or not isinstance(host, Mapping)
            or mounts != []
            or host.get("ReadonlyRootfs") is not True
            or host.get("Privileged") is True
            or host.get("PortBindings") not in (None, {})
            or host.get("Binds") not in (None, [])
            or host.get("PidMode") not in (None, "")
            or not isinstance(cap_drop, list)
            or "ALL" not in cap_drop
            or not isinstance(security, list)
            or not any("no-new-privileges" in str(value) for value in security)
        ):
            raise TerminalProxyError(
                "abandoned broker container attestation failed"
            )
        return container_id

    def _attest_abandoned_task(
        self,
        inspection: Mapping[str, Any],
        *,
        source_container_id: str,
        expected_image_digest: str,
    ) -> None:
        config = inspection.get("Config")
        labels = config.get("Labels") if isinstance(config, Mapping) else None
        image_id = inspection.get("Image")
        configured_image = (
            config.get("Image") if isinstance(config, Mapping) else None
        )
        if (
            inspection.get("Id") != source_container_id
            or not isinstance(image_id, str)
            or not isinstance(configured_image, str)
            or not isinstance(labels, Mapping)
            or labels.get("com.docker.compose.service") != "main"
            or not isinstance(labels.get("com.docker.compose.project"), str)
        ):
            raise TerminalProxyError("abandoned task container attestation failed")
        image = self._inspect(image_id, kind="image")
        repo_digests = image.get("RepoDigests")
        pinned_suffix = f"@{expected_image_digest}"
        if (
            image.get("Id") != image_id
            or not isinstance(repo_digests, list)
            or not all(isinstance(value, str) for value in repo_digests)
            or not (
                configured_image.endswith(pinned_suffix)
                or any(value.endswith(pinned_suffix) for value in repo_digests)
            )
        ):
            raise TerminalProxyError(
                "abandoned task image attestation failed"
            )

    def cleanup_abandoned(
        self,
        *,
        evaluation_id: str,
        attempt_id: str,
        source_container_id: str,
        expected_task_image_digest: str,
    ) -> None:
        """Delete a crashed trial's broker and task only after attesting both."""
        evaluation_id = _safe_identifier(evaluation_id, label="evaluation_id")
        attempt_id = _safe_identifier(attempt_id, label="attempt_id")
        if _CONTAINER_ID_RE.fullmatch(source_container_id) is None:
            raise ValueError("source_container_id must be a full Docker container id")
        if _SHA256_IMAGE_RE.fullmatch(expected_task_image_digest) is None:
            raise ValueError("task image must be pinned by a full SHA-256 digest")

        failures: list[TerminalProxyError] = []
        name = self._name(evaluation_id, attempt_id)
        try:
            broker = self._inspect_if_present(name)
            if broker is not None:
                broker_id = self._attest_abandoned_broker(
                    broker,
                    name=name,
                    evaluation_id=evaluation_id,
                    attempt_id=attempt_id,
                )
                self._remove_container_and_confirm(
                    broker_id,
                    additional_absence_target=name,
                )
        except TerminalProxyError as exc:
            failures.append(exc)

        try:
            task = self._inspect_if_present(source_container_id)
            if task is not None:
                self._attest_abandoned_task(
                    task,
                    source_container_id=source_container_id,
                    expected_image_digest=expected_task_image_digest,
                )
                self._remove_container_and_confirm(source_container_id)
        except TerminalProxyError as exc:
            failures.append(exc)

        if failures:
            raise TerminalProxyError(
                "abandoned trial cleanup could not be proven"
            ) from failures[0]

    def start(
        self,
        *,
        evaluation_id: str,
        attempt_id: str,
        source_container_id: str,
        network: str,
        model: str,
        reasoning_effort: str,
        credentials: BrokerSecrets,
    ) -> TerminalProxyHandle:
        evaluation_id = _safe_identifier(evaluation_id, label="evaluation_id")
        attempt_id = _safe_identifier(attempt_id, label="attempt_id")
        if _CONTAINER_ID_RE.fullmatch(source_container_id) is None:
            raise ValueError("source_container_id must be a full Docker container id")
        if _NETWORK_RE.fullmatch(network) is None:
            raise ValueError("network has an invalid format")
        self._attest_image()
        source_before = self._inspect(source_container_id)
        actual_source_id, source_ip = self._network_binding(
            source_before,
            network,
            require_only_network=True,
        )
        if actual_source_id != source_container_id:
            raise TerminalProxyError("source container identity changed")

        name = self._name(evaluation_id, attempt_id)
        self._require_absent(name)
        capability = secrets.token_urlsafe(48)
        certificate_pem, private_key_pem, certificate_sha256 = _generate_broker_tls_material()
        try:
            startup = BrokerStartup(
                evaluation_id=evaluation_id,
                attempt_id=attempt_id,
                source_container_id=source_container_id,
                source_ip=source_ip,
                model=model,
                reasoning_effort=reasoning_effort,
                capability=capability,
                credentials=credentials,
                tls_certificate_pem=certificate_pem,
                tls_certificate_sha256=certificate_sha256,
                tls_private_key_pem=private_key_pem,
            )
        except BaseException:
            for index in range(len(private_key_pem)):
                private_key_pem[index] = 0
            private_key_pem.clear()
            raise
        private_key_text = bytes(private_key_pem).decode("ascii")
        argv = self._build_run_argv(
            name=name,
            network=network,
            evaluation_id=evaluation_id,
            attempt_id=attempt_id,
        )
        process: subprocess.Popen[str] | None = None
        try:
            process = self._popen_factory(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            if process.stdin is None or process.stdout is None or process.stderr is None:
                raise TerminalProxyError("Docker attach pipes were not created")
            stdout = _LinePump(process.stdout)
            stderr = _LinePump(process.stderr)
            process.stdin.write(startup.stdin_json() + "\n")
            process.stdin.flush()
            process.stdin.close()
            startup.clear_tls_private_key()
            ready_line = stdout.next(_DOCKER_TIMEOUT_S)
            if private_key_text in ready_line or any(
                private_key_text in line for line in stderr.lines
            ):
                raise TerminalProxyError("broker output contained TLS private key material")
            try:
                ready = json.loads(ready_line)
            except json.JSONDecodeError as exc:
                raise TerminalProxyError("broker readiness record is invalid") from exc
            expected_ready = {
                "schema_version": 4,
                "type": "terminal_proxy_ready",
                "evaluation_id": evaluation_id,
                "attempt_id": attempt_id,
                "port": BROKER_PORT,
                "ttl_s": BROKER_TTL_S,
                "max_requests": BROKER_MAX_REQUESTS,
                "request_retry_limit": BROKER_REQUEST_RETRY_LIMIT,
                "stream_retry_limit": BROKER_STREAM_RETRY_LIMIT,
                "stream_retry_limit_per_request": (
                    BROKER_STREAM_RETRY_LIMIT_PER_REQUEST
                ),
                "tls_certificate_sha256": certificate_sha256,
            }
            if ready != expected_ready:
                raise TerminalProxyError("broker readiness record did not match the attempt")
            broker_inspection = self._inspect(name)
            broker_container_id, broker_ip = self._attest_broker(
                broker_inspection,
                name=name,
                network=network,
                evaluation_id=evaluation_id,
                attempt_id=attempt_id,
                forbidden_values=(
                    credentials.access_token,
                    credentials.account_id,
                    capability,
                    private_key_text,
                ),
            )
            source_after = self._inspect(source_container_id)
            after_id, after_ip = self._network_binding(
                source_after,
                network,
                require_only_network=True,
            )
            if after_id != actual_source_id or after_ip != source_ip:
                raise TerminalProxyError("source container binding changed during broker startup")
            return TerminalProxyHandle(
                name=name,
                evaluation_id=evaluation_id,
                attempt_id=attempt_id,
                source_container_id=actual_source_id,
                source_ip=source_ip,
                network=network,
                broker_container_id=broker_container_id,
                broker_ip=broker_ip,
                image_id=self.image_id,
                tls_certificate_pem=certificate_pem,
                tls_certificate_sha256=certificate_sha256,
                process=process,
                stdout=stdout,
                stderr=stderr,
                _capability=capability,
                _credentials=credentials,
            )
        except BaseException as error:
            cleanup_error: TerminalProxyError | None = None
            try:
                self._remove_if_present(name)
            except TerminalProxyError as exc:
                cleanup_error = exc
            if process is not None and process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
            if cleanup_error is not None:
                raise TerminalProxyError(
                    "broker startup failed and container cleanup was not confirmed"
                ) from cleanup_error
            raise error
        finally:
            startup.clear_tls_private_key()
            private_key_text = ""

    def stop(self, handle: TerminalProxyHandle) -> dict[str, object]:
        if handle._stopped:
            raise TerminalProxyError("broker handle was already stopped")
        failure: TerminalProxyError | None = None
        try:
            try:
                stopped = self._run(
                    [self.docker, "stop", "--time", "5", handle.name],
                    timeout_s=_STOP_TIMEOUT_S,
                )
                if stopped.returncode != 0:
                    failure = TerminalProxyError("Docker did not stop the broker cleanly")
            except TerminalProxyError as exc:
                failure = exc
            try:
                handle.process.wait(timeout=_STOP_TIMEOUT_S)
            except (OSError, subprocess.TimeoutExpired):
                failure = TerminalProxyError("broker attach process did not exit")
            handle.stdout.join(1.0)
            handle.stderr.join(1.0)
        finally:
            try:
                self._remove(handle.name)
                self._confirm_removed(handle.name)
            except TerminalProxyError as exc:
                if failure is None:
                    failure = exc
            handle._stopped = True

        combined = "\n".join((*handle.stdout.lines, *handle.stderr.lines))
        credentials = handle._credentials
        if credentials is None:
            raise TerminalProxyError("broker credentials were already released")
        forbidden = (credentials.access_token, credentials.account_id, handle._capability)
        leaked = any(value and value in combined for value in forbidden)
        handle._capability = ""
        handle._credentials = None
        if leaked:
            raise TerminalProxyError("broker output contained protected material")
        if failure is not None:
            raise failure

        if len(handle.stdout.lines) != 2 or any(line for line in handle.stderr.lines if line):
            raise TerminalProxyError("broker emitted unexpected output")
        records: list[dict[str, Any]] = []
        for line in handle.stdout.lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                raise TerminalProxyError("broker emitted a non-JSON stdout record") from None
            if not isinstance(item, dict):
                raise TerminalProxyError("broker emitted a non-object stdout record")
            records.append(item)
        expected_ready = {
            "schema_version": 4,
            "type": "terminal_proxy_ready",
            "evaluation_id": handle.evaluation_id,
            "attempt_id": handle.attempt_id,
            "port": BROKER_PORT,
            "ttl_s": BROKER_TTL_S,
            "max_requests": BROKER_MAX_REQUESTS,
            "request_retry_limit": BROKER_REQUEST_RETRY_LIMIT,
            "stream_retry_limit": BROKER_STREAM_RETRY_LIMIT,
            "stream_retry_limit_per_request": (
                BROKER_STREAM_RETRY_LIMIT_PER_REQUEST
            ),
            "tls_certificate_sha256": handle.tls_certificate_sha256,
        }
        if records[0] != expected_ready:
            raise TerminalProxyError("broker readiness record changed after startup")
        receipt = records[1]
        expected_receipt_keys = {
            "schema_version",
            "type",
            "evaluation_id",
            "attempt_id",
            "source_container_id",
            "started_at",
            "stopped_at",
            "ttl_s",
            "max_requests",
            "max_buffered_response_bytes",
            "request_retry_limit",
            "stream_retry_limit",
            "stream_retry_limit_per_request",
            "downstream_accepted_requests",
            "rejected_requests",
            "rejection_reasons",
            "upstream_attempts",
            "upstream_statuses",
            "stream_retries_used",
            "stream_retried_requests",
            "max_stream_retries_on_request",
            "upstream_error",
            "upstream_transport_errors",
            "upstream_stream_errors",
            "observed_content_types",
            "revoked",
            "outcome",
        }
        if (
            set(receipt) != expected_receipt_keys
            or receipt.get("schema_version") != 5
            or receipt.get("type") != "terminal_proxy_receipt"
            or receipt.get("evaluation_id") != handle.evaluation_id
            or receipt.get("attempt_id") != handle.attempt_id
            or receipt.get("source_container_id") != handle.source_container_id
            or receipt.get("ttl_s") != BROKER_TTL_S
            or receipt.get("max_requests") != BROKER_MAX_REQUESTS
            or receipt.get("max_buffered_response_bytes")
            != BROKER_MAX_BUFFERED_RESPONSE_BYTES
            or receipt.get("request_retry_limit")
            != BROKER_REQUEST_RETRY_LIMIT
            or receipt.get("stream_retry_limit") != BROKER_STREAM_RETRY_LIMIT
            or receipt.get("stream_retry_limit_per_request")
            != BROKER_STREAM_RETRY_LIMIT_PER_REQUEST
            or receipt.get("revoked") is not True
            or receipt.get("outcome") != "sigterm"
        ):
            raise TerminalProxyError("broker shutdown receipt did not match the attempt")
        accepted = receipt.get("downstream_accepted_requests")
        rejected = receipt.get("rejected_requests")
        rejection_reasons = receipt.get("rejection_reasons")
        upstream_attempts = receipt.get("upstream_attempts")
        statuses = receipt.get("upstream_statuses")
        stream_retries_used = receipt.get("stream_retries_used")
        stream_retried_requests = receipt.get("stream_retried_requests")
        max_stream_retries_on_request = receipt.get(
            "max_stream_retries_on_request"
        )
        upstream_error = receipt.get("upstream_error")
        upstream_transport_errors = receipt.get("upstream_transport_errors")
        upstream_stream_errors = receipt.get("upstream_stream_errors")
        observed_content_types = receipt.get("observed_content_types")
        if (
            type(accepted) is not int
            or not 0 <= accepted <= BROKER_MAX_REQUESTS
            or type(rejected) is not int
            or rejected < 0
            or type(upstream_attempts) is not int
            or not 0
            <= upstream_attempts
            <= BROKER_MAX_REQUESTS + BROKER_STREAM_RETRY_LIMIT
            or type(stream_retries_used) is not int
            or not 0 <= stream_retries_used <= BROKER_STREAM_RETRY_LIMIT
            or type(stream_retried_requests) is not int
            or not 0 <= stream_retried_requests <= accepted
            or type(max_stream_retries_on_request) is not int
            or not 0
            <= max_stream_retries_on_request
            <= BROKER_STREAM_RETRY_LIMIT_PER_REQUEST
            or not isinstance(rejection_reasons, dict)
            or any(
                not isinstance(reason, str)
                or reason not in BROKER_REJECTION_REASONS
                or type(count) is not int
                or count < 1
                for reason, count in rejection_reasons.items()
            )
            or sum(cast(int, count) for count in rejection_reasons.values())
            != rejected
            or not isinstance(statuses, dict)
            or any(
                not isinstance(status, str)
                or not status.isdecimal()
                or not 100 <= int(status) <= 599
                or type(count) is not int
                or count < 1
                for status, count in statuses.items()
            )
            or not _valid_observed_content_types(observed_content_types)
        ):
            raise TerminalProxyError("broker shutdown receipt contains invalid counters")
        if (
            (stream_retries_used == 0)
            != (
                stream_retried_requests == 0
                and max_stream_retries_on_request == 0
            )
            or stream_retries_used < stream_retried_requests
            or max_stream_retries_on_request > stream_retries_used
            or stream_retries_used
            > stream_retried_requests * max_stream_retries_on_request
        ):
            raise TerminalProxyError(
                "broker shutdown receipt per-request retry counts disagree"
            )
        transport_rejections = rejection_reasons.get(
            "upstream_transport_exception",
            0,
        )
        error_maps = (upstream_transport_errors, upstream_stream_errors)
        if any(
            not isinstance(errors, dict)
            or any(
                not isinstance(error_type, str)
                or _ERROR_TYPE_RE.fullmatch(error_type) is None
                or type(count) is not int
                or count < 1
                for error_type, count in errors.items()
            )
            or sum(cast(int, count) for count in errors.values())
            > BROKER_MAX_REQUESTS + BROKER_STREAM_RETRY_LIMIT
            for errors in error_maps
        ):
            raise TerminalProxyError(
                "broker shutdown receipt contains invalid retry diagnostics"
            )
        assert isinstance(upstream_transport_errors, dict)
        if sum(
            cast(int, count) for count in upstream_transport_errors.values()
        ) != transport_rejections:
            raise TerminalProxyError(
                "broker shutdown receipt transport counts disagree"
            )
        invalid_statuses = cast(
            int,
            rejection_reasons.get("upstream_invalid_status", 0),
        )
        transport_error_count = sum(
            cast(int, count) for count in upstream_transport_errors.values()
        )
        if upstream_attempts != (
            sum(cast(int, count) for count in statuses.values())
            + transport_error_count
            + invalid_statuses
        ):
            raise TerminalProxyError(
                "broker shutdown receipt upstream accounting disagrees"
            )
        attempt_timeouts = cast(
            int,
            rejection_reasons.get("upstream_attempt_timeout", 0),
        )
        if accepted + stream_retries_used != upstream_attempts + attempt_timeouts:
            raise TerminalProxyError(
                "broker shutdown receipt downstream accounting disagrees"
            )
        assert isinstance(upstream_stream_errors, dict)
        stream_error_count = sum(
            cast(int, count) for count in upstream_stream_errors.values()
        )
        final_stream_failures = sum(
            cast(int, rejection_reasons.get(reason, 0))
            for reason in _FINAL_STREAM_REJECTION_REASONS
        )
        if (
            stream_error_count != stream_retries_used + final_stream_failures
            or stream_error_count
            > sum(
                cast(int, count)
                for status, count in statuses.items()
                if 200 <= int(status) < 400
            )
        ):
            raise TerminalProxyError(
                "broker shutdown receipt stream counts disagree"
            )
        if upstream_error is not None:
            expected_error_keys = {
                "status",
                "body_sha256",
                "body_bytes",
                "error_code",
                "error_type",
                "error_param",
                "message",
            }
            if (
                not isinstance(upstream_error, dict)
                or set(upstream_error) != expected_error_keys
                or type(upstream_error.get("status")) is not int
                or cast(int, upstream_error["status"]) < 400
                or cast(int, upstream_error["status"]) > 599
                or statuses.get(str(upstream_error["status"]), 0) < 1
                or not isinstance(upstream_error.get("body_sha256"), str)
                or _SHA256_HEX_RE.fullmatch(
                    cast(str, upstream_error["body_sha256"])
                )
                is None
                or type(upstream_error.get("body_bytes")) is not int
                or not 0
                <= cast(int, upstream_error["body_bytes"])
                <= _MAX_ERROR_RESPONSE_BYTES
                or any(
                    value is not None
                    and (
                        not isinstance(value, str)
                        or _diagnostic_text(value) != value
                    )
                    for value in (
                        upstream_error.get("error_code"),
                        upstream_error.get("error_type"),
                        upstream_error.get("error_param"),
                        upstream_error.get("message"),
                    )
                )
            ):
                raise TerminalProxyError(
                    "broker shutdown receipt contains an invalid upstream error"
                )
        try:
            started_at = datetime.fromisoformat(cast(str, receipt["started_at"]))
            stopped_at = datetime.fromisoformat(cast(str, receipt["stopped_at"]))
        except (TypeError, ValueError) as exc:
            raise TerminalProxyError("broker shutdown receipt contains invalid timestamps") from exc
        if started_at.tzinfo is None or stopped_at.tzinfo is None or stopped_at < started_at:
            raise TerminalProxyError("broker shutdown receipt contains invalid timestamps")
        return cast(dict[str, object], receipt)


def broker_image_build_command(
    *,
    dockerfile: str | Path = "docker/terminal-bench-proxy.Dockerfile",
    context: str | Path = ".",
    tag: str = "lha-terminal-proxy:local",
) -> list[str]:
    """Return the auditable build command; callers must inspect the resulting ID."""
    return [
        "docker",
        "build",
        "--pull=false",
        "--platform",
        "linux/amd64",
        "--file",
        str(dockerfile),
        "--tag",
        tag,
        str(context),
    ]
