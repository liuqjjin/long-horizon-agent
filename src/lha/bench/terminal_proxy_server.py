"""Run the Terminal-Bench credential broker inside its isolated container."""

from __future__ import annotations

import json
import os
import signal
import ssl
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import NoReturn

from .terminal_proxy import (
    BROKER_MAX_REQUESTS,
    BROKER_PORT,
    BROKER_REQUEST_RETRY_LIMIT,
    BROKER_STREAM_RETRY_LIMIT,
    BROKER_STREAM_RETRY_LIMIT_PER_REQUEST,
    BROKER_TTL_S,
    BrokerStartup,
    ProxyDecision,
    ProxyRequest,
    TokenBroker,
)


class _BrokerServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(
        self,
        address: tuple[str, int],
        broker: TokenBroker,
        tls_context: ssl.SSLContext,
    ) -> None:
        self.broker = broker
        super().__init__(address, _BrokerHandler, bind_and_activate=False)
        try:
            self.server_bind()
            self.server_activate()
            self.socket = tls_context.wrap_socket(self.socket, server_side=True)
        except BaseException:
            self.server_close()
            raise


class _BrokerHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "lha-terminal-proxy"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:
        # Request headers can contain the capability; never copy them to logs.
        del format, args
        return

    def _reject_length(self) -> None:
        body = (
            b'{"error":{"code":"invalid_length","message":"request rejected by evaluation broker"}}'
        )
        self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        raw_length = self.headers.get("Content-Length")
        if (
            raw_length is None
            or not raw_length.isascii()
            or not raw_length.isdecimal()
            or self.headers.get("Transfer-Encoding") is not None
            or any(
                len(self.headers.get_all(name, [])) != 1
                for name in (
                    "Content-Length",
                    "Authorization",
                    "X-LHA-Evaluation-ID",
                    "X-LHA-Attempt-ID",
                    "X-LHA-Container-ID",
                )
            )
        ):
            self._reject_length()
            return
        length = int(raw_length)
        if length > 32 * 1024 * 1024:
            self._reject_length()
            return
        body = self.rfile.read(length)
        broker = self.server.broker  # type: ignore[attr-defined]
        try:
            decision = broker.handle(
                ProxyRequest(
                    method="POST",
                    path=self.path,
                    headers={name: value for name, value in self.headers.items()},
                    body=body,
                    source_ip=self.client_address[0],
                )
            )
        except Exception:
            # No exception text is logged because transport errors can contain
            # credential-bearing URLs or headers.
            broker.revoke()
            response = (
                b'{"error":{"code":"upstream_failed",'
                b'"message":"request rejected by evaluation broker"}}'
            )
            decision = ProxyDecision(
                status=502,
                headers=(
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response))),
                    ("Connection", "close"),
                ),
                body=(response,),
            )
        try:
            self.send_response(decision.status)
            for name, value in decision.headers:
                self.send_header(name, value)
            self.end_headers()
            for chunk in decision.body:
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            decision.close()
            self.close_connection = True

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        self.send_error(404)


class _ShutdownCoordinator:
    """Choose one terminal outcome before revoking the broker."""

    def __init__(self, server: _BrokerServer, broker: TokenBroker) -> None:
        self._server = server
        self._broker = broker
        # Signal handlers run on the main thread and can interrupt outcome reads.
        self._lock = threading.RLock()
        self._outcome: str | None = None

    def request(self, outcome: str) -> bool:
        """Revoke and stop once; a later signal cannot rewrite the receipt."""
        with self._lock:
            if self._outcome is not None:
                return False
            self._outcome = outcome
        try:
            self._broker.revoke()
        finally:
            # BaseServer.shutdown deadlocks when called by the serve_forever thread.
            threading.Thread(target=self._server.shutdown, daemon=True).start()
        return True

    @property
    def outcome(self) -> str:
        with self._lock:
            return self._outcome or "stopped"


def _load_tls_context(startup: BrokerStartup) -> ssl.SSLContext:
    """Load the attempt key, then remove every plaintext filesystem/memory copy."""
    private_key = bytearray(startup.tls_private_key_pem)
    paths: list[str] = []
    try:
        for prefix, contents in (
            ("lha-terminal-proxy-cert-", startup.tls_certificate_pem),
            ("lha-terminal-proxy-key-", private_key),
        ):
            descriptor, path = tempfile.mkstemp(prefix=prefix, dir="/tmp")
            paths.append(path)
            try:
                os.fchmod(descriptor, 0o600)
                view = memoryview(contents)
                try:
                    written = 0
                    while written < len(view):
                        written += os.write(descriptor, view[written:])
                finally:
                    view.release()
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(certfile=paths[0], keyfile=paths[1])
        return context
    finally:
        startup.clear_tls_private_key()
        for index in range(len(private_key)):
            private_key[index] = 0
        private_key.clear()
        for path in paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def _fatal() -> NoReturn:
    # Startup failures intentionally reveal neither input nor exception text.
    print(
        json.dumps(
            {
                "schema_version": 5,
                "type": "terminal_proxy_receipt",
                "outcome": "startup_failed",
            },
            separators=(",", ":"),
        ),
        flush=True,
    )
    raise SystemExit(2)


def main() -> int:
    line = sys.stdin.readline(128 * 1024)
    if not line or len(line) >= 128 * 1024:
        _fatal()
    try:
        startup = BrokerStartup.from_stdin_json(line)
    except (TypeError, ValueError):
        _fatal()
    line = ""
    watchdog_deadline = time.monotonic() + startup.ttl_s
    try:
        tls_context = _load_tls_context(startup)
    except (OSError, ssl.SSLError, ValueError):
        _fatal()
    broker = TokenBroker(startup)
    try:
        server = _BrokerServer(("0.0.0.0", BROKER_PORT), broker, tls_context)
    except OSError:
        broker.revoke()
        _fatal()

    shutdown = _ShutdownCoordinator(server, broker)

    def stop(signum: int, _frame: object) -> None:
        outcome = "sigterm" if signum == signal.SIGTERM else "sigint"
        shutdown.request(outcome)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    watchdog = threading.Timer(
        max(0.0, watchdog_deadline - time.monotonic()),
        shutdown.request,
        args=("ttl_expired",),
    )
    watchdog.daemon = True
    watchdog.start()
    print(
        json.dumps(
            {
                "schema_version": 4,
                "type": "terminal_proxy_ready",
                "evaluation_id": startup.evaluation_id,
                "attempt_id": startup.attempt_id,
                "port": BROKER_PORT,
                "ttl_s": BROKER_TTL_S,
                "max_requests": BROKER_MAX_REQUESTS,
                "request_retry_limit": BROKER_REQUEST_RETRY_LIMIT,
                "stream_retry_limit": BROKER_STREAM_RETRY_LIMIT,
                "stream_retry_limit_per_request": (
                    BROKER_STREAM_RETRY_LIMIT_PER_REQUEST
                ),
                "tls_certificate_sha256": startup.tls_certificate_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        watchdog.cancel()
        watchdog.join()
        broker.revoke()
        server.server_close()
        print(
            json.dumps(
                broker.receipt(outcome=shutdown.outcome),
                sort_keys=True,
                separators=(",", ":"),
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
