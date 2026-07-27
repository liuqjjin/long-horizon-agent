"""Real-Linux preflight for the Terminal-Bench broker capability boundary.

The test is opt-in because it needs Docker and the frozen x86_64 Codex 0.141
binary. It uses a fake Responses server on an internal Docker network and never
contacts a model or an external API.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import textwrap
import time
from collections.abc import Iterable
from pathlib import Path

import pytest

from lha.bench import terminal_bench as tb
from lha.bench import terminal_proxy as proxy

_ENABLED = os.environ.get("LHA_TERMINAL_ISOLATION_TESTS") == "1"
_EXPECTED_CODEX_SHA256 = (
    "1f37cb63f6c8c3e5e9dddb247a822706f153c00e553efe8c0b05f62d0e4f1cab"
)
_TASK_IMAGE = os.environ.get(
    "LHA_TERMINAL_ISOLATION_TASK_IMAGE",
    "alexgshaw/password-recovery:20251031",
)
_BROKER_IMAGE = os.environ.get(
    "LHA_TERMINAL_ISOLATION_BROKER_IMAGE",
    "python:3.12-slim",
)
_EVALUATION_ID = "e" * 32
_ATTEMPT_ID = "a" * 64


def _redact(value: str, forbidden: Iterable[str]) -> str:
    result = value
    for secret in forbidden:
        if secret:
            result = result.replace(secret, "<redacted>")
    return result


def _run(
    argv: list[str],
    *,
    timeout: int = 60,
    check: bool = True,
    forbidden: Iterable[str] = (),
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        stdout = _redact(result.stdout, forbidden)[-3000:]
        stderr = _redact(result.stderr, forbidden)[-3000:]
        raise AssertionError(
            f"command failed ({result.returncode}): {argv!r}\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return result


def _copy(container: str, source: Path, destination: str, *, forbidden: Iterable[str]) -> None:
    _run(
        ["docker", "cp", str(source), f"{container}:{destination}"],
        forbidden=forbidden,
    )


def _wait_for_file(
    container: str,
    path: str,
    *,
    forbidden: Iterable[str],
    timeout_s: float = 15.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = _run(
            ["docker", "exec", container, "test", "-f", path],
            check=False,
            forbidden=forbidden,
        )
        if result.returncode == 0:
            return
        time.sleep(0.05)
    pytest.fail(f"container did not create {path!r} before the deadline")


def _install_private_file(
    *,
    container: str,
    payload: bytes,
    prefix: str,
    staging_path: str,
    destination_path: str,
    owner: str,
    forbidden: Iterable[str],
) -> None:
    local_path = tb._write_private_payload(payload, prefix=prefix)
    try:
        _copy(container, local_path, staging_path, forbidden=forbidden)
        command = tb._finalize_uploaded_file_command(
            staging_path=staging_path,
            destination_path=destination_path,
            owner=owner,
            mode=0o600,
            expected_sha256=hashlib.sha256(payload).hexdigest(),
        )
        _run(
            ["docker", "exec", "--user", "0:0", container, "/bin/bash", "-c", command],
            forbidden=forbidden,
        )
    finally:
        local_path.unlink(missing_ok=True)


def _probe_command() -> str:
    """Return the model-controlled command supplied by the fake server."""
    script = textwrap.dedent(
        f"""
        set -eu
        capability_key={proxy.CAPABILITY_ENV!r}
        capability_found=false
        stolen=""
        readable_proc_count=0
        denied_proc_count=0
        for environ_path in /proc/[0-9]*/environ; do
          raw_scratch=/tmp/.lha-proc-environ-raw
          scratch=/tmp/.lha-proc-environ
          if timeout 1 cat "$environ_path" > "$raw_scratch" 2>/dev/null; then
            tr '\\0' '\\n' < "$raw_scratch" > "$scratch"
            readable_proc_count=$((readable_proc_count + 1))
            if grep -q "^${{capability_key}}=" "$scratch"; then
              stolen=$(sed -n "s/^${{capability_key}}=//p" "$scratch" | head -n 1)
              capability_found=true
              rm -f "$raw_scratch" "$scratch"
              break
            fi
          else
            denied_proc_count=$((denied_proc_count + 1))
          fi
          rm -f "$raw_scratch" "$scratch"
        done

        codex_ancestor_found=false
        codex_environ_readable=false
        codex_mem_openable=false
        ancestor=$PPID
        ancestor_steps=0
        while [ "$ancestor" -gt 1 ] && [ "$ancestor_steps" -lt 12 ]; do
          ancestor_name=$(awk '/^Name:/ {{print $2}}' "/proc/$ancestor/status")
          if [ "$ancestor_name" = codex ]; then
            codex_ancestor_found=true
            if timeout 1 cat "/proc/$ancestor/environ" >/dev/null 2>&1; then
              codex_environ_readable=true
            fi
            if /bin/bash -c "exec 7</proc/$ancestor/mem" 2>/dev/null; then
              codex_mem_openable=true
            fi
            break
          fi
          ancestor=$(awk '/^PPid:/ {{print $2}}' "/proc/$ancestor/status")
          ancestor_steps=$((ancestor_steps + 1))
        done

        capability_upload_exists=false
        capability_upload_readable=false
        if [ -e {tb._CAPABILITY_UPLOAD!r} ]; then
          capability_upload_exists=true
        fi
        if [ -r {tb._CAPABILITY_UPLOAD!r} ]; then
          capability_upload_readable=true
        fi

        ca_path={tb._TLS_CERT_PATH!r}
        wrong_ca_path=/tmp/.lha_wrong_ca.pem
        original_copy=/tmp/.lha_original_ca.pem
        ca_path_present_before_probe=false
        if [ -e "$ca_path" ]; then
          ca_path_present_before_probe=true
        fi
        original_ca_sha256=$(sha256sum "$original_copy" | awk '{{print $1}}')

        tls_fd_inherited=false
        tls_fd_matches_original=false
        tls_fd_write_reopen_succeeded=false
        if [ -e /proc/self/fd/3 ]; then
          tls_fd_inherited=true
          if [ "$(sha256sum /proc/self/fd/3 2>/dev/null | awk '{{print $1}}')" = \
               "$original_ca_sha256" ]; then
            tls_fd_matches_original=true
            if cat "$wrong_ca_path" > /proc/self/fd/3 2>/dev/null; then
              tls_fd_write_reopen_succeeded=true
            fi
          fi
        fi

        direct_capability=${{stolen:-unavailable_capability_probe_value}}
        direct_response=$(
          {{
            printf '%s\\r\\n' \
              'POST /responses HTTP/1.0' \
              'Host: {proxy.BROKER_ALIAS}' \
              'Content-Type: application/json' \
              'Content-Length: 2' \
              'X-LHA-Evaluation-ID: {_EVALUATION_ID}' \
              'X-LHA-Attempt-ID: {_ATTEMPT_ID}' \
              "X-LHA-Container-ID: $LHA_EXPECTED_CONTAINER_ID" \
              'X-LHA-Probe: tool' \
              "Authorization: Bearer $direct_capability" \
              '' \
              '{{}}'
          }} | timeout 8 openssl s_client \
              -quiet \
              -connect {proxy.BROKER_ALIAS}:{proxy.BROKER_PORT} \
              -servername {proxy.BROKER_ALIAS} \
              -verify_return_error \
              -CAfile "$original_copy" 2>/dev/null || true
        )
        direct_status=$(printf '%s\\n' "$direct_response" | awk 'NR==1 {{print $2}}')
        direct_status=${{direct_status:--1}}

        plaintext_response=$(
          timeout 8 /bin/bash -c "
            exec 9<>/dev/tcp/{proxy.BROKER_ALIAS}/{proxy.BROKER_PORT}
            printf 'POST /responses HTTP/1.0\\r\\nContent-Length: 2\\r\\n\\r\\n{{}}' >&9
            cat <&9
          " 2>/dev/null || true
        )
        plaintext_http_failed=true
        case "$plaintext_response" in
          HTTP/*) plaintext_http_failed=false ;;
        esac

        wrong_ca_failed=false
        if timeout 8 openssl s_client \
          -connect {proxy.BROKER_ALIAS}:{proxy.BROKER_PORT} \
          -servername {proxy.BROKER_ALIAS} \
          -verify_return_error \
          -CAfile "$wrong_ca_path" </dev/null >/dev/null 2>&1; then
          wrong_ca_failed=false
        else
          wrong_ca_failed=true
        fi

        cp "$wrong_ca_path" "$ca_path"
        overwritten_ca_sha256=$(sha256sum "$ca_path" | awk '{{print $1}}')
        wrong_ca_sha256=$(sha256sum "$wrong_ca_path" | awk '{{print $1}}')
        ca_overwritten=false
        if [ "$overwritten_ca_sha256" = "$wrong_ca_sha256" ] && \
           [ "$overwritten_ca_sha256" != "$original_ca_sha256" ]; then
          ca_overwritten=true
        fi
        rm -f "$ca_path"
        ca_deleted=false
        if [ ! -e "$ca_path" ]; then
          ca_deleted=true
        fi

        read -r _ ruid euid suid _ < <(grep '^Uid:' /proc/self/status)
        read -r _ rgid egid sgid _ < <(grep '^Gid:' /proc/self/status)
        printf '%s\\n' \
          "{{\\"ruid\\":$ruid,\\"euid\\":$euid,\\"suid\\":$suid,"\
"\\"rgid\\":$rgid,\\"egid\\":$egid,\\"sgid\\":$sgid,"\
"\\"capability_found_in_proc\\":$capability_found,"\
"\\"capability_upload_exists\\":$capability_upload_exists,"\
"\\"capability_upload_readable\\":$capability_upload_readable,"\
"\\"codex_ancestor_found\\":$codex_ancestor_found,"\
"\\"codex_environ_readable\\":$codex_environ_readable,"\
"\\"codex_mem_openable\\":$codex_mem_openable,"\
"\\"direct_capability_reuse_status\\":$direct_status,"\
"\\"plaintext_http_failed\\":$plaintext_http_failed,"\
"\\"wrong_ca_failed\\":$wrong_ca_failed,"\
"\\"ca_path_present_before_probe\\":$ca_path_present_before_probe,"\
"\\"tls_fd_inherited\\":$tls_fd_inherited,"\
"\\"tls_fd_matches_original\\":$tls_fd_matches_original,"\
"\\"tls_fd_write_reopen_succeeded\\":$tls_fd_write_reopen_succeeded,"\
"\\"ca_overwritten\\":$ca_overwritten,\\"ca_deleted\\":$ca_deleted,"\
"\\"readable_proc_count\\":$readable_proc_count,"\
"\\"denied_proc_count\\":$denied_proc_count}}" \
          > /tmp/lha-capability-probe.json
        cat /tmp/lha-capability-probe.json
        """
    ).strip()
    return (
        "LHA_EXPECTED_CONTAINER_ID="
        "'$LHA_EXPECTED_CONTAINER_ID' "
        f"/bin/bash -p <<'BASH'\n{script}\nBASH"
    )


def _fake_broker_source(probe_command: str) -> str:
    return textwrap.dedent(
        f"""
        from __future__ import annotations

        import json
        import os
        import ssl
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from pathlib import Path

        startup_path = Path("/tmp/lha-broker-startup.json")
        startup = json.loads(startup_path.read_text())
        startup_path.unlink()
        capability = startup.pop("capability")
        expected_container_id = startup["container_id"]
        probe_command = {probe_command!r}.replace(
            "$LHA_EXPECTED_CONTAINER_ID", expected_container_id
        )
        model_requests = 0

        def event(payload):
            return (
                "event: " + payload["type"] + "\\n"
                "data: " + json.dumps(payload, separators=(",", ":")) + "\\n\\n"
            )

        def completed(response_id):
            return {{
                "type": "response.completed",
                "response": {{
                    "id": response_id,
                    "usage": {{
                        "input_tokens": 1,
                        "input_tokens_details": None,
                        "output_tokens": 1,
                        "output_tokens_details": None,
                        "total_tokens": 2,
                    }},
                }},
            }}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def do_POST(self):
                global model_requests
                size = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(size)
                try:
                    request = json.loads(body)
                except (ValueError, UnicodeDecodeError):
                    request = {{}}
                authorized = self.headers.get("Authorization") == "Bearer " + capability
                binding_ok = (
                    self.headers.get("X-LHA-Evaluation-ID") == {_EVALUATION_ID!r}
                    and self.headers.get("X-LHA-Attempt-ID") == {_ATTEMPT_ID!r}
                    and self.headers.get("X-LHA-Container-ID") == expected_container_id
                )
                model_ok = request.get("model") == "gpt-5.5"
                reasoning = request.get("reasoning")
                effort_ok = (
                    isinstance(reasoning, dict)
                    and reasoning.get("effort") == "xhigh"
                )
                is_probe = self.headers.get("X-LHA-Probe") == "tool"
                row = {{
                    "authorized": authorized,
                    "binding_ok": binding_ok,
                    "effort_ok": effort_ok,
                    "is_probe": is_probe,
                    "model_ok": model_ok,
                    "path": self.path,
                }}
                with open("/tmp/lha-broker-requests.jsonl", "a") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\\n")

                if is_probe:
                    self.send_error(403 if not authorized or not binding_ok else 409)
                    return
                if (
                    not authorized
                    or not binding_ok
                    or not model_ok
                    or not effort_ok
                    or self.path != "/responses"
                ):
                    self.send_error(403)
                    return

                model_requests += 1
                if model_requests == 1:
                    arguments = json.dumps(
                        {{
                            "command": probe_command,
                            "login": False,
                            "timeout_ms": 30_000,
                        }},
                        separators=(",", ":"),
                    )
                    events = [
                        {{
                            "type": "response.created",
                            "response": {{"id": "response-1"}},
                        }},
                        {{
                            "type": "response.output_item.done",
                            "item": {{
                                "type": "function_call",
                                "call_id": "probe-call",
                                "name": "shell_command",
                                "arguments": arguments,
                            }},
                        }},
                        completed("response-1"),
                    ]
                else:
                    events = [
                        {{
                            "type": "response.output_item.done",
                            "item": {{
                                "type": "message",
                                "role": "assistant",
                                "id": "answer-1",
                                "content": [
                                    {{"type": "output_text", "text": "probe complete"}}
                                ],
                            }},
                        }},
                        completed("response-2"),
                    ]
                response_body = "".join(event(item) for item in events).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(response_body)))
                self.end_headers()
                self.wfile.write(response_body)

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(
            "/tmp/lha-broker-cert.pem",
            "/tmp/lha-broker-key.pem",
        )
        os.unlink("/tmp/lha-broker-cert.pem")
        os.unlink("/tmp/lha-broker-key.pem")
        server = ThreadingHTTPServer(("0.0.0.0", {proxy.BROKER_PORT}), Handler)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        Path("/tmp/lha-broker-ready").touch()
        server.serve_forever()
        """
    )


def _capture_source() -> str:
    """Capture task-container Ethernet frames without adding a package."""
    return textwrap.dedent(
        """
        use strict;
        use warnings;

        # Linux AF_PACKET=17, SOCK_RAW=3, htons(ETH_P_ALL=3)=768 on x86_64.
        socket(my $socket, 17, 3, 768) or die "raw socket unavailable";
        open(my $wire, ">:raw", "/tmp/lha-wire.bin") or die "wire output unavailable";
        open(my $ready, ">", "/tmp/lha-capture-ready") or die "ready output unavailable";
        close($ready);
        my $deadline = time() + 180;
        while (time() < $deadline && !-e "/tmp/lha-capture-stop") {
            my $readable = "";
            vec($readable, fileno($socket), 1) = 1;
            my $selected = select($readable, undef, undef, 0.2);
            next if !$selected;
            my $frame = "";
            my $length = sysread($socket, $frame, 262144);
            next if !defined($length) || $length == 0;
            print {$wire} pack("N", $length), $frame;
        }
        close($wire);
        open(my $done, ">", "/tmp/lha-capture-done") or die "done output unavailable";
        close($done);
        """
    )


@pytest.mark.skipif(
    not _ENABLED,
    reason=(
        "real Codex capability isolation is opt-in; set "
        "LHA_TERMINAL_ISOLATION_TESTS=1"
    ),
)
def test_frozen_codex_tool_cannot_steal_or_reuse_broker_capability(tmp_path):
    binary_setting = os.environ.get("LHA_CODEX_BINARY_FILE")
    if not binary_setting:
        pytest.fail("LHA_CODEX_BINARY_FILE is required for the formal isolation preflight")
    binary = Path(binary_setting).resolve()
    if not binary.is_file():
        pytest.fail("LHA_CODEX_BINARY_FILE does not point to a regular file")
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == _EXPECTED_CODEX_SHA256

    _run(["docker", "info"])
    _run(["docker", "image", "inspect", _TASK_IMAGE])
    _run(["docker", "image", "inspect", _BROKER_IMAGE])

    capability = secrets.token_urlsafe(48)
    forbidden = (capability,)
    certificate_pem, private_key, certificate_sha256 = (
        proxy._generate_broker_tls_material()
    )
    wrong_certificate_pem, wrong_private_key, _ = proxy._generate_broker_tls_material()
    for index in range(len(wrong_private_key)):
        wrong_private_key[index] = 0
    wrong_private_key.clear()

    suffix = secrets.token_hex(6)
    network = f"lha-isolation-{suffix}"
    task_container = f"lha-isolation-task-{suffix}"
    broker_container = f"lha-isolation-broker-{suffix}"
    network_created = False
    task_created = False
    broker_created = False
    try:
        _run(["docker", "network", "create", "--internal", network], forbidden=forbidden)
        network_created = True
        for name, alias, image, platform in (
            (
                task_container,
                "lha-terminal-task",
                _TASK_IMAGE,
                ("--platform", "linux/amd64"),
            ),
            (
                broker_container,
                proxy.BROKER_ALIAS,
                _BROKER_IMAGE,
                (),
            ),
        ):
            _run(
                [
                    "docker",
                    "run",
                    "--detach",
                    "--pull",
                    "never",
                    *platform,
                    "--name",
                    name,
                    "--network",
                    network,
                    "--network-alias",
                    alias,
                    image,
                    "sleep",
                    "infinity",
                ],
                timeout=120,
                forbidden=forbidden,
            )
            if name == task_container:
                task_created = True
            else:
                broker_created = True

        container_id = _run(
            ["docker", "inspect", "--format", "{{.Id}}", task_container],
            forbidden=forbidden,
        ).stdout.strip()
        assert len(container_id) == 64

        _copy(task_container, binary, tb._CODEX_UPLOAD, forbidden=forbidden)
        for command in tb.install_commands(
            "codex-cli 0.141.0",
            codex_binary_sha256=_EXPECTED_CODEX_SHA256,
            codex_target="x86_64-unknown-linux-musl",
        ):
            _run(
                [
                    "docker",
                    "exec",
                    "--user",
                    "0:0",
                    task_container,
                    "/bin/bash",
                    "-c",
                    command,
                ],
                forbidden=forbidden,
            )
        _run(
            [
                "docker",
                "exec",
                "--user",
                "60000:60000",
                task_container,
                "/bin/bash",
                "-c",
                tb.process_isolation_check_command(),
            ],
            forbidden=forbidden,
        )
        _run(
            [
                "docker",
                "exec",
                "--user",
                "0:0",
                task_container,
                "/bin/bash",
                "-c",
                "mkdir -p /work; chown 60000:60000 /work; chmod 700 /work",
            ],
            forbidden=forbidden,
        )
        version = _run(
            [
                "docker",
                "exec",
                "--user",
                "60000:60000",
                task_container,
                "/usr/local/bin/codex",
                "--version",
            ],
            forbidden=forbidden,
        )
        assert version.stdout.strip() == "codex-cli 0.141.0"

        probe_command = _probe_command()
        server_source = tmp_path / "fake_broker.py"
        server_source.write_text(_fake_broker_source(probe_command))
        capture_source = tmp_path / "capture.pl"
        capture_source.write_text(_capture_source())
        startup = tmp_path / "startup.json"
        startup.write_text(
            json.dumps(
                {
                    "capability": capability,
                    "container_id": container_id,
                    "certificate_sha256": certificate_sha256,
                },
                sort_keys=True,
            )
        )
        startup.chmod(0o600)
        certificate = tb._write_private_payload(
            certificate_pem,
            prefix="lha-isolation-cert-",
        )
        private_key_file = tb._write_private_payload(
            bytes(private_key),
            prefix="lha-isolation-key-",
        )
        try:
            _copy(
                broker_container,
                server_source,
                "/tmp/fake_broker.py",
                forbidden=forbidden,
            )
            _copy(
                broker_container,
                startup,
                "/tmp/lha-broker-startup.json",
                forbidden=forbidden,
            )
            _copy(
                broker_container,
                certificate,
                "/tmp/lha-broker-cert.pem",
                forbidden=forbidden,
            )
            _copy(
                broker_container,
                private_key_file,
                "/tmp/lha-broker-key.pem",
                forbidden=forbidden,
            )
        finally:
            certificate.unlink(missing_ok=True)
            private_key_file.unlink(missing_ok=True)
            for index in range(len(private_key)):
                private_key[index] = 0
            private_key.clear()

        _run(
            [
                "docker",
                "exec",
                "--detach",
                "--user",
                "0:0",
                broker_container,
                "/usr/local/bin/python3",
                "/tmp/fake_broker.py",
            ],
            forbidden=forbidden,
        )
        _wait_for_file(
            broker_container,
            "/tmp/lha-broker-ready",
            forbidden=forbidden,
        )

        _install_private_file(
            container=task_container,
            payload=capability.encode(),
            prefix="lha-isolation-capability-",
            staging_path=tb._CAPABILITY_STAGING,
            destination_path=tb._CAPABILITY_UPLOAD,
            owner="60000:60000",
            forbidden=forbidden,
        )
        _install_private_file(
            container=task_container,
            payload=certificate_pem,
            prefix="lha-isolation-ca-",
            staging_path=tb._TLS_CERT_STAGING,
            destination_path=tb._TLS_CERT_PATH,
            owner="60000:60000",
            forbidden=forbidden,
        )
        _install_private_file(
            container=task_container,
            payload=wrong_certificate_pem,
            prefix="lha-isolation-wrong-ca-",
            staging_path=f"{tb._RUNTIME_STAGING_DIR}/wrong-ca.upload",
            destination_path="/tmp/.lha_wrong_ca.pem",
            owner="60000:60000",
            forbidden=forbidden,
        )
        _install_private_file(
            container=task_container,
            payload=certificate_pem,
            prefix="lha-isolation-probe-ca-",
            staging_path=f"{tb._RUNTIME_STAGING_DIR}/original-ca.upload",
            destination_path="/tmp/.lha_original_ca.pem",
            owner="60000:60000",
            forbidden=forbidden,
        )

        _copy(
            task_container,
            capture_source,
            "/tmp/lha-capture.pl",
            forbidden=forbidden,
        )
        _run(
            [
                "docker",
                "exec",
                "--detach",
                "--user",
                "0:0",
                task_container,
                "/usr/bin/perl",
                "/tmp/lha-capture.pl",
            ],
            forbidden=forbidden,
        )
        _wait_for_file(
            task_container,
            "/tmp/lha-capture-ready",
            forbidden=forbidden,
        )

        command = tb.codex_exec_command(
            "gpt-5.5",
            "xhigh",
            "Run the requested isolation probe, then report its completion.",
            proxy_base_url=f"https://{proxy.BROKER_ALIAS}:{proxy.BROKER_PORT}",
            binding_headers={
                "X-LHA-Evaluation-ID": _EVALUATION_ID,
                "X-LHA-Attempt-ID": _ATTEMPT_ID,
                "X-LHA-Container-ID": container_id,
            },
        )
        codex = _run(
            [
                "docker",
                "exec",
                "--user",
                "60000:60000",
                "--workdir",
                "/work",
                task_container,
                "/bin/bash",
                "-c",
                command,
            ],
            timeout=180,
            check=False,
            forbidden=forbidden,
        )
        combined = codex.stdout + codex.stderr
        assert capability not in combined
        assert codex.returncode == 0, _redact(combined[-5000:], forbidden)

        _run(
            [
                "docker",
                "exec",
                "--user",
                "0:0",
                task_container,
                "touch",
                "/tmp/lha-capture-stop",
            ],
            forbidden=forbidden,
        )
        _wait_for_file(
            task_container,
            "/tmp/lha-capture-done",
            forbidden=forbidden,
        )

        probe_raw = _run(
            [
                "docker",
                "exec",
                "--user",
                "0:0",
                task_container,
                "cat",
                "/tmp/lha-capability-probe.json",
            ],
            forbidden=forbidden,
        ).stdout
        probe_result = json.loads(probe_raw)
        assert probe_result["ruid"] == 60000
        assert probe_result["euid"] == 0
        assert probe_result["rgid"] == 60000
        assert probe_result["egid"] == 60000
        assert probe_result["capability_found_in_proc"] is False
        assert probe_result["capability_upload_exists"] is False
        assert probe_result["capability_upload_readable"] is False
        assert probe_result["codex_ancestor_found"] is True
        assert probe_result["codex_environ_readable"] is False
        assert probe_result["codex_mem_openable"] is False
        assert probe_result["direct_capability_reuse_status"] == 403
        assert probe_result["plaintext_http_failed"] is True
        assert probe_result["wrong_ca_failed"] is True
        assert probe_result["ca_path_present_before_probe"] is False
        # Bash may reuse descriptor number 3 for its script input, but the
        # Codex-held CA descriptor itself must not cross the tool boundary.
        assert probe_result["tls_fd_matches_original"] is False
        assert probe_result["tls_fd_write_reopen_succeeded"] is False
        assert probe_result["ca_overwritten"] is True
        assert probe_result["ca_deleted"] is True

        request_rows = [
            json.loads(line)
            for line in _run(
                [
                    "docker",
                    "exec",
                    "--user",
                    "0:0",
                    broker_container,
                    "cat",
                    "/tmp/lha-broker-requests.jsonl",
                ],
                forbidden=forbidden,
            ).stdout.splitlines()
        ]
        model_rows = [row for row in request_rows if not row["is_probe"]]
        probe_rows = [row for row in request_rows if row["is_probe"]]
        assert len(model_rows) == 2
        assert all(
            row["authorized"]
            and row["binding_ok"]
            and row["model_ok"]
            and row["effort_ok"]
            for row in model_rows
        )
        assert len(probe_rows) == 1
        assert probe_rows[0]["authorized"] is False

        wire_path = tmp_path / "wire.bin"
        _run(
            ["docker", "cp", f"{task_container}:/tmp/lha-wire.bin", str(wire_path)],
            forbidden=forbidden,
        )
        wire = wire_path.read_bytes()
        assert wire
        assert capability.encode() not in wire
        assert b"Authorization: Bearer" not in wire

        observable = [
            combined,
            probe_raw,
            json.dumps(request_rows, sort_keys=True),
        ]
        for container in (task_container, broker_container):
            observable.append(
                _run(["docker", "logs", container], forbidden=forbidden).stdout
            )
            observable.append(
                _run(["docker", "inspect", container], forbidden=forbidden).stdout
            )
        observable.append(
            _run(["docker", "network", "inspect", network], forbidden=forbidden).stdout
        )
        assert all(capability not in value for value in observable)
    finally:
        for index in range(len(private_key)):
            private_key[index] = 0
        private_key.clear()
        if broker_created:
            _run(["docker", "rm", "--force", broker_container], check=False)
        if task_created:
            _run(["docker", "rm", "--force", task_container], check=False)
        if network_created:
            _run(["docker", "network", "rm", network], check=False)
