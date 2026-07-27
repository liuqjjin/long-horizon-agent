FROM python@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/lha

COPY docker/terminal-bench-proxy-requirements.txt /opt/lha/
RUN python -m pip install \
      --disable-pip-version-check \
      --no-cache-dir \
      --no-deps \
      --only-binary=:all: \
      --require-hashes \
      --requirement /opt/lha/terminal-bench-proxy-requirements.txt \
    && python -m pip check

COPY --chown=65532:65532 src/lha /opt/lha/lha

USER 65532:65532
ENTRYPOINT ["python", "-m", "lha.bench.terminal_proxy_server"]
