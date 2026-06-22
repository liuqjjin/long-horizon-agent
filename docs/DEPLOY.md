# Deploy

This repo ships as a reproducible CLI image, not a hosted web service. Build from
the repo root:

```bash
docker build -t lha .
```

The default command prints CLI help:

```bash
docker run --rm lha
```

## Smoke Test

Run the canonical self-test in the container:

```bash
docker run --rm lha uv run lha eval
```

Expected result: `ResearchAgentBench-Lite` reports `5/5`. The first `eval` run
downloads a small Hugging Face embedding model (tens of MB) into
`/home/lha/.cache/huggingface`.

Keep that cache across runs with a named volume:

```bash
docker volume create lha-hf
docker run --rm \
  -v lha-hf:/home/lha/.cache/huggingface \
  lha uv run lha eval
```

## Persist Outputs

Runs are written under `/app/runs`. Mount a local `runs/` directory when you want
the verifier trail and summaries to survive container exit:

```bash
mkdir -p runs
docker run --rm \
  -v lha-hf:/home/lha/.cache/huggingface \
  -v "$PWD/runs:/app/runs" \
  lha uv run lha run data/tasks/fix_average.yaml
```

## Optional Code Search

`ccc` (`cocoindex-code`) is intentionally absent from the image. It is an
optional external `pipx` tool for code search; the bundled demos and
`uv run lha eval` do not need it.
