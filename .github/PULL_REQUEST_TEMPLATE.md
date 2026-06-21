## What & why

<!-- A short description of the change and the motivation. -->

## Verification

<!-- No claim without a runnable check. Note what you ran and any new checks added. -->

- [ ] `uv run ruff check .` clean
- [ ] `uv run pyright src/lha` clean
- [ ] `uv run pytest -q` passes
- [ ] `uv run lha eval` is `5/5`
- [ ] facade isolation grep prints nothing (no CocoIndex import outside `live_context/`)
- [ ] no test was weakened/skipped and no threshold loosened to pass
- [ ] docs updated if behavior/claims changed
