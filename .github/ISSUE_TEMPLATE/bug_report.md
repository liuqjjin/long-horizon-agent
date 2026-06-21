---
name: Bug report
about: Something doesn't behave as documented
title: "bug: "
labels: bug
---

**What happened vs. what you expected**

**Reproduce** (commands, task spec, expected vs. actual output)

```bash
# e.g.
uv run lha run data/tasks/fix_average.yaml
```

**Gate status** — paste the relevant output:

```
uv run pytest -q
uv run lha eval
```

**Environment:** OS, Python (`python --version`), `uv --version`, and whether
`cocoindex-code` (`ccc`) is installed.
