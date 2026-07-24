"""URL path joining.

``join_url(base, *parts)`` glues path segments onto a base URL with exactly
one slash at each boundary, regardless of how many slashes the caller left on
either side. The scheme separator (``https://``) is preserved. Empty parts are
skipped. The result ends with a slash only if the last non-empty part does.
Parts are always treated as relative segments — a leading slash on a part does
not restart from the host.
"""


def join_url(base: str, *parts: str) -> str:
    return "/".join([base, *parts])
