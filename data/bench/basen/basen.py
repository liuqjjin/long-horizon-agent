"""Base-n integer formatting helpers (pure stdlib)."""

_DIGITS = "0123456789abcdef"


def to_base(n: int, b: int) -> str:
    """Render the integer ``n`` in base ``b`` (2..16) using digits 0-9a-f."""
    out = ""
    while n > 0:
        out += _DIGITS[n % b]
        n //= b
    return out
