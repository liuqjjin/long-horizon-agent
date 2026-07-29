"""Roman numeral conversion."""

_SYMBOLS = [
    (1000, "M"),
    (500, "D"),
    (100, "C"),
    (50, "L"),
    (10, "X"),
    (5, "V"),
    (1, "I"),
]


def to_roman(number: int) -> str:
    """Return the Roman numeral for an integer in the range 1..3999."""
    if not isinstance(number, int) or isinstance(number, bool):
        raise TypeError("number must be an int")
    if number < 1 or number > 3999:
        raise ValueError("number must be between 1 and 3999")

    result = []
    remaining = number
    for value, symbol in _SYMBOLS:
        count, remaining = divmod(remaining, value)
        result.append(symbol * count)
    return "".join(result)
