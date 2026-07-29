"""A tiny Caesar cipher utility."""


def encrypt(text: str, shift: int) -> str:
    """Return *text* with each character shifted by *shift* positions."""
    result = []
    for char in text:
        result.append(chr(ord(char) + shift))
    return "".join(result)
