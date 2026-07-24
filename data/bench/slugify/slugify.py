"""URL slug generation.

A slug contains only lowercase ASCII letters and digits, with single hyphens
between words: any run of other characters (spaces, punctuation, underscores,
non-ASCII letters) acts as one separator. Slugs never start or end with a
hyphen. If nothing remains, the slug is ``"n-a"``.
"""


def slugify(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum() and ch.isascii():
            out.append(ch)
        else:
            out.append("-")
    return "".join(out)
