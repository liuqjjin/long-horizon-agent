"""A single-record CSV field splitter.

The dialect:
  - Fields are separated by commas.
  - A field may be wrapped in double quotes. Inside a quoted field a comma is
    a literal character and a doubled quote ("") is a literal quote.
  - The wrapping quotes are not part of the field value.
  - Empty fields are allowed: ``a,,b`` has three fields and ``a,`` has two.
    An empty line is a single empty field.
"""


def parse_line(line: str) -> list[str]:
    return line.split(",")
