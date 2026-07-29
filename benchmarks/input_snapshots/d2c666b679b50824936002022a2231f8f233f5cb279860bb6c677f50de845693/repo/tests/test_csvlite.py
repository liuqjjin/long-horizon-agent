from csvlite import parse_line


def test_plain_fields():
    assert parse_line("a,b,c") == ["a", "b", "c"]


def test_quoted_field_keeps_its_comma():
    assert parse_line('a,"b,c",d') == ["a", "b,c", "d"]


def test_doubled_quote_is_a_literal_quote():
    assert parse_line('"say ""hi""",x') == ['say "hi"', "x"]


def test_empty_middle_field():
    assert parse_line("a,,b") == ["a", "", "b"]


def test_empty_trailing_field():
    assert parse_line("a,") == ["a", ""]


def test_quoted_empty_field():
    assert parse_line('a,"",b') == ["a", "", "b"]


def test_wrapping_quotes_are_stripped():
    assert parse_line('"a"') == ["a"]


def test_fully_quoted_single_field_with_comma():
    assert parse_line('"a,b"') == ["a,b"]


def test_empty_line_is_one_empty_field():
    assert parse_line("") == [""]


def test_quoted_field_at_end_of_line():
    assert parse_line('x,"y,z"') == ["x", "y,z"]
