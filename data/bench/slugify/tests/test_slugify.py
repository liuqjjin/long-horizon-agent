from slugify import slugify


def test_simple_words():
    assert slugify("hello world") == "hello-world"


def test_lowercases():
    assert slugify("Hello World") == "hello-world"


def test_punctuation_run_is_one_separator():
    assert slugify("Hello, World!") == "hello-world"


def test_no_trailing_hyphen():
    assert slugify("Hello!") == "hello"


def test_no_leading_hyphen():
    assert slugify("--lead and trail--") == "lead-and-trail"


def test_multiple_spaces_collapse():
    assert slugify("  spaces   everywhere ") == "spaces-everywhere"


def test_non_ascii_letters_separate():
    assert slugify("café au lait") == "caf-au-lait"


def test_underscores_are_separators():
    assert slugify("a__b--c") == "a-b-c"


def test_digits_survive():
    assert slugify("Python 3.11") == "python-3-11"


def test_nothing_left_falls_back():
    assert slugify("!!!") == "n-a"
    assert slugify("") == "n-a"
