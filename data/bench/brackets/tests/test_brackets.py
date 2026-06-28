from brackets import is_balanced


def test_empty_string_is_balanced():
    assert is_balanced("") is True


def test_simple_pairs_balanced():
    assert is_balanced("()") is True
    assert is_balanced("[]") is True
    assert is_balanced("{}") is True


def test_sequenced_pairs_balanced():
    assert is_balanced("()[]{}") is True


def test_correctly_nested_is_balanced():
    assert is_balanced("([{}])") is True
    assert is_balanced("{[()]}") is True


def test_mismatched_closer_is_unbalanced():
    # Top of stack is '(', but the closer is ']' -> must be rejected.
    assert is_balanced("(]") is False
    assert is_balanced("{)") is False


def test_interleaved_brackets_are_unbalanced():
    # Closers appear in the wrong order relative to openers.
    assert is_balanced("([)]") is False


def test_leftover_openers_are_unbalanced():
    assert is_balanced("(((") is False
    assert is_balanced("([{") is False


def test_leftover_opener_after_some_matching():
    # Even one dangling opener must make the whole string unbalanced.
    assert is_balanced("(()") is False
    assert is_balanced("([]") is False


def test_extra_closer_is_unbalanced():
    assert is_balanced("())") is False
    assert is_balanced("]") is False


def test_non_bracket_characters_are_ignored():
    assert is_balanced("a(b)c") is True
    assert is_balanced("a(b]c") is False
