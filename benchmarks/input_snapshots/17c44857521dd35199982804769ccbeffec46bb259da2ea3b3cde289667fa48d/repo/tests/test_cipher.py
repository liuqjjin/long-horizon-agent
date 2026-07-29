from cipher import encrypt


def test_basic_lowercase_shift():
    assert encrypt("abc", 1) == "bcd"


def test_basic_uppercase_shift():
    assert encrypt("ABC", 2) == "CDE"


def test_lowercase_wraps_at_end_of_alphabet():
    # 'z' shifted by 1 must wrap to 'a', not jump to the next code point.
    assert encrypt("z", 1) == "a"
    assert encrypt("xyz", 3) == "abc"


def test_uppercase_wraps_at_end_of_alphabet():
    assert encrypt("Z", 1) == "A"
    assert encrypt("XYZ", 3) == "ABC"


def test_non_letters_are_left_unchanged():
    assert encrypt("a b!", 1) == "b c!"
    assert encrypt("abc123", 1) == "bcd123"
    assert encrypt("Hello, World!", 3) == "Khoor, Zruog!"


def test_zero_shift_is_identity():
    assert encrypt("Hello, World!", 0) == "Hello, World!"
    assert encrypt("", 5) == ""


def test_shift_of_26_is_identity():
    assert encrypt("abcXYZ", 26) == "abcXYZ"


def test_shift_larger_than_alphabet_wraps_fully():
    # Naive single-step wrapping ("subtract 26 once") fails for big shifts.
    assert encrypt("a", 27) == "b"
    assert encrypt("a", 53) == "b"


def test_negative_shift_wraps_backwards():
    assert encrypt("a", -1) == "z"
    assert encrypt("A", -1) == "Z"
    assert encrypt("b", -3) == "y"


def test_large_negative_shift_wraps_fully():
    assert encrypt("a", -27) == "z"
