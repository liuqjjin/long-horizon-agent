from flatten import flatten


def test_basic_nesting():
    # Plain nested lists/tuples collapse to a single flat list.
    assert flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]
    assert flatten([1, [2, (3, [4])]]) == [1, 2, 3, 4]


def test_tuples_are_flattened():
    # Tuples must still be recursed into; a fix that only handles lists
    # would wrongly leave the tuple untouched here.
    assert flatten([1, (2, 3), 4]) == [1, 2, 3, 4]


def test_strings_are_atomic():
    # The headline case: a string is a single value, not a bag of chars.
    assert flatten([1, ["ab", 2]]) == [1, "ab", 2]
    assert flatten(["hello"]) == ["hello"]
    assert flatten([["a", ["bc"]], "d"]) == ["a", "bc", "d"]
    # Whitespace inside a string must be preserved as part of that string.
    assert flatten(["a b", ["c"]]) == ["a b", "c"]


def test_empty_string_is_preserved():
    # An empty string is a real element; it must not silently disappear.
    assert flatten(["", 5]) == ["", 5]


def test_bytes_are_atomic():
    # bytes are also atomic; a fix that special-cases only str would wrongly
    # explode bytes into their integer code points.
    assert flatten([b"ab", 1]) == [b"ab", 1]


def test_empty_inputs():
    assert flatten([]) == []
    assert flatten([[], [[], []]]) == []
