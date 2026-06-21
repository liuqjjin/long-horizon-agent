from mathutils import average, total


def test_average():
    assert average([2, 4, 6]) == 4


def test_total():
    assert total([1, 2, 3]) == 6
