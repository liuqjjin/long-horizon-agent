from rpn import eval_rpn


def test_single_number_returns_float():
    assert eval_rpn(["42"]) == 42.0


def test_addition_and_multiplication():
    assert eval_rpn(["2", "3", "+"]) == 5.0
    assert eval_rpn(["2", "3", "*"]) == 6.0


def test_subtraction_operand_order():
    # "5 3 -" means 5 - 3
    assert eval_rpn(["5", "3", "-"]) == 2.0


def test_subtraction_keeps_sign():
    # "3 5 -" must be -2, not 2 -- guards against an abs()/negate "fix".
    assert eval_rpn(["3", "5", "-"]) == -2.0


def test_division_operand_order():
    assert eval_rpn(["6", "3", "/"]) == 2.0
    assert eval_rpn(["8", "2", "/"]) == 4.0


def test_division_fraction_and_zero_numerator():
    assert eval_rpn(["3", "6", "/"]) == 0.5
    assert eval_rpn(["0", "4", "/"]) == 0.0


def test_compound_expression_order():
    # (10 - 4) / 2 == 3
    assert eval_rpn(["10", "4", "-", "2", "/"]) == 3.0


def test_negative_floats():
    assert eval_rpn(["2.5", "0.5", "-"]) == 2.0
    assert eval_rpn(["1", "4", "-"]) == -3.0
