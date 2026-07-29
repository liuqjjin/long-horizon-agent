"""Evaluate reverse-polish-notation (postfix) expressions over floats."""

_OPERATORS = {"+", "-", "*", "/"}


def eval_rpn(tokens):
    """Evaluate ``tokens`` written in reverse polish notation.

    ``tokens`` is a sequence of strings made up of numeric literals and the
    binary operators ``+``, ``-``, ``*`` and ``/``. The result is returned as
    a float.
    """
    stack = []
    for token in tokens:
        if token in _OPERATORS:
            a = stack.pop()
            b = stack.pop()
            stack.append(_apply(token, a, b))
        else:
            stack.append(float(token))
    return stack.pop()


def _apply(op, a, b):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    return a / b
