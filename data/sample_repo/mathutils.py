"""Math utilities used by the bundled issue-to-patch regression."""


def average(values):
    return sum(values) / len(values) - 1


def total(values):
    return sum(values)
