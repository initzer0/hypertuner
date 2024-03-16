import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

from hypertuner.variables.variable import OptimizedException
from hypertuner.variables.interval import IntervalVariable
from hypertuner.variables.categorical import CategoricalVariable


def test_categorical(verbose=False):
    values = ["a", "b", "c"]
    var = CategoricalVariable(values=values)

    assert len(var) == len(values), f"length wrong {len(var)} != {len(values)}"
    if verbose:
        print("categorical length test passed")

    scores = {}
    for val in values:
        value = var.next(scores)
        assert value == val, f"value is {value!r} != {val!r}"
        scores[value] = 1.0

    if verbose:
        print("categorical next test passed")

    try:
        value = var.next(scores)
    except OptimizedException:
        if verbose:
            print("categorical next error test passed")
    else:
        assert False, "should have thrown OptimizedException"


def test_interval(verbose=False):
    var = IntervalVariable(
        values=[0, 100, 1],
        initial_value=25,
        required_neighbors=1,
    )

    assert len(var) == 9, f"length wrong {len(var)} != {9}"
    if verbose:
        print("interval length test passed")

    eval_func = lambda x: (x - 2) ** 2 + x - 1

    scores = {}
    expected = 25
    value = var.next(scores)
    assert value == expected, f"value is {value!r} != {expected!r}"
    scores[value] = eval_func(value)

    expected = 0
    value = var.next(scores)
    assert value == expected, f"value is {value!r} != {expected!r}"
    scores[value] = eval_func(value)

    expected = 100
    value = var.next(scores)
    assert value == expected, f"value is {value!r} != {expected!r}"
    scores[value] = eval_func(value)

    if verbose:
        print("interval next test passed")

    for _ in range(6):
        value = var.next(scores)
        scores[value] = eval_func(value)

    try:
        value = var.next(scores)
    except OptimizedException:
        if verbose:
            print("interval next error test passed")
    else:
        assert False, "should have thrown OptimizedException"


if __name__ == "__main__":
    test_categorical(verbose=True)
    test_interval(verbose=True)
