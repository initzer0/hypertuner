import os
import sys
import math

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

from hypertuner.variables.variable import OptimizedException
from hypertuner.variables.interval import IntervalVariable
from hypertuner.variables.categorical import CategoricalVariable
from hypertuner.hypertuner.hypertuner import HyperTuner


def test_hypertuner_single(verbose=False):
    ht = HyperTuner(optimize_method=min)

    ht.add_parameter(
        IntervalVariable(values=[-1, 1, 0.05], name="x", required_neighbors=2)
    )

    ht.run_model = lambda params: (params["x"] - 0.2) ** 2

    best_params = ht.run()
    assert math.isclose(best_params["x"], 0.2), f"{best_params['x']} != 0.2"
    if verbose:
        print("single best params:", best_params)


def test_hypertuner_bounds_upper(verbose=False):
    ht = HyperTuner(optimize_method=min)

    ht.add_parameter(
        IntervalVariable(values=[-1, 1, 0.05], name="x", required_neighbors=2)
    )

    ht.run_model = lambda params: (params["x"] - 1.0) ** 2

    best_params = ht.run()
    assert math.isclose(best_params["x"], 1.0), f"{best_params['x']} != 1.0"
    if verbose:
        print("bounds best params:", best_params)


def test_hypertuner_bounds_lower(verbose=False):
    ht = HyperTuner(optimize_method=min)

    ht.add_parameter(
        IntervalVariable(values=[-1, 1, 0.05], name="x", required_neighbors=2)
    )

    ht.run_model = lambda params: (params["x"] + 1.0) ** 2

    best_params = ht.run()
    assert math.isclose(best_params["x"], -1.0), f"{best_params['x']} != -1.0"
    if verbose:
        print("bounds best params:", best_params)


def test_hypertuner_multi(verbose=False):
    ht = HyperTuner(optimize_method=min)

    ht.add_parameter(CategoricalVariable(values=[-1, 1], name="a"))
    ht.add_parameter(
        IntervalVariable(values=[-1, 1, 0.05], name="x", required_neighbors=2)
    )
    ht.add_parameter(
        IntervalVariable(values=[-1, 1, 0.05], name="y", required_neighbors=2)
    )

    ht.run_model = (
        lambda params: params["a"] + (0.1 - params["x"]) ** 2 + (params["y"] - 0.4) ** 4
    )

    best_params = ht.run()

    assert math.isclose(best_params["a"], -1), f"{best_params['a']} != -1"
    assert math.isclose(best_params["x"], 0.1), f"{best_params['x']} != 0.1"
    assert math.isclose(best_params["y"], 0.4), f"{best_params['y']} != 0.4"

    if verbose:
        print("multi best params:", best_params)


if __name__ == "__main__":
    test_hypertuner_single(verbose=True)
    test_hypertuner_bounds_upper(verbose=True)
    test_hypertuner_bounds_lower(verbose=True)
    test_hypertuner_multi(verbose=True)
