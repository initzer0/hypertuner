import copy

from .performance_registry import PerformanceRegistry
from hypertuner.variables.variable import Variable, OptimizedException, argm_dict


class HyperTuner:

    def __init__(self, optimize_method=min):
        self.optimize_method = optimize_method
        self.parameters = []
        self.epochs = 1

        # save runs, so we dont run a model more than once
        self.performace_registry = PerformanceRegistry()

    def run_model(self, parameters: dict[Variable, None]):
        raise NotImplementedError()

    def _run_model(self, parameters: dict[Variable, None]):
        if parameters in self.performace_registry:
            return self.performace_registry.get_score(parameters)
        return self.run_model(parameters)

    def run_single_parameter(
        self, parameters: dict[Variable, None], to_optimize: Variable
    ):
        scores = {}

        sub_param = copy.deepcopy(parameters)
        del sub_param[to_optimize]
        for p, v in self.performace_registry.match_scores(sub_param):
            scores[p[to_optimize]] = v

        while True:
            try:
                next_param_value = to_optimize.next(
                    scores, optimize_method=self.optimize_method
                )
                run_params = copy.deepcopy(parameters)
                run_params[to_optimize] = next_param_value

                score = self._run_model(run_params)
                scores[next_param_value] = score
                self.performace_registry.add_score(run_params, score)

            except OptimizedException:
                break

        best_params = copy.deepcopy(parameters)
        best_value = argm_dict(scores, method=self.optimize_method)
        best_params[to_optimize] = best_value
        return best_params

    def run_epoch(self, parameters: dict[Variable, None]):
        params = sorted(self.parameters)
        for param in params:
            parameters = self.run_single_parameter(parameters, param)
        return parameters

    def run(self):
        parameters = {k: k.get_initial_value() for k in self.parameters}
        for epoch in range(self.epochs):
            parameters = self.run_epoch(parameters)
        return parameters

    def add_parameter(self, parameter):
        self.parameters.append(parameter)
