from hypertuner.variables.variable import Variable


class PerformanceRegistry:
    def __init__(self):
        self.parameter_sets: list[dict[Variable, None]] = []
        self.parameter_scores: list[float] = []

    def __contains__(self, parameters: dict[Variable, None]):
        return parameters in self.parameter_sets

    def add_score(self, parameters: dict[Variable, None], score):
        self.parameter_sets.append(parameters)
        self.parameter_scores.append(score)

    def get_score(self, parameters: dict[Variable, None]):
        i = self.parameter_sets.index(parameters)
        return self.parameter_scores[i]

    def match_scores(self, parameters: dict[Variable, None]):
        for i, param_set in enumerate(self.parameter_sets):
            for key, val1 in parameters.items():
                if param_set[key] != val1:
                    break
            else:
                yield param_set, self.parameter_scores[i]
