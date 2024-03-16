from .variable import Variable, OptimizedException


class CategoricalVariable(Variable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        return len(self.values)

    def next(self, scores: dict[None, float], optimize_method=min):
        for value in self.values:
            if scores.get(value) is None:
                return value
        raise OptimizedException("all values tested")
