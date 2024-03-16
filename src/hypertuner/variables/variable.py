def argm_dict(dict_, method=min):
    m = method.__call__(dict_.values())
    for k, v in dict_.items():
        if m == v:
            return k


class OptimizedException(Exception):
    pass


class Variable:
    _existing_names = []

    def __init__(
        self, values=None, initial_value=None, name=None, method=None, priority=0
    ):
        self.values = values
        self.initial_value = initial_value

        self.method = method
        self.priority = priority

        if name is None:
            name = self.get_new_name()
        if name in Variable._existing_names:
            raise ValueError(f"name {name!r} already exists")
        Variable._existing_names.append(name)
        self.name = name

    @classmethod
    def get_new_name(cls) -> str:
        offset = 0
        name = f"var_{len(Variable._existing_names) + offset}"
        while name in Variable._existing_names:
            offset += 1
            name = f"var_{len(Variable._existing_names) + offset}"
        return name

    def __del__(self):
        if self.name in Variable._existing_names:
            Variable._existing_names.remove(self.name)

    def __repr__(self):
        return (
            f"<{self.__class__.__qualname__} name={self.name!r} value={self.values!r}>"
        )

    def __str__(self):
        return self.name

    def __eq__(self, other):
        if type(other) == str:
            return self.name == other
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __lt__(self, other):
        if self.priority == other.priority:
            # fewer values means earlier
            return len(self) < len(other)
        else:
            # higher priority means earlier
            return -self.priority < -other.priority

    def __len__(self):
        return 0

    def next(self, scores: dict[None, float], optimize_method=min):
        raise NotImplementedError()

    def get_initial_value(self):
        if self.initial_value is None:
            return self.values[0]
        return self.initial_value
