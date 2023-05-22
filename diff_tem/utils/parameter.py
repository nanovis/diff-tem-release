class Parameter:

    def __init__(self, name: str, parameter_type, description=None):
        self.name = name
        self.type = parameter_type
        self.description = description

    def __str__(self):
        return f"Parameter: {self.name}\n  type: {self.type}\n  description: {self.description}"

    def __hash__(self):
        return self.name.__hash__()

    def __eq__(self, other):
        if isinstance(other, Parameter):
            return self.name == other.name
        else:
            return self.name == other
