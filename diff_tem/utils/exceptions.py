class WrongArgumentTypeException(Exception):
    def __init__(self, data, required_class):
        super(WrongArgumentTypeException, self).__init__(
            f"Wrong type, data is of type {type(data)}, but {required_class} is required")


class RequiredParameterNotSpecifiedException(Exception):
    def __init__(self, required_parameter_type):
        super(RequiredParameterNotSpecifiedException, self).__init__(
            f"The required {required_parameter_type} is not specified")


class ParameterLockedException(Exception):
    def __init__(self, parameter_type):
        super(ParameterLockedException, self).__init__(f"Try to modify {parameter_type} while parameters are locked")
