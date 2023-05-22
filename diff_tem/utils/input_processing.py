from ..constants.engineering import COMMENT_CHAR
import json


def convert_bool(string: str):
    if string.lower() in ["yes", "true", "1"]:
        return True
    if string.lower() in ["no", "false", "0"]:
        return False
    raise Exception(f"Invalid value {string}")


def read_input_file(filepath: str):
    """
    Read configurations of a simulation from a txt or json file
    :param filepath: file path
    :return: a dictionary containing {component_type: [parameters of each instance]}
    """
    if filepath.endswith(".txt"):
        return read_input_file_legacy(filepath)
    elif filepath.endswith(".json"):
        return read_input_file_json(filepath)
    else:
        raise Exception(f"Only support json or txt input file now, got {filepath}")


def read_input_file_legacy(filepath):
    """
    Read configurations from a txt file.
    The format of the input txt file is specified in TEM simulator manual

    :param filepath: file path
    :return: a dictionary containing {component_type: [parameters of each instance]}
    """
    with open(filepath, "r") as f:
        strings = f.readlines()
    valid_strings = list(filter(lambda x: not x.startswith(COMMENT_CHAR) and len(x) > 1, strings))
    component_parameter_list = []
    component_parameters = {}
    for s in valid_strings:
        if s.startswith("==="):
            component_parameters = {}
            component_type = s.replace("===", "").strip()
            if component_type.startswith("particle "):
                component_type, component_name = component_type.split(" ")
                component_parameters["name"] = component_name
            component_parameters["component"] = component_type
            component_parameter_list.append(component_parameters)
        else:
            parameter_name, parameter_value = s.split("=")
            parameter_name = parameter_name.strip()
            parameter_value = parameter_value.strip()
            component_parameters[parameter_name] = parameter_value

    components = {}
    for parameter_set in component_parameter_list:
        component_type = parameter_set.pop("component")
        if component_type in components:
            components[component_type].append(parameter_set)
        else:
            components[component_type] = [parameter_set]
    return components


def export_input_configs_json(component_parameter_table, file_path):
    with open(file_path, "w") as f:
        json.dump(component_parameter_table, f, indent=4)


def read_input_file_json(filepath):
    with open(filepath, "r") as f:
        return json.load(f)
