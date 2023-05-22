from typing import Union

import torch

from ..constants.io_units import ANGLE_UNIT


def degree_to_radian(x: Union[float, torch.Tensor]):
    if isinstance(x, float):
        return x * ANGLE_UNIT
    if isinstance(x, torch.Tensor):
        return torch.deg2rad(x)
    raise Exception("Only accepts float or tensor")


def radian_to_degree(x: Union[float, torch.Tensor]):
    if isinstance(x, float):
        return x / ANGLE_UNIT
    if isinstance(x, torch.Tensor):
        return torch.rad2deg(x)
    raise Exception("Only accepts float or tensor")
