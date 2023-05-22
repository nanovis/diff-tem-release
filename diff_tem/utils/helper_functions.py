import sys
from typing import Union, Iterable

import torch


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def check_tensor(tensor: torch.Tensor,
                 dimensionality: Union[None, int] = None,
                 shape: Union[None, Iterable[int]] = None,
                 dtype: Union[None, torch.dtype] = None,
                 device: Union[None, torch.device] = None):
    """
    Check tensor with some criteria

    :param tensor: tensor to be check
    :param dimensionality: dimensionality requirement. For example, 1 means a vector, 2 means a matrix
    :param shape: shape requirement
    :param dtype: data type requirement
    :param device: device requirement
    """
    tensor_shape = tensor.shape
    if dimensionality is not None and shape is not None:
        assert len(shape) == dimensionality, "checking parameters not match"

    if dimensionality is not None:
        assert len(tensor_shape) == dimensionality, f"dimensionality not match, " \
                                                    f"expect = {dimensionality}, got = {len(tensor_shape)}"
    if shape is not None:
        assert tensor_shape == tuple(shape), f"shape not match, " \
                                             f"expect = {tuple(shape)}, got = {tensor_shape}"
    if dtype is not None:
        assert tensor.dtype == dtype, f"dtype not match, expect = {dtype}, got = {tensor.dtype}"

    if device is not None:
        assert tensor.device == device, f"device not match, expect = {device}, got = {tensor.device}"
