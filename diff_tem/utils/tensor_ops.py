"""
This module contains some useful operators for tensors

CORRESPOND: array.cpp
"""
from typing import Union

import torch
import torch.nn.functional as F


def tensor_indexing(indexee: torch.Tensor, indexer: torch.Tensor, index_dim_num: int, only_indices: bool = False):
    """
    Index into a tensor with a tensor storing indices

    :param indexee: The tensor to be indexed, of shape (d_0, ... ,d_n-1)
    :param indexer: Indices, of shape (*indexer_leading_shape, index_dim_num)
    :param index_dim_num: Number of leading dimensions that are to be indexed
    :param only_indices: whether only output indices
    :return: Tensor of shape (*indexer_leading_shape, d_(index_dim_num), ..., d_n-1) and indices
    """
    assert indexer.dtype == torch.int or indexer.dtype == torch.long
    assert indexee.device == indexer.device
    assert len(indexee.shape) >= index_dim_num
    assert indexer.shape[-1] == index_dim_num
    axes = list(range(len(indexer.shape)))
    last_axis = axes.pop()
    axes.insert(0, last_axis)
    indices = indexer.permute(*axes).long()
    indices = tuple(index_tensor for index_tensor in indices)
    if only_indices:
        return indices
    else:
        return indexee[indices], indices


def change_axes(tensor: torch.Tensor, axis_order: str):
    dim = len(tensor.shape)
    assert dim == 2 or dim == 3, "Only support 2D or 3D tensor"
    if dim == 3:
        valid_options = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
    else:
        valid_options = ["xy", "yx"]
    if axis_order in valid_options:
        axes = []
        for axis in axis_order:
            if axis == "x":
                axes.append(0)
            elif axis == "y":
                axes.append(1)
            else:
                axes.append(2)
        tensor = tensor.permute(*axes)
        return tensor
    else:
        raise Exception(f"Invalid axis order = {axis_order}, valid options = {valid_options}")


def boundary_mean(t: torch.Tensor):
    """
    Compute the mean value of entries along the boundary of array.
    Boundary elements are those for which at least one of the indices is either 0 or its maximal value.

    CORRESPOND: boundary_mean_array()

    :param t: 3D tensor
    :return: mean value
    """
    assert len(t.shape) == 3, "This function only applies to 3D tensors"
    x_slice0 = t[0, :, :].flatten()
    x_slice1 = t[-1, :, :].flatten()
    y_slice0 = t[1:-1, 0, :].flatten()
    y_slice1 = t[1:-1, -1, :].flatten()
    z_slice0 = t[1:-1, 1:-1, 0].flatten()
    z_slice1 = t[1:-1, 1:-1, -1].flatten()
    all_boundary_elements = torch.cat([x_slice0, x_slice1,
                                       y_slice0, y_slice1,
                                       z_slice0, z_slice1])
    return all_boundary_elements.mean()


class DiscreteLaplace(torch.nn.Module):
    """
    PyTorch operator of Discrete Laplace

    CORRESPOND: laplace_array()
    """

    def __init__(self, _requires_grad=False):
        super(DiscreteLaplace, self).__init__()
        weight_tensor = torch.tensor([
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]],
            [[0, 1, 0],
             [1, -6, 1],
             [0, 1, 0]],
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]],
        ], dtype=torch.double)
        self.weights = torch.nn.Parameter(weight_tensor.reshape(1, 1, 3, 3, 3), requires_grad=_requires_grad)

    def forward(self, tensor_original):
        tensor_dim = len(tensor_original.shape)
        if tensor_dim == 3:
            tensor = tensor_original.reshape(1, 1, *tensor_original.shape).to(self.weights.dtype)
        elif tensor_dim == 4:
            tensor = tensor_original.unsqueeze(1).to(self.weights.dtype)
        else:
            raise Exception(f"Invalid tensor of shape {tensor_original.shape}")
        return F.conv3d(tensor, self.weights, padding="same").reshape(*tensor_original.shape)  # pad with zeros


def absolute(x: Union[torch.Tensor, float]):
    if isinstance(x, torch.Tensor):
        return torch.abs(x)
    else:
        return abs(x)
