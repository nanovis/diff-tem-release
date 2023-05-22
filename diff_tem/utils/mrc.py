import mrcfile
import torch

from .tensor_ops import change_axes
from ..constants.derived_units import ONE_ANGSTROM


def write_tensor_to_mrc(tensor: torch.Tensor,
                        filepath: str,
                        voxel_size: float = None,
                        as_type: torch.dtype = None,
                        axis_order: str = None):
    if voxel_size is not None:
        voxel_size /= ONE_ANGSTROM
    dim = len(tensor.shape)
    assert dim == 2 or dim == 3, "Only support 2D or 3D tensor"
    if axis_order is not None:
        tensor = change_axes(tensor, axis_order)
    if as_type is not None:
        tensor = tensor.to(as_type)
    with mrcfile.new(filepath, overwrite=True) as mrc:
        if voxel_size is not None:
            mrc.voxel_size = voxel_size
        mrc.set_data(tensor.numpy())
