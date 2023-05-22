import torch
from ..utils.transformations import rotate_3d_rows
from ..constants import PI

def test_rotate_3d_rows():
    x_axis = torch.tensor([1., 0., 0.])
    z_axis = torch.tensor([0., 0., 1.])
    angles = torch.tensor([0., 90., 90.])
    angles = torch.deg2rad(angles)
    r0 = rotate_3d_rows(z_axis, -angles[2])
    r1 = rotate_3d_rows(x_axis, -angles[1])
    r2 = rotate_3d_rows(z_axis, -angles[0])
    rotate_matrix = r2 @ r1 @ r0
    print(rotate_matrix)
