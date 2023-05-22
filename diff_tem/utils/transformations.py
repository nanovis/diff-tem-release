import torch

from ..constants import PI


def rand_orientation(length: int):
    random_orientations = torch.rand(length, 3)
    random_orientations[:, 0] *= 2. * PI
    random_orientations[:, 1] = torch.acos(random_orientations[:, 1] * 2 - 1)
    random_orientations[:, 2] *= 2.0 * PI
    return random_orientations


def rotate_3d_rows(rotate_axis: torch.Tensor, angle_radian: torch.Tensor):
    assert rotate_axis.shape[0] == 3 and len(rotate_axis.shape) == 1
    normalized_rotate_axis = rotate_axis / torch.sqrt((rotate_axis ** 2).sum())
    x = normalized_rotate_axis[0]
    y = normalized_rotate_axis[1]
    z = normalized_rotate_axis[2]
    cos = torch.cos(angle_radian)
    sin = torch.sin(angle_radian)
    rotation_matrix = torch.zeros(3, 3)
    rotation_matrix[0, 0] = cos + x * x * (1 - cos)
    rotation_matrix[0, 1] = x * y * (1 - cos) - z * sin
    rotation_matrix[0, 2] = x * z * (1 - cos) + y * sin
    rotation_matrix[1, 0] = x * y * (1 - cos) + z * sin
    rotation_matrix[1, 1] = cos + y * y * (1 - cos)
    rotation_matrix[1, 2] = y * z * (1 - cos) - x * sin
    rotation_matrix[2, 0] = x * z * (1 - cos) - y * sin
    rotation_matrix[2, 1] = y * z * (1 - cos) + x * sin
    rotation_matrix[2, 2] = cos + z * z * (1 - cos)
    return rotation_matrix
