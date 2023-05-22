import torch
import numpy as np

from diff_tem.utils.tensor_io import *


def test_read_matrix():
    file_path = "diff_tem/tests/matrix_file.txt"
    matrix = read_matrix_text(file_path)
    ref_matrix = torch.arange(4, dtype=torch.float).reshape(2, 2) + 1
    assert torch.allclose(matrix, ref_matrix)


def test_write_matrix():
    file_path = ".pytest_cache/test_file.txt"
    matrix = torch.arange(4).float().reshape(2, 2) + 2
    write_matrix_text_with_headers(matrix, file_path, "column1", "column2")
    read_matrix = read_matrix_text(file_path)
    assert torch.allclose(read_matrix, matrix)


def test_write_tensor():
    file_path = ".pytest_cache/tensor.bin"
    tensor = torch.ones(3, 3, 3, dtype=torch.float)
    write_tensor_to_binary(tensor, file_path)
    t = read_tensor_from_binary(file_path, torch.float, (3, 3, 3))
    assert torch.allclose(t, tensor)
