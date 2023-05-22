"""
CORRESPOND: matrix IO utilities in matrix.cpp
"""
import sys
from typing import Optional, Iterable

import torch
import numpy as np
from ..constants.engineering import COMMENT_CHAR
from .. import VERSION
from .helper_functions import check_tensor
from enum import Enum


class ByteOrder(Enum):
    BIG_ENDIAN = "be"
    LITTLE_ENDIAN = "le"
    NATIVE = "native"

    @staticmethod
    def options():
        return ["be", "le", "native"]

    def to_np_representation(self):
        if self == ByteOrder.BIG_ENDIAN:
            return '>'
        elif self == ByteOrder.LITTLE_ENDIAN:
            return '<'
        else:
            return '='


def need_reverse_byte_order(byte_order: ByteOrder):
    if byte_order == ByteOrder.BIG_ENDIAN:
        return sys.byteorder == "little"
    elif byte_order == ByteOrder.LITTLE_ENDIAN:
        return sys.byteorder == "big"
    else:
        return False


def read_matrix_text(filepath):
    """
    Read a matrix from a txt file,
    whose first line species dimensions and the rest are data
    :param filepath:
    :return: 2D tensor
    """
    with open(filepath, "r") as f:
        lines = f.readlines()
    valid_lines = []
    for line in lines:
        if not line.startswith(COMMENT_CHAR):
            valid_lines.append(line.strip())

    first_line = valid_lines.pop(0).split(" ")
    m = int(first_line[0])
    n = int(first_line[1])

    matrix_content_strings = " ".join(valid_lines)
    matrix = np.fromstring(matrix_content_strings, dtype=np.float64, sep=" ").reshape(m, n)
    return torch.from_numpy(matrix).double()


def write_matrix_text_with_headers(matrix: torch.Tensor, file_path, *header_strings):
    """
    Writes a matrix to a txt file, with optional headers
    :param matrix: matrix to be written
    :param file_path: file path
    :param header_strings: optional
    """
    # CORRESPOND: write_matrix_text_valist() without conv
    check_tensor(matrix, dimensionality=2)
    header_string = " ".join(header_strings)
    matrix_strings = str(matrix.numpy()).replace("[", "").replace("]", "").split("\n")
    matrix_strings = [string.strip() for string in matrix_strings]
    file_content = []
    file_content.append(f"{COMMENT_CHAR} File created by TEM-simulator, version {VERSION}\n")
    file_content.append(f"{matrix.shape[0]} {matrix.shape[1]}\n")
    file_content.append(f"{COMMENT_CHAR} {header_string}\n")
    file_content.append("\n".join(matrix_strings))
    with open(file_path, "w") as f:
        f.writelines(file_content)


def write_tensor_to_binary(tensor: torch.Tensor,
                           filepath: str,
                           as_type: Optional[torch.dtype] = None,
                           append: Optional[bool] = False,
                           endian: Optional[ByteOrder] = None):
    """
    Dumps a tensor to a binary file

    :param tensor: tensor to be dumped
    :param filepath: file path
    :param as_type: Optional casting data type
    :param append: enable append mode
    :param endian:
    """
    if as_type is not None:
        tensor = tensor.to(as_type)
    mode = "ab" if append else "wb"
    with open(filepath, mode) as f:
        if endian is None:
            f.write(tensor.numpy().tobytes())
        else:
            np_array = tensor.numpy()
            dtype = np.dtype(np_array.dtype).newbyteorder(endian.to_np_representation())
            f.write(np_array.astype(dtype).tobytes())


def read_tensor_from_binary(filepath: str, dtype: torch.dtype,
                            shape: Optional[Iterable[int]] = None,
                            endian: Optional[ByteOrder] = None):
    """
    Read a tensor from a binary file

    :param filepath: file path
    :param dtype: data type of the reading tensor
    :param shape: shape of the reading tensor
    :param endian:
    :return: read tensor
    """
    np_dtype = torch.ones(1, dtype=dtype).numpy().dtype
    if endian is not None:
        np_dtype = np.dtype(np_dtype).newbyteorder(endian.to_np_representation())
    with open(filepath, "rb") as f:
        byte_array = f.read()
        t = torch.from_numpy(np.frombuffer(byte_array, dtype=np_dtype))

    if shape is None:
        return t
    else:
        return t.reshape(*shape)
