import mrcfile
import torch
from diff_tem.utils.mrc import write_tensor_to_mrc


def test_mrc_write():
    a = torch.rand(3, 2, 3)
    file_path = ".pytest_cache/test.mrc"
    write_tensor_to_mrc(a, file_path, 1.)
    with mrcfile.open(file_path) as mrc:
        data = mrc.data
    b = torch.from_numpy(data)
    assert torch.allclose(a, b)
