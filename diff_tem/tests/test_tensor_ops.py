from diff_tem.utils.tensor_ops import *
import torch


def test_discrete_laplace():
    laplace = DiscreteLaplace()
    a = torch.ones(5, 5, 5)
    b = laplace(a)
    print(b.shape)
    for k in range(5):
        for j in range(5):
            for i in range(5):
                print(f"{k} {j} {i} : {b[k, j, i]}")


def test_tensor_indexing():
    indexee = torch.randn(5, 5, 3, 4)
    x_indices = torch.arange(5).reshape(5, 1).repeat(1, 5).int()
    y_indices = torch.arange(5).reshape(1, 5).repeat(5, 1).int()
    indexer = torch.stack([x_indices, y_indices], dim=-1)
    index_result, _ = tensor_indexing(indexee, indexer, 2)
    assert torch.allclose(index_result, indexee)


def test_mean():
    t = torch.randn(2, 3, 4)
    mean = boundary_mean(t)
