"""
Reference:
* https://github.com/scipy/scipy/blob/54c4b3eb0d9a6319df8c38cf8ae188f1f2e91e02/scipy/special/cephes/polevl.h
"""
import torch
from typing import List


@torch.jit.script
def polevl(x: torch.Tensor, coefficient: List[float], N: int):
    ans = torch.full_like(x, coefficient[0])
    for i in range(1, N + 1):
        ans = ans * x + coefficient[i]
    return ans


@torch.jit.script
def p1evl(x: torch.Tensor, coefficient: List[float], N: int):
    ans = x + coefficient[0]
    for i in range(1, N):
        ans = ans * x + coefficient[i]
    return ans


from .j0 import j0, J0
from .j1 import j1

__all__ = ["j0", "J0", "j1"]
