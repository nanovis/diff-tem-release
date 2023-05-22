import torch
from . import polevl, p1evl
from .j1 import j1

"""
reference: 
* https://github.com/scipy/scipy/blob/701ffcc8a6f04509d115aac5e5681c538b5265a2/scipy/special/cephes/j0.c
"""


@torch.jit.script
def _j0_le_5_ge_1e_neg5(x):
    """
    x in [1e-5, 5]
    """
    if x.shape == (0,):
        return x
    RQ = [
        4.99563147152651017219E2,
        1.73785401676374683123E5,
        4.84409658339962045305E7,
        1.11855537045356834862E10,
        2.11277520115489217587E12,
        3.10518229857422583814E14,
        3.18121955943204943306E16,
        1.71086294081043136091E18,
    ]
    RP = [
        -4.79443220978201773821E9,
        1.95617491946556577543E12,
        -2.49248344360967716204E14,
        9.70862251047306323952E15,
    ]
    DR1 = 5.78318596294678452118E0
    DR2 = 3.04712623436620863991E1

    z = x * x
    p = (z - DR1) * (z - DR2)
    p = p * polevl(z, RP, 3) / p1evl(z, RQ, 8)
    return p


@torch.jit.script
def _j0_lt_1e_neg5(x):
    """
    x in [0, 1e-5)
    """
    if x.shape == (0,):
        return x
    z = x * x
    return 1.0 - z / 4.0


@torch.jit.script
def _j0_gt_5(x):
    """
    x in (5, inf)
    """
    if x.shape == (0,):
        return x
    QQ = [
        6.43178256118178023184E1,
        8.56430025976980587198E2,
        3.88240183605401609683E3,
        7.24046774195652478189E3,
        5.93072701187316984827E3,
        2.06209331660327847417E3,
        2.42005740240291393179E2,
    ]
    QP = [
        -1.13663838898469149931E-2,
        -1.28252718670509318512E0,
        -1.95539544257735972385E1,
        -9.32060152123768231369E1,
        -1.77681167980488050595E2,
        -1.47077505154951170175E2,
        -5.14105326766599330220E1,
        -6.05014350600728481186E0,
    ]
    PQ = [
        9.24408810558863637013E-4,
        8.56288474354474431428E-2,
        1.25352743901058953537E0,
        5.47097740330417105182E0,
        8.76190883237069594232E0,
        5.30605288235394617618E0,
        1.00000000000000000218E0,
    ]
    PP = [
        7.96936729297347051624E-4,
        8.28352392107440799803E-2,
        1.23953371646414299388E0,
        5.44725003058768775090E0,
        8.74716500199817011941E0,
        5.30324038235394892183E0,
        9.99999999999999997821E-1,
    ]
    NPY_PI_4 = .78539816339744830962
    SQ2OPI = 7.9788456080286535587989E-1  # sqrt( 2/pi )
    w = 5.0 / x
    q = 25.0 / x ** 2
    p = polevl(q, PP, 6) / polevl(q, PQ, 6)
    q = polevl(q, QP, 7) / p1evl(q, QQ, 7)
    xn = x - NPY_PI_4
    p = p * torch.cos(xn) - w * q * torch.sin(xn)
    p = (p * SQ2OPI) / torch.sqrt(x)
    return p


@torch.jit.script
def j0(x: torch.Tensor):
    x = x.clone()
    x_lt_0 = x < 0.
    x[x_lt_0] = -x[x_lt_0]
    x_le_5 = x <= 5
    x_lt_1e_neg5 = x < 1e-5
    x_ge_1e_neg5 = ~x_lt_1e_neg5
    x_le_5_ge_1e_neg5 = x_le_5 & x_ge_1e_neg5
    x_gt_5 = ~x_le_5

    x[x_lt_1e_neg5] = _j0_lt_1e_neg5(x[x_lt_1e_neg5])
    x[x_le_5_ge_1e_neg5] = _j0_le_5_ge_1e_neg5(x[x_le_5_ge_1e_neg5])
    x[x_gt_5] = _j0_gt_5(x[x_gt_5])
    return x


class J0(torch.autograd.Function):
    """
    Optimized bessel j0 function with customized backward gradient calculation
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return j0(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        dj0_dx = -j1(input)
        return dj0_dx * grad_output
