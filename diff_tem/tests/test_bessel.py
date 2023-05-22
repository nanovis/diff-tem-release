import torch
from scipy.special import j0, j1
from diff_tem.utils import bessel


def test_bessel_j0():
    a = torch.randn(10, 10, dtype=torch.double)
    j0_scipy = j0(a.numpy())
    j0_scipy = torch.from_numpy(j0_scipy)
    j0_torch = bessel.j0(a)
    assert torch.allclose(j0_torch, j0_scipy)


def test_bessel_J0():
    a = torch.randn(10, 10, dtype=torch.double)
    j0_scipy = j0(a.numpy())
    j0_scipy = torch.from_numpy(j0_scipy)
    j0_torch = bessel.J0.apply(a)
    assert torch.allclose(j0_torch, j0_scipy)


def test_bessel_j1():
    a = torch.randn(100, 100, dtype=torch.double) * 100
    j1_scipy = j1(a.numpy())
    j1_scipy = torch.from_numpy(j1_scipy)
    j1_torch = bessel.j1(a)
    assert torch.allclose(j1_torch, j1_scipy)


def test_bessel_j0_grad():
    x = torch.randn(100, 100, dtype=torch.double, requires_grad=True)
    j0 = bessel.j0(x)
    loss = j0.sum()
    loss.backward()
    a_grad = x.grad
    a_ = x.detach()
    manual_grad = -bessel.j1(a_)
    assert torch.allclose(a_grad, manual_grad)


def test_bessel_grad():
    a = torch.randn(5, 5, dtype=torch.double, requires_grad=True)
    j0 = bessel.j0(a)
    loss = (j0 ** 2).sum()
    loss.backward()
    print(a.grad)


def test_BP():
    a = torch.randn(100, 100, requires_grad=True)
    j0 = bessel.j0(a)
    loss = (j0 ** 2).sum()
    loss.backward()
    a_grad0 = a.grad.clone()
    a.grad.zero_()
    j0_func = bessel.J0.apply
    j0 = j0_func(a)
    loss = (j0 ** 2).sum()
    loss.backward()
    a_grad1 = a.grad.clone()
    assert torch.allclose(a_grad1, a_grad0)
