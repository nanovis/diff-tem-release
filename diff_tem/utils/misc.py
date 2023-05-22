import taichi as ti
import torch


def ti_run_on_single_precision():
    torch_run_on_single_precision = torch.get_default_dtype() == torch.float32
    ti_on_vulkan = ti.cfg.arch == ti.vulkan
    return torch_run_on_single_precision or ti_on_vulkan
