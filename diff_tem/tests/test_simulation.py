from copy import deepcopy
from typing import List
from focal_frequency_loss import FocalFrequencyLoss as FFL
import torch
from torch import optim

from diff_tem.simulation_grad import Simulation
import taichi as ti
import logging

_logger = logging.getLogger(__name__)


def _write_log(string, *logging_strings):
    strings = [string, *logging_strings]
    if len(strings) > 1:
        logging_string = "\n    ".join(strings)
        _logger.debug(f"{_logger.name}:\n    {logging_string}", stacklevel=2)
    else:
        _logger.debug(f"{_logger.name}:  {strings[0]}", stacklevel=2)


def test_generate_micrographs():
    torch.autograd.set_detect_anomaly(True)
    torch.set_default_dtype(torch.float64)
    ti.init(arch=ti.cpu)
    file_path = "./input_TMV.txt"
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    simulation = Simulation.construct_from_file(file_path)
    simulation.finish_init_()
    forward_results = simulation.generate_micrographs()


def test_generate_micrographs_with_detector_grad_TMV():
    torch.set_default_dtype(torch.float64)
    ti.init(arch=ti.cpu)
    file_path = "./input_TMV_noisy.txt"
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    simulation = Simulation.construct_from_file(file_path)
    simulation.finish_init_()

    # copy detectors
    ref_detectors = simulation.detectors
    learned_detectors = deepcopy(ref_detectors)

    # set requires_grad
    learning_parameter_names = ["mtf_a", "mtf_b", "mtf_c", "mtf_alpha", "mtf_beta", "mtf_p", "mtf_q", "gain"]
    for d in learned_detectors:
        d.parameters_require_grad = True

    # randomize
    for d in learned_detectors:
        for parameter_name in learning_parameter_names:
            val = getattr(d, parameter_name).detach()
            setattr(d, parameter_name, val * torch.rand_like(val))

    learned_noisy_detector = learned_detectors[0]
    ref_noisy_detector = ref_detectors[0]

    for pn in learning_parameter_names:
        _write_log(f"Ref {pn} = {getattr(ref_noisy_detector, pn)}, ",
                   f"initial learning {pn} (from noisy ref) = {getattr(learned_noisy_detector, pn)}")

    lr = 0.01
    optimizer_noisy = optim.Adam([getattr(learned_noisy_detector, "_" + pn) for pn in learning_parameter_names], lr=lr)
    ffl = FFL(loss_weight=1.0, alpha=1.0)
    iterations = 20
    sample_num = 20
    ref_results, predetector_vvfs, snode_trees = simulation.generate_micrographs()
    for snode_tree in snode_trees:
        snode_tree.destroy()
    ref_noisy_predetector_vvfs = predetector_vvfs[0]

    for vvf in ref_noisy_predetector_vvfs:
        assert not torch.any(torch.isnan(vvf.values))

    # get references
    ref_noisy_results = []
    for s in range(sample_num):
        noisy_vvf_values = []
        for np_vvf in ref_noisy_predetector_vvfs:
            noisy_vvf = ref_noisy_detector.apply_quantization(np_vvf)
            noisy_vvf = ref_noisy_detector.apply_mtf(noisy_vvf)
            noisy_vvf_values.append(noisy_vvf.values)

        ref_noisy_results.append(torch.stack(noisy_vvf_values))
    ref_noisy_results = torch.stack(ref_noisy_results)  # (sample_num. tilt_num, *resolution)

    for i in range(iterations):
        optimizer_noisy.zero_grad()
        pred_noisy_results = []
        for s in range(sample_num):
            noisy_vvf_values = []
            for np_vvf in ref_noisy_predetector_vvfs:
                noisy_vvf = learned_noisy_detector.apply_quantization(np_vvf)
                noisy_vvf = learned_noisy_detector.apply_mtf(noisy_vvf)
                noisy_vvf_values.append(noisy_vvf.values)

            pred_noisy_results.append(torch.stack(noisy_vvf_values))
        pred_noisy_results = torch.stack(pred_noisy_results)  # (sample_num, tilt_num, *resolution)
        loss_noisy = ffl(pred_noisy_results.real, ref_noisy_results.real)
        loss_noisy.backward()
        optimizer_noisy.step()
        for pn in learning_parameter_names:
            _write_log(f"{i=} Ref {pn} = {getattr(ref_noisy_detector, pn)}, ",
                       f"learning {pn} (from noisy ref) = {getattr(learned_noisy_detector, pn)}")


def test_generate_micrographs_with_detector_grad():
    torch.autograd.set_detect_anomaly(True)
    torch.set_default_dtype(torch.float64)
    ti.init(arch=ti.cpu)
    file_path = "./Blank/input.txt"
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    simulation = Simulation.construct_from_file(file_path)
    simulation.finish_init_()

    # copy detectors
    ref_detectors = simulation.detectors
    learned_detectors = deepcopy(ref_detectors)

    # set requires_grad
    learning_parameter_names = ["mtf_a", "mtf_b", "mtf_c", "mtf_alpha", "mtf_beta", "mtf_p", "mtf_q", "gain"]
    for d in learned_detectors:
        d.parameters_require_grad = True

    # randomize
    for d in learned_detectors:
        for parameter_name in learning_parameter_names:
            val = getattr(d, parameter_name).detach()
            setattr(d, parameter_name, val * torch.rand_like(val))

    learned_noisy_detector, learned_noise_free_detector = learned_detectors
    ref_noisy_detector, ref_noise_free_detector = ref_detectors

    for pn in learning_parameter_names:
        _write_log(f"Ref {pn} = {getattr(ref_noisy_detector, pn)}, ",
                   f"initial learning {pn} (from noisy ref) = {getattr(learned_noisy_detector, pn)}, ",
                   f"initial learning {pn} (from clean ref) = {getattr(learned_noise_free_detector, pn)}")

    lr = 0.01
    optimizer_noisy = optim.Adam([getattr(learned_noisy_detector, "_" + pn) for pn in learning_parameter_names], lr=lr)
    optimizer_noise_free = optim.Adam(
        [getattr(learned_noise_free_detector, "_" + pn) for pn in learning_parameter_names],
        lr=lr)
    ffl = FFL(loss_weight=1.0, alpha=1.0)
    iterations = 20
    sample_num = 20
    ref_results, predetector_vvfs = simulation.generate_micrographs()
    ref_noisy_predetector_vvfs, ref_noise_free_predetector_vvfs = predetector_vvfs

    for vvf in ref_noisy_predetector_vvfs:
        assert not torch.any(torch.isnan(vvf.values))

    for vvf in ref_noise_free_predetector_vvfs:
        assert not torch.any(torch.isnan(vvf.values))

    # get references
    ref_noisy_results = []
    ref_noise_free_results = []
    for s in range(sample_num):
        noisy_vvf_values = []
        noise_free_vvf_values = []
        for np_vvf, nfp_vvf in zip(ref_noisy_predetector_vvfs, ref_noise_free_predetector_vvfs):
            noisy_vvf = ref_noisy_detector.apply_quantization(np_vvf)
            noisy_vvf = ref_noisy_detector.apply_mtf(noisy_vvf)
            noisy_vvf_values.append(noisy_vvf.values)

            noise_free_vvf = ref_noise_free_detector.apply_mtf(nfp_vvf)
            noise_free_vvf_values.append(noise_free_vvf.values)

        ref_noisy_results.append(torch.stack(noisy_vvf_values))
        ref_noise_free_results.append(torch.stack(noise_free_vvf_values))
    ref_noisy_results = torch.stack(ref_noisy_results)  # (sample_num. tilt_num, *resolution)
    ref_noise_free_results = torch.stack(ref_noise_free_results)  # (sample_num. tilt_num, *resolution)
    # FIXME: convergence speed is slow

    for i in range(iterations):
        optimizer_noisy.zero_grad()
        optimizer_noise_free.zero_grad()
        pred_noisy_results = []
        pred_noise_free_results = []
        for s in range(sample_num):
            noisy_vvf_values = []
            noise_free_vvf_values = []
            for np_vvf, nfp_vvf in zip(ref_noisy_predetector_vvfs, ref_noise_free_predetector_vvfs):
                noisy_vvf = learned_noisy_detector.apply_quantization(np_vvf)
                noisy_vvf = learned_noisy_detector.apply_mtf(noisy_vvf)
                noisy_vvf_values.append(noisy_vvf.values)

                noise_free_vvf = learned_noise_free_detector.apply_mtf(nfp_vvf)
                noise_free_vvf_values.append(noise_free_vvf.values)

            pred_noisy_results.append(torch.stack(noisy_vvf_values))
            pred_noise_free_results.append(torch.stack(noise_free_vvf_values))
        pred_noisy_results = torch.stack(pred_noisy_results)  # (sample_num, tilt_num, *resolution)
        pred_noise_free_results = torch.stack(pred_noise_free_results)  # (sample_num, tilt_num, *resolution)
        loss_noisy = ffl(pred_noisy_results.real, ref_noisy_results.real)
        loss_noise_free = ffl(pred_noise_free_results.real, ref_noise_free_results.real)
        loss_noisy.backward()
        loss_noise_free.backward()
        optimizer_noisy.step()
        optimizer_noise_free.step()
        for pn in learning_parameter_names:
            _write_log(f"{i=} Ref {pn} = {getattr(ref_noisy_detector, pn)}, ",
                       f"learning {pn} (from noisy ref) = {getattr(learned_noisy_detector, pn)}, ",
                       f"learning {pn} (from clean ref) = {getattr(learned_noise_free_detector, pn)}")


def test_generate_micrographs_grad_debug_ti():
    torch.autograd.set_detect_anomaly(True)
    torch.set_default_dtype(torch.float64)
    ti.init(arch=ti.vulkan)
    file_path = "./input_TMV.txt"
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    simulation = Simulation.construct_from_file(file_path)
    simulation.finish_init_()
    learning_rate = 0.05
    for particle_set in simulation.particle_sets:
        particle_set.coordinates.requires_grad_(True)
    optimizer = torch.optim.Adam([particle_set.coordinates for particle_set in simulation.particle_sets],
                                 lr=learning_rate)

    loss = []
    vis_results = []
    n_iterations = 2
    init_coordinates = []
    for i in range(n_iterations):
        forward_results = simulation.generate_micrographs()
        crop_forward_result = forward_results[0]

        vis_results.append(crop_forward_result)

        _write_log(f'forward_results Grad = {crop_forward_result.grad_fn}')
        _write_log(f'FORWARD RES = {crop_forward_result.shape}')

        mse = (crop_forward_result.real ** 2).mean()
        loss.append(mse)

        optimizer.zero_grad()

        mse.backward()

        for particle_set in simulation.particle_sets:
            _write_log(f'Grad of coordinates = {particle_set.coordinates.grad}')
            _write_log(f'Coordinates = {particle_set.coordinates}')

        optimizer.step()

        for particle_set in simulation.particle_sets:
            init_coordinates.append(particle_set.coordinates.detach().clone())
