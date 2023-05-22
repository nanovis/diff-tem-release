import os
from multiprocessing.pool import ThreadPool
import time
import torch
from ..constants import PI
import numpy as np

from diff_tem.pdb.pdb_protein import PDB_Protein
from diff_tem.particle import Particle
from diff_tem.eletronbeam import Electronbeam, wave_number, potential_conv_factor
from diff_tem.wavefunction import WaveFunction
from diff_tem.optics import Optics
import taichi as ti

from ..detector import Detector
from ..sample import Sample
from ..simulation import Simulation


def test_complex():
    a = torch.ones(1, dtype=torch.cfloat)
    a *= 2 + 0j
    print(a)


def test_project():
    start = time.time()
    thread_pool = ThreadPool(processes=os.cpu_count() * 2)
    end = time.time()
    print(end - start)
    a = [1, 2, 3]

    def map_func(x):
        return 2 * x

    b = thread_pool.map(map_func, a)
    print(b)


def test_fft():
    a = np.zeros(8, dtype=complex)
    a = a.reshape(4, 2)
    a[0, 0] = 0 + 1j
    a[0, 1] = 2 + 3j
    a[1, 0] = 4 + 5j
    a[1, 1] = 6 + 7j
    a[2, 0] = 0 + 0j
    a[2, 1] = 0 + 0j
    a[3, 0] = 0 + 0j
    a[3, 1] = 0 + 0j
    a = torch.from_numpy(a)
    print(a)
    ft = torch.fft.fft2(a, norm="backward")
    print(ft)
    ft /= (8)
    a = torch.fft.ifft2(ft, norm="forward")
    print(a)


def test_add():
    a = torch.tensor([0, 1, 2])  # 1 +  2)
    indices = torch.ones(3, dtype=torch.long)
    a[indices] += a
    print(a)


def map_function(arguments):
    particle, proj_matrix, position, _potential_conv_factor, _wave_num = arguments
    pf = particle.project(proj_matrix, position, _potential_conv_factor, _wave_num)
    return pf


def test_particle_project():
    sim = Simulation()

    ti.init(ti.cuda)
    pool = ThreadPool()
    ed = Electronbeam()
    ed.acc_voltage = 200.0
    ed.energy_spread = 1.3
    ed.total_dose = 6000.
    ed.gen_dose = True
    ed.finish_init_(1)
    acc_en = torch.tensor(ed.acceleration_energy)
    print(ed.acc_voltage)

    p = Particle()
    p.pdb_file_in = "./diff_tem/tests/test.pdb"
    p.pdb = PDB_Protein(p.pdb_file_in)
    p.voxel_size = 0.1
    p.use_imag_pot = True
    p.use_defocus_corr = False
    p.famp = 0.
    p._get_pot_from_pdb_(ed)
    p.lap_pot_im = torch.zeros((1, 1, 1))
    p.lap_pot_re = torch.zeros((1, 1, 1))
    # add substract boundary mean for particle
    # print(p.pot_im.shape)
    # for k in range(0, p.pot_im.shape[0]):
    #     for j in range(0, p.pot_im.shape[1]):
    #         for i in range(0, p.pot_im.shape[2]):
    #             print(f'NGAN: pot_re{k, j, i} = {p.pot_re[k, j, i]}')

    pm = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]])
    pos = torch.tensor([0., -50., 0.])

    o = Optics()
    o.magnification = 30000.
    o.cs = 2.
    o.cc = 2.
    o.aperture = 50.
    o.focal_length = 3.
    o.cond_ap_angle = 0.1
    o.defocus_nominal = 5.
    o.gen_defocus = True
    o.defocus_syst_err = 0.
    o.defocus_nonsyst_err = 0.
    o.finish_init_(1)

    sim.optics = o

    det_pix_x = 400
    det_pix_y = 400
    pixel_size = 16.
    gain = 10.
    dqe = 0.4
    mtf_a = 0.7
    mtf_b = 0.2
    mtf_c = 0.1
    mtf_alpha = 10.
    mtf_beta = 40.
    d = Detector()
    d.det_pix = torch.tensor([det_pix_x, det_pix_y]).type(torch.int32)
    d.pixel_size = pixel_size
    d.gain = gain
    d.dqe = dqe
    d.mtf_a = mtf_a
    d.mtf_b = mtf_b
    d.mtf_c = mtf_c
    d.mtf_alpha = mtf_alpha
    d.mtf_beta = mtf_beta
    d.use_quantization = True
    d.image_file_out = "test_no_noise.mrc"
    d.finish_init_(sim, None)

    det_pixel_size = 16000.
    det_area_all = torch.tensor([-3520000., 3520000., -3520000., 3520000.])
    wf = WaveFunction(ed, o, det_pixel_size, det_area_all)
    wf.set_incoming_()

    slice_thickness = wf.pixel_size * wf.pixel_size / (4.0 * PI) * wave_number(torch.tensor(ed.acceleration_energy))
    print(slice_thickness)
    tilt = 0
    np = 1
    if p.hits_wavefunction(wf, pm, pos):
        k = torch.ceil(pos[2] / slice_thickness).type(torch.long)
    kmin = kmax = 0
    arguments = []

    # Add many phase with zero values, add for each slice, then propagate.
    for k in range(kmin, kmax + 1):
        pos[2] -= k * slice_thickness
        arguments.append([p, pm, pos.clone(), potential_conv_factor(acc_en), wave_number(acc_en)])
    start = time.time()
    pfs = pool.map(map_function, arguments)
    end = time.time()
    print(f'Time particle: {end - start}')

    wf.phase.values = torch.zeros_like(wf.phase.values, dtype=torch.cdouble)
    for pf in pfs:
        wf.phase.add_(pf)
    wf._adj_phase_()
    # Add defocus later
    # de_focus = optics_get_defocus(wf->opt, tilt)
    wf._propagate_inelastic_scattering_(tilt, k * slice_thickness)
    wf._propagate_elastic_scattering_through_vacuum_(slice_thickness)
    wf._propagate_elastic_scattering_optical_(tilt, kmax * slice_thickness)

    s = Sample(1000., 50., 100.)
    pm = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    pos = torch.tensor([0., 0., 0.])
    wf._background_project_(s, pm, pos)
    wf._apply_background_blur_(tilt)

    start = time.time()
    # FIXME: update test code related to Detector because of API changes
    d.add_intensity_from_wave_function_to_vvf(wf, tilt)
    end = time.time()
    print(f'Time add intensity: {end - start}')
    # d.apply_quantization()
    d.apply_mtf()

    # f = open("d_count_final.log", "w")
    # for i in range(0, d.count.values.shape[0]):
    #     for j in range(0, d.count.values.shape[1]):
    #         f.write(f'{i, j} = {d.count.values[i, j]}\n')
    # f.close()

    d.write_image()
