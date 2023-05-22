import logging
import math
from copy import copy
from typing import List

import torch
from taichi._snode.snode_tree import SNodeTree

from . import UNINITIALIZED
from .constants import PI, ONE_NANOMETER
from .constants.derived_units import ONE_MICROMETER
from .constants.physical_constants import ICE_DENS, ELEC_REST_ENERGY
from .eletronbeam import Electronbeam, total_cross_section, differential_cross_section, wave_number, \
    potential_conv_factor
from .optics import Optics
from .sample import sample_support_abs_pot, sample_ice_abs_pot, Sample
from .utils import bessel
from .utils.helper_functions import check_tensor, eprint
from .vector_valued_function import VectorValuedFunction, VirtualField
import taichi as ti

_logger = logging.getLogger(__name__)


def _write_log(string, *logging_strings):
    strings = [string, *logging_strings]
    if len(strings) > 1:
        logging_string = "\n    ".join(strings)
        _logger.debug(f"{_logger.name}:\n    {logging_string}", stacklevel=2)
    else:
        _logger.debug(f"{_logger.name}:  {strings[0]}", stacklevel=2)


class WaveFunction:

    def __init__(self,
                 electronbeam: Electronbeam,
                 optics: Optics,
                 det_pixel_size: float,
                 det_range: torch.Tensor):
        check_tensor(det_range, shape=(4,), dtype=torch.double)
        assert electronbeam.initialized and optics.initialized, \
            "The electron beam and optics passed to wave function should be initialized first"
        self.electronbeam: Electronbeam = electronbeam
        self.optics: Optics = optics
        self._det_pixel_size: float = det_pixel_size
        self._det_range: torch.Tensor = det_range
        magnification = optics.magnification
        pixels_x = math.ceil((det_range[1] - det_range[0]) / det_pixel_size)
        pixels_y = math.ceil((det_range[3] - det_range[2]) / det_pixel_size)
        self.pixel_size: float = det_pixel_size / magnification
        phase_basis_x = torch.tensor([self.pixel_size, 0.], dtype=torch.double)
        phase_basis_y = torch.tensor([0., self.pixel_size], dtype=torch.double)
        phase_offset = torch.tensor([det_range[0] + det_range[1], det_range[2] + det_range[3]]) * 0.5 / magnification
        phase_value = torch.zeros(pixels_y, pixels_x, dtype=torch.cdouble)
        intens_basis_x = torch.tensor([det_pixel_size, 0.])
        intens_basis_y = torch.tensor([0., det_pixel_size])
        intens_offset = torch.tensor([det_range[0] + det_range[1], det_range[2] + det_range[3]]) * 0.5
        intens_values = torch.zeros(pixels_y, pixels_x, dtype=torch.cdouble)
        self.phase: VectorValuedFunction = VectorValuedFunction(phase_value, phase_basis_x, phase_basis_y, phase_offset)
        self.wf: torch.Tensor = torch.full((pixels_y, pixels_x), 1. + 0j, dtype=torch.cdouble)
        self.inel: torch.Tensor = torch.zeros(pixels_y, pixels_x)
        self.pathlength_ice: torch.Tensor = torch.zeros(pixels_y, pixels_x)
        self.pathlength_supp: torch.Tensor = torch.zeros(pixels_y, pixels_x)
        self.intens: VectorValuedFunction = VectorValuedFunction(intens_values,
                                                                 intens_basis_x, intens_basis_y,
                                                                 intens_offset)
        self.inel_ctf: torch.Tensor = UNINITIALIZED
        self.inel_ctf_dx: float = UNINITIALIZED
        self._set_inel_ctf_()

    def spawn(self):
        wf = copy(self)
        wf.phase = self.phase.spawn()
        wf.wf = torch.zeros_like(self.wf)
        wf.inel = torch.zeros_like(self.inel)
        wf.pathlength_ice = torch.zeros_like(self.pathlength_ice)
        wf.pathlength_supp = torch.zeros_like(self.pathlength_supp)
        wf.intens = self.intens.spawn()
        wf.inel_ctf = self.inel_ctf.clone()
        wf.inel_ctf_dx = self.inel_ctf_dx
        return wf

    def _set_inel_ctf_(self):
        # CORRESPOND: wavefunction_set_inel_ctf
        carbon_atomic_number = torch.tensor(6)
        oxygen_atomic_number = torch.tensor(8)
        hydrogen_atomic_number = torch.tensor(1)
        _, max_indices = torch.max(torch.abs(self.optics.defocus), 0, True)
        def_max = torch.abs(self.optics.get_defocus_nanometer(max_indices[0]))
        t_max = 0.5 * self.optics.aperture_nanometer / self.optics.focal_length_nanometer
        self.inel_ctf_dx = 1. / t_max
        x_max = PI * (def_max + ONE_MICROMETER) / self.pixel_size
        dt = 1. / x_max
        nt = int(math.floor(t_max / dt))
        self.inel_ctf = torch.zeros(2, int(math.ceil(x_max / self.inel_ctf_dx))).double()
        acc_en = self.electronbeam.acceleration_energy
        f0 = total_cross_section(acc_en, carbon_atomic_number)
        i = torch.arange(1, nt + 1).view(-1, 1)  # (nt, 1)
        t = i * dt  # (nt, 1)
        f = 2 * PI * t * dt / f0 * differential_cross_section(acc_en, carbon_atomic_number, t)  # (nt, 1)
        x = (torch.arange(self.inel_ctf.shape[1]) * self.inel_ctf_dx).view(1, -1)  # (1, self.inel_ctf.shape[1])
        i0_xt = bessel.J0.apply(t * x)
        self.inel_ctf[0, :] += torch.sum(f * i0_xt, dim=0)  # (nt, self.inel_ctf.shape[1]) -> (self.inel_ctf.shape[1], )
        f = 2 * PI * t * dt \
            * (differential_cross_section(acc_en, oxygen_atomic_number, t)
               + 2 * differential_cross_section(acc_en, hydrogen_atomic_number, t))  # (nt, 1)
        self.inel_ctf[1, :] += torch.sum(f * i0_xt, dim=0)  # (nt, self.inel_ctf.shape[1]) -> (self.inel_ctf.shape[1], )
        f0 = total_cross_section(acc_en, oxygen_atomic_number) + 2 * total_cross_section(acc_en, hydrogen_atomic_number)
        self.inel_ctf[1, :] = ICE_DENS * (f0 - self.inel_ctf[1, :])

    def to(self, device):
        """
        Out-of-place move to a device, creating new copy of self whose data is on the device

        :param device: device that holds data
        :return: a new copy of self
        """
        wf = WaveFunction(self.electronbeam.to(device), self.optics.to(device), self._det_pixel_size, self._det_range)
        wf.phase.to_(device)
        wf.wf = wf.wf.to(device)
        wf.inel = wf.inel.to(device)
        wf.pathlength_ice = wf.pathlength_ice.to(device)
        wf.intens.to_(device)
        wf.inel_ctf = wf.inel_ctf.to(device)
        return wf

    def to_(self, device):
        """
        In-place move to a device

        :param device: device that holds data
        """
        self.electronbeam.to_(device)
        self.optics.to_(device)
        self._det_range = self._det_range.to(device)
        self.phase.to_(device)
        self.wf = self.wf.to(device)
        self.inel = self.inel.to(device)
        self.pathlength_ice = self.pathlength_ice.to(device)
        self.pathlength_supp = self.pathlength_supp.to(device)
        self.intens.to_(device)
        self.inel_ctf = self.inel_ctf.to(device)

    def set_incoming_(self):
        """
        Re-initialize tensors in WaveFunction
        """
        # CORRESPOND: wavefunction_set_incoming
        self.wf[:, :] = 1. + 0.j
        self.intens.values = torch.zeros_like(self.intens.values)

    def _adj_phase_(self):
        # CORRESPOND: wavefunction_adj_phase
        ph = self.phase.values
        w = torch.view_as_real(self.wf)
        x = torch.exp(-ph.imag)
        inel = (1.0 - x * x) * (w ** 2).sum(dim=-1)
        self.inel = inel
        y = x * torch.sin(ph.real)
        x = x * torch.cos(ph.real)
        w = torch.view_as_complex(w)
        self.wf = w * torch.complex(x, y)

    def _propagate_inelastic_scattering_(self, tilt, z):
        # CORRESPOND: wavefunction_prop_inel
        de_focus = self.optics.get_defocus_nanometer(tilt)
        df = torch.abs(z + de_focus)
        dxi = 2 * PI * df / (self.pixel_size * self.inel.shape[1])
        dxj = 2 * PI * df / (self.pixel_size * self.inel.shape[0])
        c = 1.0 / (self.inel.shape[0] * self.inel.shape[1])

        id = torch.fft.rfft2(self.inel,
                             norm='backward')  # id.shape[0] = self.inel.shape[0], id.shape[1] = self.inel.shape[1] / 2 + 1
        j = torch.arange(id.shape[0])
        j = torch.minimum(j, self.inel.shape[0] - j)  # (id.shape[0], )
        xj = dxj * j
        i = torch.arange(id.shape[1])
        i = torch.minimum(i, self.inel.shape[1] - i)  # (id.shape[1], )
        xi = dxi * i

        xj = xj.view(-1, 1)  # (id.shape[0], 1)
        xi = xi.view(1, -1)  # (1, id.shape[1])

        flat_inel = self.inel_ctf.view(-1)
        s = torch.sqrt(xi * xi + xj * xj) / self.inel_ctf_dx  # (id.shape[0], id.shape[1])
        k = torch.floor(s).long()  # (id.shape[0], id.shape[1])
        k_mask = k < (self.inel_ctf.shape[1] - 1)  # (id.shape[0], id.shape[1])
        s_if = s[k_mask] - k[k_mask]  # (id.shape[0], id.shape[1])
        id_zero = torch.zeros_like(id)
        id_zero[k_mask] = c * ((1 - s_if) * flat_inel[k[k_mask]] + s_if * flat_inel[k[k_mask] + 1]) * id[k_mask]
        id_zero[~k_mask] = c * (flat_inel[self.inel_ctf.shape[1] - 1]) * id[~k_mask]
        id = id_zero

        self.inel = torch.fft.irfft2(id, norm='forward')
        self.intens.values = torch.complex(self.intens.values.real + self.inel, self.intens.values.imag)

    def _propagate_elastic_scattering_through_vacuum_(self, dist):
        m = self.wf.shape[1]  # px
        n = self.wf.shape[0]  # py
        dxi = 2. * PI / (m * self.pixel_size)
        dxj = 2. * PI / (n * self.pixel_size)
        wavenum = wave_number(self.electronbeam.acceleration_energy)
        a = -0.5 * dist / wavenum
        env = 1. / (n * m)

        ft = torch.fft.fft2(self.wf, norm="backward")
        j = torch.arange(n)
        j = torch.minimum(j, n - j)  # (n, ): py
        xj = dxj * j
        i = torch.arange(m)
        i = torch.minimum(i, m - i)  # (m, ): px
        xi = dxi * i

        xj = xj.view(-1, 1)  # (n, 1)
        xi = xi.view(1, -1)  # (1, m)

        x2 = xi * xi + xj * xj  # (n, m)
        chi = x2 * a
        ctfr = env * torch.cos(chi)  # (n, m) : (py, px)
        ctfi = env * torch.sin(chi)  # (n, m) : (py, px)
        ctf = torch.complex(ctfr, ctfi)  # (n, m) : (py, px)

        ft = ft * ctf
        self.wf = torch.fft.ifft2(ft, norm="forward")

    def _propagate_elastic_scattering_optical_(self, tilt, z):
        assert tilt >= 0 and tilt < self.optics.defocus.shape[0]
        m = self.wf.shape[1]  # px
        n = self.wf.shape[0]  # py
        dxi = 2. * PI / (m * self.pixel_size)
        dxj = 2. * PI / (n * self.pixel_size)
        acc_en = self.electronbeam.acceleration_energy
        k = wave_number(acc_en)
        acc_en_spr = self.electronbeam.energy_spread_with_unit
        de_focus = self.optics.get_defocus_nanometer(tilt)
        df = torch.abs(z + de_focus)
        cs = self.optics.cs_nanometer
        epsilon = 0.5 / ELEC_REST_ENERGY
        ccp = ((1. + 2. * epsilon * acc_en) / (acc_en * (1 + epsilon * acc_en))) * self.optics.cc_nanometer
        ap_ang = self.optics.cond_ap_angle_radian

        a = 0.5 * df / k
        b = -0.25 * cs / (k ** 3)
        c = -0.25 * df * df * ap_ang * ap_ang
        d = (0.5 * df * cs * ap_ang * ap_ang - 0.0625 * acc_en_spr * acc_en_spr * ccp * ccp) / (k * k)
        e = -0.25 * cs * cs * ap_ang * ap_ang / (k ** 4)

        B = torch.minimum(torch.tensor(PI / self.pixel_size),
                          0.5 * k * self.optics.aperture_nanometer / self.optics.focal_length_nanometer)

        ft = torch.fft.fft2(self.wf, norm="backward")
        j = torch.arange(n)
        j = torch.minimum(j, n - j)  # (n, ) : py
        xj = dxj * j
        i = torch.arange(m)
        i = torch.minimum(i, m - i)  # (m, ) : px
        xi = dxi * i

        xj = xj.view(-1, 1)  # (n, 1)
        xi = xi.view(1, -1)  # (1, m)

        x2 = xi * xi + xj * xj  # (n, m) : (py, px)
        x = torch.sqrt(x2)
        x_mask = x < B
        ctf = torch.zeros_like(ft, dtype=torch.cdouble)
        masked_x2 = x2[x_mask]

        masked_x4 = masked_x2 * masked_x2
        masked_x6 = masked_x2 * masked_x4
        env = (1. / (n * m)) * torch.exp(c * masked_x2 + d * masked_x4 + e * masked_x6)
        chi = a * masked_x2 + b * masked_x4

        real = env * torch.cos(chi)
        imag = env * torch.sin(chi)
        ctf[x_mask] = torch.complex(real.double(), imag.double())

        ft = ft * ctf
        self.wf = torch.fft.ifft2(ft, norm="forward")
        self.intens.values = torch.complex(self.intens.values.real + self.wf.real ** 2 + self.wf.imag ** 2,
                                           self.intens.values.imag)

    def _apply_background_blur_(self, tilt):
        acc_en = self.electronbeam.acceleration_energy
        c = sample_support_abs_pot(acc_en) / sample_ice_abs_pot(acc_en)

        path_length_ice = self.pathlength_ice.clone()
        path_length_supp = self.pathlength_supp.clone()

        # Find maximum and minimum path_length
        path_length_ice = c * path_length_supp + path_length_ice
        path_length_min = torch.min(path_length_ice)
        mask = path_length_supp < ONE_NANOMETER
        if torch.any(mask):
            path_length_max = torch.max(torch.maximum(path_length_supp[mask], path_length_ice[mask]))
            path_length_max = torch.maximum(path_length_max, path_length_min)
        else:
            path_length_max = torch.maximum(torch.zeros(1), path_length_min)

        if path_length_max < ONE_NANOMETER:
            return

        # Determine number of thickness values to use
        m = 20  # maximum number
        c = 10.

        diff_path_length = path_length_max - path_length_min
        if diff_path_length < ONE_NANOMETER:
            m = 0  # all path_lengths can be considered equal
        elif c * diff_path_length < m * path_length_min:
            m = torch.ceil(c * diff_path_length / path_length_min).long()

        if m > 0:
            pl_step = diff_path_length / m
        else:
            pl_step = 1.  # just to avoid division by 0

        # Compute blurring for finite set of thicknesses and interpolate
        df = self.optics.get_defocus_nanometer(tilt)
        c = 1. / (self.inel.shape[0] * self.inel.shape[1])
        dxi = 2 * PI * df / (self.pixel_size * self.inel.shape[1])
        dxj = 2 * PI * df / (self.pixel_size * self.inel.shape[0])

        ctfv = self.inel_ctf.view(-1)[self.inel_ctf.shape[1]:]
        path_length_ice = self.pathlength_ice.clone()  # (path_ice.shape[0], path_ice.shape[1])
        self.inel = self.intens.values.real.clone()
        self.intens.values = torch.zeros_like(self.intens.values)

        inel_ft = torch.fft.rfft2(self.inel, norm='backward')  # (self.inel.shape[0], self.inel.shape[1]/2 + 1)
        j = torch.arange(inel_ft.shape[0])
        j = torch.minimum(j, self.inel.shape[0] - j)
        xj = dxj * j

        i = torch.arange(inel_ft.shape[1])
        i = torch.minimum(i, self.inel.shape[1] - i)
        xi = dxi * i

        xj = xj.view(-1, 1)
        xi = xi.view(1, -1)

        s = torch.sqrt(xi * xi + xj * xj) / self.inel_ctf_dx
        k = torch.floor(s).long()
        k_mask = k < (self.inel_ctf.shape[1] - 1)
        s_if = s[k_mask] - k[k_mask]

        ctf = torch.zeros_like(inel_ft, dtype=torch.double)  # (inel_ft.shape[0], inel_ft_shape[1])
        ctf[k_mask] = (1 - s_if) * ctfv[k[k_mask]] + s_if * ctfv[k[k_mask] + 1]
        ctf[~k_mask] = ctfv[self.inel_ctf.shape[1] - 1]

        path_length_mins = torch.arange(0, m + 1) * pl_step + path_length_min  # (m+1, )
        path_length_mins = path_length_mins.view(-1, 1, 1)
        ctf = ctf.unsqueeze(0)  # (1, inel_ft.shape[0], inel_ft_shape[1])
        inel_ft = inel_ft.unsqueeze(0)  # (1, inel_ft.shape[0], inel_ft_shape[1])
        inel_fts = c * inel_ft * torch.exp(-ctf * path_length_mins)  # (m+1, inel_ft.shape[0], inel_ft.shape[1])
        inels = torch.fft.irfft2(inel_fts, norm="forward")  # (m+1, inel.shape[0], inel.shape[1])
        path_length_ice = path_length_ice.unsqueeze(0)  # (1, path_ice.shape[0], path_ice.shape[1])
        s = path_length_mins - path_length_ice  # (m+1, path_ice.shape[0], path_ice.shape[1])

        # for the first m iterations
        s_first_m = torch.abs(s[:m])
        s_first_m_mask = s_first_m < pl_step
        inel_first_m = inels[:m]
        inel_first_m_zero = torch.zeros_like(inel_first_m)
        inel_first_m_zero[s_first_m_mask] = (1. - s_first_m[s_first_m_mask] / pl_step) * inel_first_m[s_first_m_mask]
        inel_first_m = inel_first_m_zero
        inels[:m] = inel_first_m

        # for the last iteration
        s_last = s[-1]
        s_mask_1 = (s_last >= 0) & (s_last < pl_step)
        s_mask_3 = s_last < 0.
        inel_last_value = inels[-1]
        inel_last_value_zero = torch.zeros_like(inel_last_value)
        inel_last_value_zero[s_mask_1] = (1. - s_last[s_mask_1] / pl_step) * inel_last_value[s_mask_1]
        inel_last_value_zero[s_mask_3] = torch.exp(s_last[s_mask_3] * ctfv[0]) * inel_last_value[s_mask_3]
        inel_last_value = inel_last_value_zero
        inels[-1] = inel_last_value
        self.intens.values = torch.complex(self.intens.values.real + inels.sum(0), self.intens.values.imag)

    def propagate_(self, simulation, slice_thickness, tilt) -> List[SNodeTree]:
        # CORRESPOND: wavefunction_propagate
        sample = simulation.sample
        geometry = simulation.geometry
        geometry.finish_init_()
        for particle_set in simulation.particle_sets:
            particle_set.finish_init_(simulation)
        assert geometry.data.shape[0] > tilt >= 0, "Error in wavefunction propagate_: tilt number out of range.\n"
        _write_log(f"Computing projection number {tilt + 1}")
        self.set_incoming_()
        num_particles = 0
        for particle_set in simulation.particle_sets:
            num_particles += particle_set.coordinates.shape[0]

        slice_idx_min = slice_idx_max = 0
        count = 0
        first_hit = True

        potential_factor = potential_conv_factor(self.electronbeam.acceleration_energy)
        wave_num = wave_number(self.electronbeam.acceleration_energy)
        snode_trees = []

        if num_particles > 0:
            slice_indices_of_all_particles = torch.zeros(num_particles, dtype=torch.long)
            hit = torch.zeros(num_particles, dtype=torch.bool)
            particle_index = 0

            for particle_set in simulation.particle_sets:
                particle = simulation.get_particle(particle_set.particle_type)
                assert particle is not None, f"Particle {particle_set.particle_type} not found"
                particle.finish_init_(simulation)
                if particle is not None:
                    for i in range(0, particle_set.coordinates.shape[0]):
                        proj_matrix, pos = geometry.get_particle_geom(particle_set, i, tilt)
                        if particle.hits_wavefunction(self, proj_matrix, pos):
                            slice_idx_of_particle = torch.round(pos[2] / slice_thickness).int()
                            slice_indices_of_all_particles[particle_index] = slice_idx_of_particle
                            hit[particle_index] = True
                            if first_hit or slice_idx_of_particle > slice_idx_max:
                                slice_idx_max = slice_idx_of_particle
                            if first_hit or slice_idx_of_particle < slice_idx_min:
                                slice_idx_min = slice_idx_of_particle
                            first_hit = False
                        else:
                            hit[particle_index] = False
                        particle_index += 1

            # Project particles slice by slice
            for slice_idx_of_particle in range(slice_idx_min, slice_idx_max + 1):
                # preparations
                particle_index = 0
                pfs = []
                for particle_set in simulation.particle_sets:
                    particle = simulation.get_particle(particle_set.particle_type)
                    assert particle is not None, f"Particle {particle_set.particle_type} not found"
                    particle.finish_init_(simulation)
                    if particle is not None:
                        for i in range(0, particle_set.coordinates.shape[0]):
                            if hit[particle_index] \
                                    and slice_idx_of_particle == slice_indices_of_all_particles[particle_index]:
                                proj_matrix, pos = geometry.get_particle_geom(particle_set, i, tilt)
                                _write_log(f'Particle {i}-th is projected')
                                _write_log(f'proj_matrix Grad = {proj_matrix.grad_fn}')
                                _write_log(f'pos Grad = {pos.grad_fn}')
                                pos[2] = pos[2] - slice_idx_of_particle * slice_thickness
                                pf = particle.project(proj_matrix, pos, potential_factor, wave_num)
                                pfs.append(pf)
                                count += 1
                            particle_index += 1

                standins = []
                self.phase.values = torch.zeros_like(self.phase.values, dtype=torch.cdouble)
                tasks = []
                for pf in pfs:
                    standin = self.phase.spawn()
                    tasks.append(standin.add_(pf))  # async VVF add_() task
                    standins.append(standin)

                virtual_fields: List[List[VirtualField]] = [task.send(None) for task in tasks]
                field_builder = ti.FieldsBuilder()
                concrete_fields = []
                for virtual_field_list in virtual_fields:
                    if virtual_field_list is None:
                        concrete_fields.append(None)
                    else:
                        concrete_field_list = list(map(lambda x: x.concretize(field_builder), virtual_field_list))
                        concrete_fields.append(concrete_field_list)
                snode_handle_for_1st_interpolation = field_builder.finalize()
                # drive to finish first interpolation
                virtual_fields = [task.send(concrete_field_list)
                                  for task, concrete_field_list in zip(tasks, concrete_fields)]
                field_builder = ti.FieldsBuilder()
                concrete_fields = []
                for virtual_field_list in virtual_fields:
                    if virtual_field_list is None:
                        concrete_fields.append(None)
                    else:
                        concrete_field_list = list(map(lambda x: x.concretize(field_builder), virtual_field_list))
                        concrete_fields.append(concrete_field_list)
                snode_handle_for_2nd_interpolation = field_builder.finalize()
                # drive to finish second interpolation
                for task, concrete_field_list in zip(tasks, concrete_fields):
                    try:
                        task.send(concrete_field_list)
                    except StopIteration:
                        pass

                snode_trees.append(snode_handle_for_1st_interpolation)
                snode_trees.append(snode_handle_for_2nd_interpolation)
                self.phase.merge_standins_(standins)
                _write_log(f'phase Grad = {self.phase.values.grad_fn}')
                self._adj_phase_()
                _write_log(f'wf Grad _adj_phase_ = {self.wf.grad_fn}')
                _write_log(f'inel Grad _adj_phase_ = {self.inel.grad_fn}')
                self._propagate_inelastic_scattering_(tilt, slice_idx_of_particle * slice_thickness)
                _write_log(f'intens Grad _propagate_inelastic_scattering_ = {self.intens.values.grad_fn}')
                _write_log(f'inel Grad _propagate_inelastic_scattering_ = {self.inel.grad_fn}')
                if slice_idx_of_particle < slice_idx_max:
                    self._propagate_elastic_scattering_through_vacuum_(slice_thickness)

        _write_log(f"Projected {count} particles")
        self._propagate_elastic_scattering_optical_(tilt, slice_idx_max * slice_thickness)
        _write_log(f'intens Grad _propagate_elastic_scattering_optical_ = {self.intens.values.grad_fn}')
        _write_log(f'wf Grad _propagate_elastic_scattering_optical_ = {self.wf.grad_fn}')
        proj_matrix, pos = geometry.get_sample_geom(sample, tilt)
        self._background_project_(sample, proj_matrix, pos)
        self._apply_background_blur_(tilt)
        _write_log(f'intens Grad _apply_background_blur_ = {self.intens.values.grad_fn}')
        _write_log(f'inel Grad _apply_background_blur_ = {self.inel.grad_fn}')

        return snode_trees

    @staticmethod
    def _find_root(a, b, c):
        """
        Finds the root closest to 0 of the equation a*x^2 + 2*b*x + c = 0.
        Assumes that the equation has real roots.
        The coefficient a can be 0, but b must be nonzero.
        """
        b2 = b ** 2
        ac_lt_1e2_b2 = torch.abs(a * c) < (1e-2 * b2)
        result = torch.zeros_like(ac_lt_1e2_b2, dtype=torch.double)
        masked_b = b[ac_lt_1e2_b2]
        masked_c = c[ac_lt_1e2_b2]
        masked_b2 = b2[ac_lt_1e2_b2]
        result[ac_lt_1e2_b2] = -0.5 * (1 + 0.25 * a * masked_c / masked_b2) * masked_c / masked_b
        ac_ge_1e2_b2 = ~ac_lt_1e2_b2
        result[ac_ge_1e2_b2] = (b[ac_ge_1e2_b2] / a) * (torch.sqrt(1 - a * c[ac_ge_1e2_b2] / b2[ac_ge_1e2_b2]) - 1)
        return result

    def _background_project_(self, sample: Sample, project_matrix: torch.Tensor, pos: torch.Tensor):
        e1 = 1e-6
        e3 = 1e-3
        check_tensor(project_matrix, shape=(3, 3))
        check_tensor(pos, shape=(3,))
        if torch.abs(project_matrix[2, 2]) < e3:
            torch.fill_(self.pathlength_ice, 0.0)
            torch.fill_(self.pathlength_supp, 0.0)
            eprint("Ignoring background in projection parallel to sample plane.\n")
            return

        k = project_matrix[2, :2] / project_matrix[2, 2]  # (2, )
        k2 = (k ** 2).sum()
        sk2 = torch.sqrt(1 + k2)
        diameter = sample.diameter
        r2 = 0.25 * diameter * diameter
        zb = 0.5 * sample.thickness_edge
        zc = 0.5 * sample.thickness_center
        a = 2. * (zb - zc) / r2
        ic = 0.5 * (self.phase.values.shape[1] - 1)
        jc = 0.5 * (self.phase.values.shape[0] - 1)

        i = torch.arange(self.phase.values.shape[1]).unsqueeze(-1)  # (wf.phase.values.shape[1], 1)
        j = torch.arange(self.phase.values.shape[0]).unsqueeze(-1)  # (wf.phase.values.shape[0], 1)

        x1 = (i - ic) * self.phase.x_basis.view(1, 2)  # (wf.phase.values.shape[1], 2)
        x2 = (j - jc) * self.phase.y_basis.view(1, 2)  # (wf.phase.values.shape[0], 2)
        x1 = x1.view(1, -1, 2)  # (1, wf.phase.values.shape[1], 2)
        x2 = x2.view(-1, 1, 2)  # (wf.phase.values.shape[0], 1, 2)
        # x: (wf.phase.values.shape[0], wf.phase.values.shape[1], 2)
        x = x1 + x2 + self.phase.offset.view(1, 1, 2) - pos[:2].view(1, 1, 2)
        pm = project_matrix[:, :2].view(1, 1, 3, 2)
        x = x.view(x.shape[0], x.shape[1], 2, 1)
        m = pm @ x  # (wf.phase.values.shape[0], wf.phase.values.shape[1], 3, 1)
        m = m.squeeze(-1)  # (wf.phase.values.shape[0], wf.phase.values.shape[1], 3)
        m2 = m[:, :, 2]
        m0 = m[:, :, 0] - k[0] * m2
        m1 = m[:, :, 1] - k[1] * m2
        m = torch.stack([m0, m1, m2], dim=-1)
        s = k[0] * m[:, :, 0] + k[1] * m[:, :, 1]  # (wf.phase.values.shape[0], wf.phase.values.shape[1])
        t = (m[:, :, :2] ** 2).sum(dim=-1)  # (wf.phase.values.shape[0], wf.phase.values.shape[1])
        path_length_ice = self.pathlength_ice
        path_length_support = self.pathlength_supp
        if k2 < e1:
            t_lt_r2 = t < r2
            path_length_ice[t_lt_r2] = 2. * zc + a * t[t_lt_r2]
            path_length_support[t_lt_r2] = 0.
            t_ge_r2 = ~t_lt_r2
            path_length_ice[t_ge_r2] = 0.
            path_length_support[t_ge_r2] = 2. * zb
        else:
            b = s / k2  # (wf.phase.values.shape[0], wf.phase.values.shape[1])
            c = (t - r2) / k2  # (wf.phase.values.shape[0], wf.phase.values.shape[1])
            d = b * b - c  # (wf.phase.values.shape[0], wf.phase.values.shape[1])
            d_gt_0 = d > 0
            # if (d > 0)
            # 		z1 = -b - sqrt(d);
            # 		z2 = -b + sqrt(d);
            # else
            # 		z1 = z2 = zb + 1; /* Arbitrary interval which does not intersect [-zb, zb] */
            z1 = torch.full_like(d, zb + 1)
            z2 = torch.full_like(d, zb + 1)
            sqrt_d_gt0 = torch.sqrt(d[d_gt_0])
            b_masked = -b[d_gt_0]
            z1[d_gt_0] = b_masked - sqrt_d_gt0
            z2[d_gt_0] = b_masked + sqrt_d_gt0

            z2_le_neg_zb_or_z1_ge_zb = (z2 <= zb) | (z1 >= zb)
            path_length_support[z2_le_neg_zb_or_z1_ge_zb] = 2 * sk2 * zb
            path_length_ice[z2_le_neg_zb_or_z1_ge_zb] = 0.

            otherwise = ~z2_le_neg_zb_or_z1_ge_zb
            masked_path_length_support = path_length_support[otherwise]
            torch.fill_(masked_path_length_support, 0.)
            masked_z1 = z1[otherwise]
            masked_z2 = z2[otherwise]
            masked_s = s[otherwise]
            masked_t = t[otherwise]
            a_t_2_zc = a * masked_t + 2 * zc

            z1_lt_neg_zb = masked_z1 < -zb
            z1_ge_neg_zb = ~z1_lt_neg_zb
            masked_z3 = masked_z1.clone()
            masked_path_length_support[z1_ge_neg_zb] = sk2 * (masked_z1[z1_ge_neg_zb] + zb) + \
                                                       masked_path_length_support[z1_ge_neg_zb]
            masked_z3[z1_lt_neg_zb] = self._find_root(a * k2, a * masked_s[z1_lt_neg_zb] + 1, a_t_2_zc[z1_lt_neg_zb])

            z2_gt_zb = masked_z2 > zb
            z2_le_zb = ~z2_gt_zb
            masked_z4 = masked_z2.clone()
            masked_z4[z2_gt_zb] = self._find_root(a * k2, a * masked_s[z2_gt_zb] - 1, a_t_2_zc[z2_gt_zb])
            masked_path_length_support[z2_le_zb] = sk2 * (zb - masked_z2[z2_le_zb]) + \
                                                   masked_path_length_support[z2_le_zb]
            path_length_ice[otherwise] = sk2 * (masked_z4 - masked_z3)
            path_length_support[otherwise] = masked_path_length_support

    @property
    def device(self):
        return self.inel.device
