from typing import Tuple, Optional

import taichi as ti
import torch
from . import NUM_SCATTERING_PARAMETERS, NUM_ELEM
from ..constants import PI
from ..constants.io_units import POTENTIAL_UNIT
from ..constants.physical_constants import PLANCKS_CONSTANT_SQ, ELECTRON_MASS, ELEMENTARY_CHARGE, ELEC_REST_ENERGY
from ..constants.derived_units import ONE_ANGSTROM_CU
from . import SCATTERING_PARAMETERS
from ..utils.helper_functions import check_tensor


@ti.func
def cast_int(a):
    return ti.cast(a, ti.int32)


@ti.func
def cast_float(a):
    return ti.cast(a, ti.f32)


@ti.data_oriented
class PDBKernel:
    def __init__(self):
        scfact = SCATTERING_PARAMETERS
        self.scfact_a = ti.Vector.field(NUM_SCATTERING_PARAMETERS, ti.f32)
        self.scfact_b = ti.Vector.field(NUM_SCATTERING_PARAMETERS, ti.f32)
        ti.root.dense(ti.i, NUM_ELEM).place(self.scfact_a, self.scfact_b)
        self.scfact_a.from_torch(scfact[:, :5].float())
        self.scfact_b.from_torch(scfact[:, 5:].float())

    @ti.func
    def get_scattering_parameters(self, atomic_number):
        a = self.scfact_a[atomic_number - 1]
        b = self.scfact_b[atomic_number - 1]
        return a, b

    @ti.func
    def add_gaussians_n_5(self, vol, coord, a, b, voxel_size):
        b *= voxel_size * voxel_size
        r2 = ti.max(0., (10. / b).max())
        r = ti.sqrt(r2)
        coord /= voxel_size
        kmin = ti.max(0, cast_int(ti.ceil(coord.z - r)))
        kmax = ti.min(vol.shape[0] - 1, cast_int(ti.floor(coord.z + r)))
        for k in range(kmin, kmax + 1):
            z = coord.z - cast_float(k)
            z2 = z * z
            s = ti.sqrt(r2 - z2)
            jmin = ti.max(0, cast_int(ti.ceil(coord.y - s)))
            jmax = ti.min(vol.shape[1] - 1, cast_int(ti.floor(coord.y + s)))
            for j in range(jmin, jmax + 1):
                y = coord.y - cast_float(j)
                v2 = z2 + y * y
                t = ti.sqrt(r2 - v2)
                imin = ti.max(0, cast_int(ti.ceil(coord.x - t)))
                imax = ti.min(vol.shape[2] - 1, cast_int(ti.floor(coord.x + t)))
                for i in range(imin, imax + 1):
                    x = coord.x - cast_float(i)
                    u2 = v2 + x * x
                    vol[k, j, i] += (a * ti.exp(-b * u2)).sum()

    @ti.func
    def add_gaussians_n_1(self, vol, coord, a, b, voxel_size):
        b *= voxel_size * voxel_size
        r2 = ti.max(0., 10. / b)
        r = ti.sqrt(r2)
        coord /= voxel_size
        kmin = ti.max(0, cast_int(ti.ceil(coord.z - r)))
        kmax = ti.min(vol.shape[0] - 1, cast_int(ti.floor(coord.z + r)))
        for k in range(kmin, kmax + 1):
            z = coord.z - cast_float(k)
            z2 = z * z
            s = ti.sqrt(r2 - z2)
            jmin = ti.max(0, cast_int(ti.ceil(coord.y - s)))
            jmax = ti.min(vol.shape[1] - 1, cast_int(ti.floor(coord.y + s)))
            for j in range(jmin, jmax + 1):
                y = coord.y - cast_float(j)
                v2 = z2 + y * y
                t = ti.sqrt(r2 - v2)
                imin = ti.max(0, cast_int(ti.ceil(coord.x - t)))
                imax = ti.min(vol.shape[2] - 1, cast_int(ti.floor(coord.x + t)))
                for i in range(imin, imax + 1):
                    x = coord.x - cast_float(i)
                    u2 = v2 + x * x
                    vol[k, j, i] += a * ti.exp(-b * u2)

    @ti.func
    def add_atomic_pot(self, vol, coord, atomic_num, temp_fact, voxel_size, occ):
        if atomic_num > 0:
            a, b = self.get_scattering_parameters(atomic_num)
            a *= occ
            b += temp_fact
            b = ti.max(b, 200. * voxel_size * voxel_size)
            a *= 2132.79 * pow(b, -1.5)
            b = 4. * PI * PI / b
            self.add_gaussians_n_5(vol, coord, a, b, voxel_size)

    @ti.func
    def add_atomic_abs_pot(self,
                           pot_im,
                           coord,
                           temp_fact,
                           voxel_size,
                           occ,
                           acc_en,
                           wave_number_p,
                           cross_sec_p):
        b = 4 * PI * PI / ti.max(temp_fact + 20., 200. * voxel_size * voxel_size)
        a = PLANCKS_CONSTANT_SQ / (8. * PI * PI * ELECTRON_MASS * ELEMENTARY_CHARGE * ONE_ANGSTROM_CU * POTENTIAL_UNIT) \
            * occ * wave_number_p / (1. + acc_en / ELEC_REST_ENERGY) * pow(b / PI, 1.5) * cross_sec_p
        if b != 0.:
            self.add_gaussians_n_1(pot_im, coord, a, b, voxel_size)

    @ti.kernel
    def _get_pot_re_im_from_pdb_kernel(self,
                                       pot_re: ti.template(),
                                       pot_im: ti.template(),
                                       coords: ti.template(),
                                       atomic_numbers: ti.template(),
                                       temp_facts: ti.template(),
                                       occs: ti.template(),
                                       cross_sec_ps: ti.template(),
                                       vox_sz_angst: float,
                                       acc_en: float,
                                       wave_number_p: float):
        for pdb_idx, atom_idx in coords:
            self.add_atomic_pot(pot_re,
                                coords[pdb_idx, atom_idx],
                                atomic_numbers[atom_idx],
                                temp_facts[atom_idx],
                                vox_sz_angst,
                                occs[atom_idx])
            self.add_atomic_abs_pot(pot_im,
                                    coords[pdb_idx, atom_idx],
                                    temp_facts[atom_idx],
                                    vox_sz_angst,
                                    occs[atom_idx],
                                    acc_en,
                                    wave_number_p,
                                    cross_sec_ps[atom_idx])

    @ti.kernel
    def _get_pot_re_from_pdb_kernel(self,
                                    pot_re: ti.template(),
                                    coords: ti.template(),
                                    atomic_numbers: ti.template(),
                                    temp_facts: ti.template(),
                                    occs: ti.template(),
                                    vox_sz_angst: float):
        for pdb_idx, atom_idx in coords:
            self.add_atomic_pot(pot_re,
                                coords[pdb_idx, atom_idx],
                                atomic_numbers[atom_idx],
                                temp_facts[atom_idx],
                                vox_sz_angst,
                                occs[atom_idx])

    def get_pot_re_from_pdb(self,
                            pot_shape: Tuple[int, int, int],
                            coords: torch.Tensor,
                            atomic_numbers: torch.Tensor,
                            temp_facts: torch.Tensor,
                            occs: torch.Tensor,
                            vox_sz_angst: float,
                            pot_re_value: Optional[torch.Tensor] = None):
        # check inputs
        check_tensor(coords, dimensionality=3)
        pdb_num = coords.shape[0]
        atom_num = coords.shape[1]
        assert coords.shape[2] == 3
        assert atom_num == atomic_numbers.shape[0] \
               and atom_num == temp_facts.shape[0] \
               and atom_num == occs.shape[0]
        assert coords.device == atomic_numbers.device \
               and coords.device == temp_facts.device \
               and coords.device == occs.device
        fields_builder = ti.FieldsBuilder()
        # declare fields
        pot_re = ti.field(ti.f32)
        coord_field = ti.Vector.field(3, dtype=ti.f32)
        atomic_number_field = ti.field(ti.int32)
        temp_fact_field = ti.field(ti.f32)
        occ_field = ti.field(ti.f32)
        # place fields together
        fields_builder.dense(ti.ijk, pot_shape).place(pot_re)
        fields_builder.dense(ti.ij, (pdb_num, atom_num)).place(coord_field)
        fields_builder.dense(ti.i, atom_num).place(atomic_number_field, temp_fact_field, occ_field)
        snode = fields_builder.finalize()
        # load fields
        if pot_re_value is None:
            pot_re.fill(0.)
        else:
            pot_re.from_torch(pot_re_value.float())
        coord_field.from_torch(coords)
        atomic_number_field.from_torch(atomic_numbers.int())
        temp_fact_field.from_torch(temp_facts.float())
        occ_field.from_torch(occs.float())
        self._get_pot_re_from_pdb_kernel(pot_re,
                                         coord_field, atomic_number_field, temp_fact_field, occ_field,
                                         vox_sz_angst)
        pot_re_tensor = pot_re.to_torch(device=coords.device)
        snode.destroy()
        return pot_re_tensor

    def get_pot_re_im_from_pdb(self,
                               pot_shape: Tuple[int, int, int],
                               coords: torch.Tensor,
                               atomic_numbers: torch.Tensor,
                               temp_facts: torch.Tensor,
                               occs: torch.Tensor,
                               vox_sz_angst: float,
                               cross_sec_ps: torch.Tensor,
                               acc_en: float,
                               wave_number_p: float,
                               pot_re_value: Optional[torch.Tensor] = None,
                               pot_im_value: Optional[torch.Tensor] = None):
        # check inputs
        check_tensor(coords, dimensionality=3)
        pdb_num = coords.shape[0]
        atom_num = coords.shape[1]
        assert coords.shape[2] == 3
        assert atom_num == atomic_numbers.shape[0] \
               and atom_num == temp_facts.shape[0] \
               and atom_num == occs.shape[0] \
               and atom_num == cross_sec_ps.shape[0]
        assert coords.device == atomic_numbers.device \
               and coords.device == temp_facts.device \
               and coords.device == occs.device \
               and coords.device == cross_sec_ps.device

        fields_builder = ti.FieldsBuilder()
        # declare fields
        pot_re = ti.field(ti.f32)
        pot_im = ti.field(ti.f32)
        coord_field = ti.Vector.field(3, dtype=ti.f32)
        atomic_number_field = ti.field(ti.int32)
        temp_fact_field = ti.field(ti.f32)
        occ_field = ti.field(ti.f32)
        cross_sec_ps_field = ti.field(ti.f32)
        # place fields together
        fields_builder.dense(ti.ijk, pot_shape).place(pot_re, pot_im)
        fields_builder.dense(ti.ij, (pdb_num, atom_num)).place(coord_field)
        fields_builder.dense(ti.i, atom_num).place(atomic_number_field, temp_fact_field, occ_field, cross_sec_ps_field)
        snode = fields_builder.finalize()
        # load fields
        if pot_re_value is None:
            pot_re.fill(0.)
        else:
            pot_re.from_torch(pot_re_value.float())
        if pot_im_value is None:
            pot_im.fill(0.)
        else:
            pot_im.from_torch(pot_im_value.float())
        coord_field.from_torch(coords.float())
        atomic_number_field.from_torch(atomic_numbers.int())
        temp_fact_field.from_torch(temp_facts.float())
        occ_field.from_torch(occs.float())
        cross_sec_ps_field.from_torch(cross_sec_ps.float())
        self._get_pot_re_im_from_pdb_kernel(pot_re, pot_im,
                                            coord_field, atomic_number_field, temp_fact_field, occ_field,
                                            cross_sec_ps_field,
                                            vox_sz_angst, acc_en, wave_number_p)
        pot_re_tensor = pot_re.to_torch(coords.device)
        pot_im_tensor = pot_im.to_torch(coords.device)
        snode.destroy()
        return pot_re_tensor, pot_im_tensor
