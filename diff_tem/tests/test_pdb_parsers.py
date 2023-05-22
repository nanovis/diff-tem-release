from diff_tem.pdb.pdb_protein import PDB_Protein
from diff_tem.particle import Particle
from diff_tem.eletronbeam import Electronbeam, wave_number, potential_conv_factor
import taichi as ti
import torch
import os

def test_read_spike():
    ti.init(ti.cuda)
    ed = Electronbeam()
    ed.acc_voltage = 200.0
    ed.energy_spread = 1.3
    ed.total_dose = 6000.
    ed.gen_dose = True
    ed.finish_init_(1)
    acc_en = torch.tensor(ed.acceleration_energy)
    print(ed.acc_voltage)

    p = Particle()
    p.pdb_file_in = "2OM3.pdb"
    p.pdb = PDB_Protein(p.pdb_file_in)
    p.pdb_transf_file_in = "2OM3_transf.txt"
    p.voxel_size = 0.1
    p.use_imag_pot = True
    p.use_defocus_corr = False
    p.famp = 0.
    p._get_pot_from_pdb_(ed)