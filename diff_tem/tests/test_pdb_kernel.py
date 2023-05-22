from ..pdb.pdb_kernel import PDBKernel
import torch
import taichi as ti
from ..eletronbeam import Electronbeam
from ..particle import Particle
from ..pdb.pdb_protein import PDB_Protein


def test_runnable():
    ti.init(ti.cuda)
    ed = Electronbeam()
    ed.acc_voltage = 200.
    p = Particle()
    p.pdb_file_in = "test.pdb"
    p.pdb_transf_file_in = "test_transf.txt"
    p.pdb = PDB_Protein(p.pdb_file_in)
    p.voxel_size = 0.1
    p.use_imag_pot = True
    p.use_imag_pot = p.use_imag_pot
    p._get_pot_from_pdb_(ed)


