import logging
from copy import deepcopy
from pathlib import Path

import torch
import numpy as np

from . import ELEMENT_SYMBOL_TABLE

_logger = logging.getLogger(__name__)


def _write_log(string, *logging_strings):
    strings = [string, *logging_strings]
    if len(strings) > 1:
        logging_string = "\n    ".join(strings)
        _logger.debug(f"{_logger.name}:\n    {logging_string}")
    else:
        _logger.debug(f"{_logger.name}:  {strings[0]}")

def parse_line_pdb(line: str):
    # Bad line or not ATOM or not HETATM
    if len(line) < 78:
        return None
    elif line[:4] == 'ATOM' or line[:6] == 'HETATM':
        coord = np.empty(3)
        # Atom name, column 13-16
        atom_name = (line[12:16]).strip()

        # Residue name, column 18-20
        residue_name = (line[17:20]).strip()

        # x coordinate, column 31-38
        coord[0] = float(line[30:38])

        # y coordinate, column 39-46
        coord[1] = float(line[38:46])

        # z coordinate, column 47-54
        coord[2] = float(line[46:54])

        # Occupancy, column 55-60
        occupancy = float(line[54:60])

        # Temperature or B factor, column 61-66
        B_factor = float(line[60:66])

        # Element symbol, columns 77-78
        element_symbol = (line[76:78]).strip()

        if element_symbol == "":
            element_symbol = atom_name[0]

        if element_symbol in ELEMENT_SYMBOL_TABLE:
            atomic_number = ELEMENT_SYMBOL_TABLE.index(element_symbol) + 1
        else:
            atomic_number = 0

        pdb_record = [atom_name, residue_name, coord, occupancy, B_factor, element_symbol, atomic_number]
        return pdb_record
    else:
        return None

def parse_pdb(filepath: str):
    filepath = Path(filepath)
    try:
        with open(filepath) as f:
            lines = f.readlines()
        f.close()
        atom_names = []
        residues_names = []
        coordinates = []
        occupancies = []
        B_factors = []
        element_symbols = []
        atomic_numbers = []
        for line in lines:
            line = str(line)
            pdb_record = parse_line_pdb(line)
            if pdb_record is not None:
                atom_names.append(pdb_record[0])
                residues_names.append(pdb_record[1])
                coordinates.append(pdb_record[2])
                occupancies.append(pdb_record[3])
                B_factors.append(pdb_record[4])
                element_symbols.append(pdb_record[5])
                atomic_numbers.append(pdb_record[6])
        _write_log("Finish parsing pdb file")
        return atom_names, residues_names, coordinates, occupancies, B_factors, element_symbols, atomic_numbers
    except:
        _write_log("Could not find pdb in current working directory!")
        return None


class PDB_Protein:
    def __init__(self, filepath: str):
        pdb_id = Path(filepath).stem
        if parse_pdb(filepath) is not None:
            atom_names, residues_names, coordinates, occupancies, B_factors, element_symbols, atomic_numbers = parse_pdb(
                filepath)

            self._list_atom_name = atom_names
            self._list_residues = residues_names
            self._coordinates: torch.Tensor = torch.from_numpy(np.stack(coordinates, axis=0))
            self._occupancies: torch.Tensor = torch.from_numpy(np.array(occupancies))
            self._B_factors: torch.Tensor = torch.from_numpy(np.array(B_factors))
            self._list_element_symbol = element_symbols
            self._atomic_numbers: torch.Tensor = torch.from_numpy(np.array(atomic_numbers))

            self._add_hydrogen: bool = not ('H' in self._list_element_symbol)
            self._hydrogens: torch.Tensor = torch.zeros(len(self._list_atom_name))
            if self._add_hydrogen:
                self.add_hydrogen()
            # Only for spike proteins of SARS-CoV-2:
            lst_pdb_custom = ['so', 'sc', 'ntd_sc_merge']
            is_custom = any(word in pdb_id for word in lst_pdb_custom)
            if is_custom:
                self._B_factors = torch.full_like(self._B_factors, 50.)
                self._occupancies = torch.ones_like(self._occupancies)

    def to(self, device):
        pdb_protein = deepcopy(self)
        pdb_protein._coordinates = pdb_protein._coordinates.to(device)
        pdb_protein._occupancies = pdb_protein._occupancies.to(device)
        pdb_protein._B_factors = pdb_protein._B_factors.to(device)
        pdb_protein._atomic_numbers = pdb_protein._atomic_numbers.to(device)
        pdb_protein._hydrogens = pdb_protein._hydrogens.to(device)
        return pdb_protein

    def to_(self, device):
        self._coordinates = self._coordinates.to(device)
        self._occupancies = self._occupancies.to(device)
        self._B_factors = self._B_factors.to(device)
        self._atomic_numbers = self._atomic_numbers.to(device)
        self._hydrogens = self._hydrogens.to(device)

    def add_hydrogen(self):
        for i in range(len(self._list_atom_name)):
            if self._atomic_numbers[i] != 0:
                atom_name = self._list_atom_name[i]
                residue = self._list_residues[i]
                number_hydrogen = 0.0
                # Water molecule
                if residue == 'HOH':
                    number_hydrogen = 2.0
                # Peptide bond C and O
                elif atom_name == 'C' or atom_name == 'O':
                    number_hydrogen = 0.0
                # Peptide bond N
                elif atom_name == 'N':
                    if residue != 'PRO':
                        number_hydrogen = 1.0
                # Alpha carbon CA
                elif atom_name == 'CA':
                    if residue == 'GLY':
                        number_hydrogen = 2.0
                    else:
                        number_hydrogen = 1.0
                # Side chain position nomenclature depends on amino acid or base
                # ALA
                elif residue == 'ALA' and atom_name == 'CB':
                    number_hydrogen = 3.0
                # VAL
                elif residue == 'VAL':
                    if atom_name == 'CB':
                        number_hydrogen = 1.0
                    elif atom_name == 'CG1' or atom_name == 'CG2':
                        number_hydrogen = 3.0
                # LEU
                elif residue == 'LEU':
                    if atom_name == 'CB':
                        number_hydrogen = 2.0
                    elif atom_name == 'CG':
                        number_hydrogen = 1.0
                    elif atom_name == 'CD1' or atom_name == 'CD2':
                        number_hydrogen = 3.0
                # ILE
                elif residue == 'ILE':
                    if atom_name == 'CB':
                        number_hydrogen = 1.0
                    elif atom_name == 'CG1' or atom_name == 'CD1':
                        number_hydrogen = 3.0
                    elif atom_name == 'CG2':
                        number_hydrogen = 2.0
                # SER
                elif residue == 'SER':
                    if atom_name == 'CB':
                        number_hydrogen = 2.0
                    elif atom_name == 'OG':
                        number_hydrogen = 1.0
                # MET
                elif residue == 'MET':
                    if atom_name == 'CB' or atom_name == 'CG':
                        number_hydrogen = 2.0
                    elif atom_name == 'CE':
                        number_hydrogen = 3.0
                # THR
                elif residue == 'THR':
                    if atom_name == 'CB' or atom_name == 'OG1':
                        number_hydrogen = 1.0
                    elif atom_name == 'CG2':
                        number_hydrogen = 3.0
                # PHE
                elif residue == 'PHE':
                    if atom_name == 'CB':
                        number_hydrogen = 2.0
                    elif (atom_name == 'CD1'
                          or atom_name == 'CD2'
                          or atom_name == 'CE1'
                          or atom_name == 'CE2'
                          or atom_name == 'CZ'):
                        number_hydrogen = 1.0
                # TYR
                elif residue == 'TYR':
                    if atom_name == 'CB':
                        number_hydrogen = 2.0
                    elif (atom_name == 'CD1'
                          or atom_name == 'CD2'
                          or atom_name == 'CE1'
                          or atom_name == 'CE2'
                          or atom_name == 'OH'):
                        number_hydrogen = 1.0
                # TRP
                elif residue == 'TRP':
                    if atom_name == 'CB':
                        number_hydrogen = 2.0
                    elif (atom_name == 'CD1'
                          or atom_name == 'NE1'
                          or atom_name == 'CE3'
                          or atom_name == 'CZ2'
                          or atom_name == 'CZ3'
                          or atom_name == 'CH2'):
                        number_hydrogen = 1.0
                # CYS
                elif residue == 'CYS':
                    if atom_name == 'CB':
                        number_hydrogen = 2.0
                    elif atom_name == 'SG':
                        number_hydrogen = 1.0
                # PRO
                elif residue == 'PRO':
                    if (atom_name == 'CB'
                            or atom_name == 'CG'
                            or atom_name == 'CD'):
                        number_hydrogen = 2.0
                # ASP
                elif residue == 'ASP':
                    if atom_name == 'CB':
                        number_hydrogen = 2.0
                # ASN
                elif residue == 'ASN':
                    if atom_name == 'CB' or atom_name == 'ND2':
                        number_hydrogen = 2.0
                # GLU
                elif residue == 'GLU':
                    if atom_name == 'CB' or atom_name == 'CG':
                        number_hydrogen = 2.0
                # GLN
                elif residue == 'GLN':
                    if atom_name == 'CB' or atom_name == 'CG' or atom_name == 'NE2':
                        number_hydrogen = 2.0
                # HIS
                elif residue == 'HIS':
                    if atom_name == 'CB':
                        number_hydrogen = 2.0
                    elif (atom_name == 'ND1'
                          or atom_name == 'CD2'
                          or atom_name == 'CE1'
                          or atom_name == 'NE2'):
                        number_hydrogen = 1.0
                # ARG
                elif residue == 'ARG':
                    if (atom_name == 'CB'
                            or atom_name == 'CG'
                            or atom_name == 'CD'
                            or atom_name == 'NH1'
                            or atom_name == 'NH2'):
                        number_hydrogen = 2.0
                    elif atom_name == 'NE':
                        number_hydrogen = 1.0
                # LYS
                elif residue == 'LYS':
                    if (atom_name == 'CB'
                            or atom_name == 'CG'
                            or atom_name == 'CD'
                            or atom_name == 'CE'):
                        number_hydrogen = 2.0
                    elif atom_name == 'NZ':
                        number_hydrogen = 3.0
                # U
                elif residue == 'U':
                    if atom_name == 'C5\'':
                        number_hydrogen = 2.0
                    elif (atom_name == 'C4\''
                          or atom_name == 'C3\''
                          or atom_name == 'C2\''
                          or atom_name == 'O2\''
                          or atom_name == 'C1\''
                          or atom_name == 'N3'
                          or atom_name == 'C5'
                          or atom_name == 'C6'):
                        number_hydrogen = 1.0
                # T
                elif residue == 'T':
                    if atom_name == 'C5\'':
                        number_hydrogen = 2.0
                    elif (atom_name == 'C4\''
                          or atom_name == 'C3\''
                          or atom_name == 'C2\''
                          or atom_name == 'O2\''
                          or atom_name == 'C1\''
                          or atom_name == 'N3'
                          or atom_name == 'C6'):
                        number_hydrogen = 1.0
                    elif atom_name == 'C7':
                        number_hydrogen = 3.0
                # C
                elif residue == 'C':
                    if atom_name == 'C5\'' or atom_name == 'N4':
                        number_hydrogen = 2.0
                    elif (atom_name == 'C4\''
                          or atom_name == 'C3\''
                          or atom_name == 'C2\''
                          or atom_name == 'O2\''
                          or atom_name == 'C1\''
                          or atom_name == 'C5'
                          or atom_name == 'C6'):
                        number_hydrogen = 1.0
                # A
                elif residue == 'A':
                    if atom_name == 'C5\'' or atom_name == 'N6':
                        number_hydrogen = 2.0
                    elif (atom_name == 'C4\''
                          or atom_name == 'C3\''
                          or atom_name == 'C2\''
                          or atom_name == 'O2\''
                          or atom_name == 'C1\''
                          or atom_name == 'C8'
                          or atom_name == 'C2'):
                        number_hydrogen = 1.0
                # G
                elif residue == 'G':
                    if atom_name == 'C5\'' or atom_name == 'N2':
                        number_hydrogen = 2.0
                    elif (atom_name == 'C4\''
                          or atom_name == 'C3\''
                          or atom_name == 'C2\''
                          or atom_name == 'O2\''
                          or atom_name == 'C1\''
                          or atom_name == 'C8'
                          or atom_name == 'N1'):
                        number_hydrogen = 1.0
                # DT
                elif residue == 'DT':
                    if atom_name == 'C5\'' or atom_name == 'C2\'':
                        number_hydrogen = 2.0
                    elif (atom_name == 'C4\''
                          or atom_name == 'C3\''
                          or atom_name == 'C1\''
                          or atom_name == 'N3'
                          or atom_name == 'C6'):
                        number_hydrogen = 1.0
                    elif atom_name == 'C7':
                        number_hydrogen = 3.0
                # DC
                elif residue == 'DC':
                    if atom_name == 'C5\'' or atom_name == 'C2\'' or atom_name == 'N4':
                        number_hydrogen = 2.0
                    elif (atom_name == 'C4\''
                          or atom_name == 'C3\''
                          or atom_name == 'C1\''
                          or atom_name == 'C5'
                          or atom_name == 'C6'):
                        number_hydrogen = 1.0
                # DA
                elif residue == 'DA':
                    if atom_name == 'C5\'' or atom_name == 'C2\'' or atom_name == 'N6':
                        number_hydrogen = 2.0
                    elif (atom_name == 'C4\''
                          or atom_name == 'C3\''
                          or atom_name == 'C1\''
                          or atom_name == 'C8'
                          or atom_name == 'C2'):
                        number_hydrogen = 1.0
                # DG
                elif residue == 'DG':
                    if atom_name == 'C5\'' or atom_name == 'C2\'' or atom_name == 'N2':
                        number_hydrogen = 2.0
                    elif (atom_name == 'C4\''
                          or atom_name == 'C3\''
                          or atom_name == 'C1\''
                          or atom_name == 'C8'
                          or atom_name == 'N1'):
                        number_hydrogen = 1.0

                self._hydrogens[i] = number_hydrogen

    def transform_coordinate(self, transformation_matrix: torch.Tensor, numbers_particles):
        extend_coordinate = torch.cat((self._coordinates, torch.ones(self._coordinates.shape[0], 1)), dim=1)
        extend_coordinate = extend_coordinate.view(1, self._coordinates.shape[0], 4, 1)

        extend_transf = torch.cat(
            (transformation_matrix, torch.zeros(numbers_particles, 1, transformation_matrix.shape[2])),
            dim=1)
        extend_transf[:, 3, 3] = 1.0
        extend_transf = extend_transf.view(numbers_particles, 1, 4, 4).double()

        res = torch.matmul(extend_transf, extend_coordinate)
        res = res.view(*(res.size()[:3]))
        return res[:, :, :3]

    def find_box(self, transformation_matrix: torch.Tensor, padding, numbers_particles):
        box = torch.zeros((2, 3))
        transf_coords = self.transform_coordinate(transformation_matrix, numbers_particles)
        box[0] = torch.min(transf_coords.view(-1, 3), dim=0)[0] - padding
        box[1] = torch.max(transf_coords.view(-1, 3), dim=0)[0] + padding
        return box, transf_coords

    @property
    def list_atom_name(self):
        return self._list_atom_name

    @property
    def list_residues(self):
        return self._list_residues

    @property
    def list_element_symbol(self):
        return self._list_element_symbol

    @property
    def list_atomic_number(self):
        return self._atomic_numbers

    @property
    def get_coordinates(self):
        return self._coordinates

    @property
    def occupancies(self):
        return self._occupancies

    @property
    def B_factors(self):
        return self._B_factors

    @property
    def atomic_numbers(self):
        return self._atomic_numbers

    @property
    def hydrogens(self):
        return self._hydrogens

    @property
    def need_add_hydrogen(self):
        return self._add_hydrogen
