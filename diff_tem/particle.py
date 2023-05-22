import logging
import math
from copy import deepcopy
from enum import Enum
from typing import Any, List, Dict, Union

import mrcfile
import torch

from . import UNINITIALIZED
from .constants import PI
from .constants.derived_units import ONE_ANGSTROM
from .constants.io_units import POTENTIAL_UNIT, INT_FILE_POT_UNIT
from .eletronbeam import Electronbeam, total_cross_section, wave_number
from .pdb.pdb_kernel import PDBKernel
from .pdb.pdb_protein import PDB_Protein
from .sample import sample_ice_pot, sample_ice_abs_pot
from .utils import BinaryFileFormat
from .utils.conversion import degree_to_radian
from .utils.exceptions import ParameterLockedException, WrongArgumentTypeException, \
    RequiredParameterNotSpecifiedException
from .utils.helper_functions import check_tensor, eprint
from .utils.mrc import write_tensor_to_mrc
from .utils.parameter import Parameter
from .utils.tensor_io import read_matrix_text, read_tensor_from_binary, write_tensor_to_binary, ByteOrder, \
    need_reverse_byte_order
from .utils.tensor_ops import boundary_mean, DiscreteLaplace, change_axes
from .utils.transformations import rotate_3d_rows
from .vector_valued_function import VectorValuedFunction
from .utils.input_processing import convert_bool

_logger = logging.getLogger(__name__)


def _write_log(string, *logging_strings):
    strings = [string, *logging_strings]
    if len(strings) > 1:
        logging_string = "\n    ".join(strings)
        _logger.debug(f"{_logger.name}:\n    {logging_string}", stacklevel=2)
    else:
        _logger.debug(f"{_logger.name}:  {strings[0]}", stacklevel=2)


class Particle:
    """
    A particle component defines a single particle, for example a macromolecule.
    Once a particle is defined, one or several copies of the particle
    can be placed in the sample using a particleset component.
    A simulation of micrographs can use one or several particle components,
    or possibly none at all if an empty sample is to be simulated.
    """

    class Parameters(Enum):
        SOURCE = Parameter("source", str,
                           description="Defines what kind of source of information is used to obtain a potential map. "
                                       "Possible sources of the potential map are data in a PDB file, "
                                       "reading a map directly from an MRC file or a raw data file, "
                                       "or generating a random map.")
        USE_IMAG_POT = Parameter("use_imag_pot", bool,
                                 description="Controls whether a separate map of an \"imaginary\" absorption potential "
                                             "should be used in the simulation along with the real valued electrostatic "
                                             "potential. If a separate map is not used, "
                                             "the imaginary potential is taken to be a multiple of the real potential, "
                                             "given by the parameter famp. "
                                             "When using a PDB file as source, \"use_imag_pot\" should be set to yes "
                                             "in order to make the simulation as correct as possible.")
        FAMP = Parameter("famp", float,
                         description="If use_imag_pot is set to no, "
                                     "the same map is used to define both the real and the imaginary potential. "
                                     "The imaginary potential is taken as the map multiplied by famp, "
                                     "and the real potential as the map multiplied by sqrt(1-famp^2).")
        USE_DEFOCUS_CORR = Parameter("use_defocus_corr", bool,
                                     description="A thick sample is always split into a number of slices to account "
                                                 "for the varying defocus within the sample. "
                                                 "This parameter tells the simulator to also correct for "
                                                 "the varying defocus within each slice "
                                                 "when simulating the interaction with this particle type. "
                                                 "The computational work increases and the effect "
                                                 "on the end result is usually negligible.")
        VOXEL_SIZE = Parameter("voxel_size", float,
                               description="The voxel size in nm of the map of "
                                           "the particle potential used in the simulations.")
        NX = Parameter("NX", int,
                       description="The number of voxels along the x-axis in the map of the particle potential. "
                                   "The size of the particle map must be specified "
                                   "if the map is to be read from a raw data file (without header) or randomly generated. "
                                   "It can optionally be specified if the map is generated from a PDB file. "
                                   "If the map is read from an MRC file the size is determined by the file header.")
        NY = Parameter("NY", int,
                       description="The number of voxels along the y-axis in the map of the particle potential. "
                                   "The size of the particle map must be specified "
                                   "if the map is to be read from a raw data file (without header) or randomly generated. "
                                   "It can optionally be specified if the map is generated from a PDB file. "
                                   "If the map is read from an MRC file the size is determined by the file header."
                       )
        NZ = Parameter("NZ", int,
                       description="The number of voxels along the z-axis in the map of the particle potential. "
                                   "The size of the particle map must be specified "
                                   "if the map is to be read from a raw data file (without header) or randomly generated. "
                                   "It can optionally be specified if the map is generated from a PDB file. "
                                   "If the map is read from an MRC file the size is determined by the file header."
                       )
        MAP_FILE_RE_IN = Parameter("map_file_re_in", str,
                                   description="The name of a file from which the potential map is read "
                                               "if source is set to \"map\".")
        MAP_FILE_IM_IN = Parameter("map_file_im_in", str,
                                   description="The name of a file "
                                               "from which the imaginary absorption potential map is read "
                                               "if source is set to \"map\" and use_imag_pot is set to yes.")
        MAP_FILE_RE_OUT = Parameter("map_file_re_out", str,
                                    description="The name of a file to which the potential map is written "
                                                "if \"source\" is set to \"pdb\" or \"random\". "
                                                "If the parameter is not given, the potential map is not written to file.")
        MAP_FILE_IM_OUT = Parameter("map_file_im_out", str,
                                    description="The name of a file "
                                                "to which the imaginary absorption potential map is written "
                                                "if source is set to \"pdb\" or \"random\" and use_imag_pot is set to yes. "
                                                "If the parameter is not given, the imaginary potential map is not written to file.")
        MAP_FILE_FORMAT = Parameter("map_file_format", BinaryFileFormat,
                                    description="The format of files used for input and "
                                                "output of potential maps. \"mrc\" means "
                                                "MRC file with 4-byte floating point data, "
                                                "\"mrc-int\" means MRC file with 2-byte integer data. "
                                                "\"raw\" is the same format as \"mrc\" but without the file header. "
                                                "For input files, \"mrc\" and \"mrc-int\" are equivalent, "
                                                "the type of MRC file is determined by the file header. "
                                                "The potential map is assumed to be in V for floating point file formats, "
                                                "and mV for integer file formats.")
        MAP_FILE_BYTE_ORDER = Parameter("map_file_byte_order", ByteOrder,
                                        description="Controls if input and output binary files "
                                                    "are in little endian or big endian format. "
                                                    "For MRC input files the parameter is ignored, "
                                                    "instead the format is determined by the file header.")
        MAP_AXIS_ORDER = Parameter("map_axis_order", str,
                                   description="Controls the order in which voxel values appear in input and output map files. "
                                               "The first letter is the fastest varying index, and the last letter is the slowest varying. "
                                               "\"xyz\" means that the simulator's use of x, y, and z coordinates "
                                               "is consistent with the MRC file convention.")
        PDB_FILE_IN = Parameter("pdb_file_in", str,
                                description="The name of a PDB file used to generate a potential map "
                                            "if source is set to \"pdb\".")
        PDB_TRANSF_FILE_IN = Parameter("pdb_transf_file_in", str,
                                       description="The name of a text file "
                                                   "which defines a number of geometric transformations "
                                                   "applied to the atomic coordinates in the PDB file. "
                                                   "This can be used to define particles built up of several identical subunits. "
                                                   "If the parameter is not given, "
                                                   "the atomic coordinates are used just as they appear in the PDB file "
                                                   "without any transformation applied. "
                                                   "See the manual for details on the file format.")
        ADD_HYDROGEN = Parameter("add_hydrogen", bool,
                                 description="Tells the simulator to add hydrogen atoms, "
                                             "which are usually not specified in PDB files, "
                                             "at the appropriate locations in the amino acid residuals. "
                                             "The hydrogen atoms are added at the coordinates of the atom to which they are bonded. "
                                             "This is of course not really correct, but it is a good enough approximation for many purposes.")
        CONTRAST_RE = Parameter("contrast_re", float,
                                description="The overall contrast level of a randomly generated potential map.")
        CONTRAST_IM = Parameter("contrast_im", float,
                                description="The overall contrast level of a randomly generated imaginary absorption potential map.")
        SMOOTHNESS = Parameter("smoothness", float,
                               description="The smoothness of randomly generated maps. "
                                           "Higher values means smoother maps. "
                                           "The numerical value does not have any obvious physical interpretation.")
        MAKE_POSITIVE = Parameter("make_positive", bool,
                                  description="Tells the simulator to make random maps "
                                              "always have values above the background level, "
                                              "rather than randomly going above and below the background level. "
                                              "(Values always below the background value can be accomplished "
                                              "by setting \"make_positive\" to \"yes\" and making \"contrast_re\" "
                                              "and \"contrast_im\" negative.)")

        def type(self):
            return self.value.type

    MAP_AXIS_ORDER_OPTIONS = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]
    SOURCE_OPTIONS = ["map", "pdb", "random"]

    REQUIRED_PARAMETERS = [Parameters.SOURCE]
    PARAMETER_DEFAULTS = {
        Parameters.USE_IMAG_POT: True,
        Parameters.USE_DEFOCUS_CORR: False,
        Parameters.MAP_FILE_FORMAT: BinaryFileFormat.MRC,
        Parameters.MAP_FILE_BYTE_ORDER: ByteOrder.NATIVE,
        Parameters.MAP_AXIS_ORDER: MAP_AXIS_ORDER_OPTIONS[0],
        Parameters.ADD_HYDROGEN: True,
        Parameters.FAMP: 0.0
    }

    ELEMENT_SYMBOLS = ["H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE",
                       "NA", "MG", "AL", "SI", "P", "S", "CL", "AR", " K", "CA",
                       "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN",
                       "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR", "Y", "ZR",
                       "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN",
                       "SB", "TE", "I", "XE", "CS", "BA", "LA", "CE", "PR", "ND",
                       "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB",
                       "LU", "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG",
                       "TL", "PB", "BI", "PO", "AT", "RN", "FR", "RA", "AC", "TH",
                       "PA", "U", "NP", "PU", "AM", "CM", "BK", "CF"]

    def __init__(self, name: str = None):
        self.name = name
        self.pot_re: torch.Tensor = UNINITIALIZED
        self.pot_im: torch.Tensor = UNINITIALIZED
        self.lap_pot_re: torch.Tensor = UNINITIALIZED
        self.lap_pot_im: torch.Tensor = UNINITIALIZED
        self.need_rev_byte_order: bool = UNINITIALIZED
        self.pdb: PDB_Protein = UNINITIALIZED
        parameters = {}
        for p in self.Parameters:
            if p in self.PARAMETER_DEFAULTS:
                parameters[p] = self.PARAMETER_DEFAULTS[p]
            else:
                parameters[p] = UNINITIALIZED
        self.pdb_kernel: PDBKernel = PDBKernel()
        self._parameters = parameters
        self._initialized = False
        self._lock_parameters = False

    def to(self, device: torch.device):
        """
        Out-of-place move to a device, creating new copy of self whose data is on the device

        :param device: device that holds data
        :return: a new copy of self
        """
        assert self._initialized, "Initialize particle first"
        particle = Particle(self.name)
        particle.pot_re = self.pot_re.to(device)
        particle.pot_im = self.pot_im.to(device)
        particle.lap_pot_re = self.lap_pot_re.to(device)
        particle.lap_pot_im = self.lap_pot_im.to(device)
        particle.need_rev_byte_order = self.need_rev_byte_order
        particle.pdb = self.pdb.to(device)
        particle._parameters = deepcopy(self._parameters)
        particle._initialized = self._initialized
        particle._lock_parameters = self._lock_parameters
        return particle

    def to_(self, device: torch.device):
        """
        In-place move to a device

        :param device: device that holds data
        """
        assert self._initialized, "Initialize particle first"
        self.pot_re = self.pot_re.to(device)
        self.pot_im = self.pot_im.to(device)
        self.lap_pot_re = self.lap_pot_re.to(device)
        self.lap_pot_im = self.lap_pot_im.to(device)
        self.pdb.to_(device)

    @staticmethod
    def prototype_from_parameters(parameters: Union[List[Dict[str, str]], Dict[str, str]]):
        """
        Construct **uninitialized** instances of Particle given parameters

        :param parameters: list of parameter dictionaries or a dictionary containing parameters
        :return: list of instances or one instance
        """
        if isinstance(parameters, list):
            return [Particle._from_parameters(parameter_table) for parameter_table in parameters]
        elif isinstance(parameters, dict):
            return Particle._from_parameters(parameters)
        else:
            raise Exception(f"Invalid parameter of type {type(parameters)}")

    @staticmethod
    def _from_parameters(parameters: Dict[str, str]):
        if "name" in parameters:
            particle = Particle(name=parameters["name"])
        else:
            particle = Particle()
        for p in Particle.Parameters:
            parameter_name = p.value.name
            parameter_type = p.type()
            if parameter_type == bool:
                parameter_type = convert_bool
            if parameter_name in parameters:
                parameter_val = parameter_type(parameters[parameter_name])
                setattr(particle, parameter_name, parameter_val)
        return particle

    def finish_init_(self, simulation):
        """
        Finish initialization
        :param simulation: Simulation instance
        :return: self
        """
        if self._initialized:
            return self
        for parameter_type in self.REQUIRED_PARAMETERS:
            if self._parameters[parameter_type] is None:
                raise RequiredParameterNotSpecifiedException(parameter_type)
        # particle_check_input
        use_imag_pot = self.use_imag_pot
        self.need_rev_byte_order = need_reverse_byte_order(self.map_file_byte_order)
        source = self.source
        if source == self.SOURCE_OPTIONS[0]:  # map
            pot_re, pot_im = self._read_maps_()
            self.pot_re = pot_re
            self.pot_im = pot_im
        elif source == self.SOURCE_OPTIONS[1]:  # pdb
            ed = simulation.electronbeam
            pdb_filepath = self.pdb_file_in
            self.pdb = PDB_Protein(pdb_filepath)
            pot_re, pot_im = self._get_pot_from_pdb_(ed)
            self.pot_re = pot_re
            self.pot_im = pot_im
            _write_log("Particle map generated from pbd file.")
            self._write_maps()
        else:  # random
            nx = self.nx
            ny = self.ny
            nz = self.nz
            smoothness = self.smoothness
            make_positive = self.make_positive
            pot_re = self._gen_random_particle(nx, ny, nz, self.contrast_re, smoothness, make_positive)
            self.pot_re = pot_re
            if use_imag_pot:
                self.pot_im = self._gen_random_particle(nx, ny, nz, self.contrast_im, smoothness, make_positive)
            self._write_maps()

        self.pot_re = self.pot_re - boundary_mean(self.pot_re)
        if self.use_imag_pot:
            self.pot_im = self.pot_im - boundary_mean(self.pot_im)
        if self.use_defocus_corr:
            voxel_size = self.voxel_size
            laplace_op = DiscreteLaplace().to(self.pot_re.device)
            lap_pot_re = laplace_op(self.pot_re)
            _write_log(f"lap_pot_re.shape = {lap_pot_re.shape}")
            self.lap_pot_re = 1.0 / (voxel_size * voxel_size) * lap_pot_re
            if self.use_imag_pot:
                lap_pot_im = laplace_op(self.pot_im)
                self.lap_pot_im = 1.0 / (voxel_size * voxel_size) * lap_pot_im

        self._initialized = True
        self._lock_parameters = True
        _write_log("Particle component initialized.")

    def _read_maps_(self):
        pot_re = self._read_map_(self.map_file_re_in)
        pot_im = None
        if self.use_imag_pot:
            pot_im = self._read_map_(self.map_file_im_in)
            check_tensor(pot_im, shape=pot_re.shape)
        return pot_re, pot_im

    def _read_map_(self, filepath):
        map_axis_order = self.map_axis_order
        format = self.map_file_format
        if format == BinaryFileFormat.RAW:
            raw_tensor = read_tensor_from_binary(filepath, torch.float32, endian=self.map_file_byte_order)
            axes = []
            for a in map_axis_order:
                if a == 'x':
                    axes.append(self.nx)
                elif a == 'y':
                    axes.append(self.ny)
                else:
                    axes.append(self.nz)
            pot_map = raw_tensor.reshape(*axes)
        else:  # mrc-int or mrc
            with mrcfile.open(filepath, mode="r", permissive=True) as mrc:
                data = mrc.data
                # MRC-format: https://www.ccpem.ac.uk/mrc_format/mrc2014.php
                voxel_size = mrc.header.cella.sum()
                n = data.shape.sum()
                voxel_size /= n
                if voxel_size != 0.0:
                    for i in range(3):
                        if abs(1.0 - mrc.header.cella[i] / (voxel_size * data.shape[i])) > 0.01:
                            eprint(f"Map read from MRC file {filepath} has non-cubic voxels, "
                                   f"which will not be handled correctly.")
                            break
            voxel_size *= ONE_ANGSTROM
            if self._parameters[self.Parameters.VOXEL_SIZE] is not None:
                if voxel_size != 0.0 and abs(1.0 - self.voxel_size / voxel_size) > 0.01:
                    eprint(f"Voxel size set in input does not agree with voxel size found in MRC file {filepath}.")
            else:
                if voxel_size > 0.0:
                    self.voxel_size = voxel_size
                else:
                    raise Exception(f"Voxel size of MRC file {filepath} could not be determined. "
                                    f"Set voxel size of particle {self.name} manually.\n")
            pot_map = torch.from_numpy(data)
            is_integer_mode = pot_map.dtype == torch.int16
            if is_integer_mode:
                pot_map *= INT_FILE_POT_UNIT / POTENTIAL_UNIT
        # change it back to x-y-z order
        pot_map = change_axes(pot_map, map_axis_order)  # op of changing axes is the inverse of itself
        _write_log(f"Particle map read from file {filepath}.")
        return pot_map

    def _write_maps(self):
        map_file_re_out = self._parameters[self.Parameters.MAP_FILE_RE_OUT]
        map_file_im_out = self._parameters[self.Parameters.MAP_FILE_IM_OUT]
        if map_file_re_out is not None:
            self._write_map(self.pot_re, map_file_re_out)
        if map_file_im_out is not None:
            self._write_map(self.pot_im, map_file_im_out)

    def _find_box(self, transf: torch.Tensor, padding):
        box = self.pdb.find_box(transf, padding, transf.shape[0])
        return box

    def _atom_name_lookup(self, number):
        return self.ELEMENT_SYMBOLS[number - 1]

    def _get_pot_from_pdb_(self, ed: Electronbeam):
        vox_sz_angst = self.voxel_size / ONE_ANGSTROM
        _write_log(f"Generating potential from pdb file {self.pdb_file_in}.")
        transf = self._read_pdf_trans_file()
        padding = 5.
        box, transf_coords = self._find_box(transf, padding)

        try:
            nx = self.nx
        except RequiredParameterNotSpecifiedException:
            nx = math.ceil((box[1, 0] - box[0, 0]) / vox_sz_angst)
            self.nx = nx

        try:
            ny = self.ny
        except RequiredParameterNotSpecifiedException:
            ny = math.ceil((box[1, 1] - box[0, 1]) / vox_sz_angst)
            self.ny = ny

        try:
            nz = self.nz
        except RequiredParameterNotSpecifiedException:
            nz = math.ceil((box[1, 2] - box[0, 2]) / vox_sz_angst)
            self.nz = nz

        box[0, 0] += 0.5 * (box[1, 0] - box[0, 0] - (nx - 1) * vox_sz_angst)
        box[0, 1] += 0.5 * (box[1, 1] - box[0, 1] - (ny - 1) * vox_sz_angst)
        box[0, 2] += 0.5 * (box[1, 2] - box[0, 2] - (nz - 1) * vox_sz_angst)

        pot_im = None
        transf_coords = transf_coords - box[0]

        pot_shape = (nz, ny, nx)

        if self.pdb.need_add_hydrogen:
            nh_occ = self.pdb.hydrogens * self.pdb.occupancies
            nh_atomic_number = torch.zeros_like(self.pdb.atomic_numbers)
            index_hydrogen = self.pdb.hydrogens != 0
            nh_atomic_number[index_hydrogen] = 1

        if self.use_imag_pot:
            assert ed is not None, "Can't determine acceleration energy needed to compute absorption potential"
            acc_en = ed.acceleration_energy
            wave_number_p = wave_number(acc_en).item()

            cross_sec_ps = total_cross_section(acc_en, self.pdb.atomic_numbers)
            torch.set_printoptions(precision=10)

            pot_re, pot_im = self.pdb_kernel.get_pot_re_im_from_pdb(pot_shape,
                                                                    transf_coords,
                                                                    self.pdb.atomic_numbers,
                                                                    self.pdb.B_factors,
                                                                    self.pdb.occupancies,
                                                                    vox_sz_angst,
                                                                    cross_sec_ps,
                                                                    acc_en.item(),
                                                                    wave_number_p)

            if self.pdb.need_add_hydrogen:
                pot_re, pot_im = self.pdb_kernel.get_pot_re_im_from_pdb(pot_shape,
                                                                        transf_coords,
                                                                        nh_atomic_number,
                                                                        self.pdb.B_factors,
                                                                        nh_occ,
                                                                        vox_sz_angst,
                                                                        cross_sec_ps,
                                                                        acc_en.item(),
                                                                        wave_number_p,
                                                                        pot_re_value=pot_re,
                                                                        pot_im_value=pot_im)

        else:
            pot_re = self.pdb_kernel.get_pot_re_from_pdb(pot_shape,
                                                         transf_coords,
                                                         self.pdb.atomic_numbers,
                                                         self.pdb.B_factors,
                                                         self.pdb.occupancies,
                                                         vox_sz_angst)
            if self.pdb.need_add_hydrogen:
                pot_re = self.pdb_kernel.get_pot_re_from_pdb(pot_shape,
                                                             transf_coords,
                                                             nh_atomic_number,
                                                             self.pdb.B_factors,
                                                             nh_occ,
                                                             vox_sz_angst,
                                                             pot_re_value=pot_re)

        solvent_pot = sample_ice_pot().clone()
        pot_re = torch.maximum(pot_re, solvent_pot)
        if self.use_imag_pot:
            solvent_pot = sample_ice_abs_pot(acc_en).clone()
            pot_im = torch.maximum(pot_im, solvent_pot)

        _write_log(f"Size of particle map: {nx} x {ny} x {nz}")
        _write_log(f"PDB coordinates of map center: "
                   f"({box[0, 0] + 0.5 * (nx - 1) * vox_sz_angst}, "
                   f"{box[0, 1] + 0.5 * (ny - 1) * vox_sz_angst}, "
                   f"{box[0, 2] + 0.5 * (nz - 1) * vox_sz_angst})")

        element_unique, count_element_unique = torch.unique(self.pdb.list_atomic_number, return_counts=True)
        num_atoms = [0 for _ in range(len(self.ELEMENT_SYMBOLS) + 1)]
        element_count_strings = []
        for i in range(0, len(element_unique)):
            name = self._atom_name_lookup(element_unique[i])
            element_count_strings.append(f"{name} : {count_element_unique[i]}")
            num_atoms[element_unique[i]] += count_element_unique[i]
        _write_log("Number of atoms used to compute potential:", *element_count_strings)
        if num_atoms[0] > 0:
            _write_log(f"Unknown entries in PDB file: {num_atoms[0]}")
        _write_log(f"Number of transformed copies of PDB model: {transf.shape[0]}")

        self.pot_re = pot_re
        self.pot_im = pot_im

        return pot_re, pot_im

    def _read_pdf_trans_file(self) -> torch.Tensor:
        pdb_transf_file_in = self._parameters[self.Parameters.PDB_TRANSF_FILE_IN]
        if pdb_transf_file_in is not None:
            b = read_matrix_text(pdb_transf_file_in)
            x_axis = torch.tensor([1., 0., 0.])
            z_axis = torch.tensor([0., 0., 1.])
            if b.shape[1] == 6:
                translations = b[:, :3]
                rotations = b[:, 3:]
                rotations = -degree_to_radian(rotations)
                rotation_matrices = []
                for rotation_vec in rotations:
                    r0 = rotate_3d_rows(z_axis, rotation_vec[2])
                    r1 = rotate_3d_rows(x_axis, rotation_vec[1])
                    r2 = rotate_3d_rows(z_axis, rotation_vec[0])
                    rotation_matrix = r0 @ r1 @ r2
                    rotation_matrices.append(rotation_matrix)
                transformation_matrices = []
                for i in range(b.shape[0]):
                    translation = translations[i].reshape(3, 1)
                    rotation_matrix = rotation_matrices[i]
                    transformation_matrix = torch.cat([rotation_matrix, translation], dim=-1)
                    transformation_matrices.append(transformation_matrix)
                transformations = torch.stack(transformation_matrices, dim=0)
            elif b.shape[1] == 4:
                transformations = b.view(-1, 3, 4)
            else:
                raise Exception(f"Error reading pdb transform file {pdb_transf_file_in}: {b.shape[1]} columns found,"
                                f" but should be 4 or 6.")
        else:
            transformations = torch.eye(n=3, m=4)  # (3,4) matrix
            transformations = transformations.unsqueeze(0)
        return transformations  # (particle_num,3,4) tensor

    @staticmethod
    @torch.jit.script
    def _hits_wavefunction(x_basis: torch.Tensor,
                           y_basis: torch.Tensor,
                           offset: torch.Tensor,
                           project_matrix: torch.Tensor,
                           position: torch.Tensor,
                           pot_re_shape: List[int], vox_sz: float, values_shape: List[int]):
        # project_matrix: x - y - z
        # pot_re.shape: (nz, ny, nx)
        for axis, basis in enumerate([x_basis, y_basis]):
            a = torch.abs(torch.matmul(project_matrix[:3, :2], basis))
            a *= torch.tensor(pot_re_shape)
            a *= 0.5 * vox_sz
            a = a.sum()
            # Area of parallelogram extended by basis and position of particle
            pos = position[:2] - offset
            b = torch.abs(basis[1] * pos[0] - basis[0] * pos[1])
            c = 0.5 * values_shape[axis] * torch.abs(
                x_basis[0] * y_basis[1] - x_basis[1] * y_basis[0])
            if b > a + c:
                return False
        return True

    # Can change to check the coordinates instead of check area
    def hits_wavefunction(self, wf, project_matrix: torch.Tensor, position: torch.Tensor) -> bool:
        """
        Checks whether particle hits a wave function

        :param wf: WaveFunction instance
        :param project_matrix: projection matrix
        :param position: position
        :return: bool
        """
        return self._hits_wavefunction(wf.phase.x_basis, wf.phase.y_basis, wf.phase.offset, project_matrix,
                                       position,
                                       list(self.pot_re.shape), self.voxel_size, list(wf.phase.values.shape))

    @staticmethod
    # @torch.jit.script  # comment this line to debug
    def _calc_pf_values(pf_values: torch.Tensor,
                        pd: torch.Tensor,
                        w: torch.Tensor,
                        ds0: torch.Tensor,
                        ds1: torch.Tensor,
                        wave_num: torch.Tensor,
                        project_matrix: torch.Tensor,
                        lap_pot_re: torch.Tensor,
                        lap_pot_im: torch.Tensor,
                        n: List[int],
                        dim: List[int],
                        voxel_size: float,
                        famp: float,
                        use_imag_pot: bool,
                        use_defocus_corr: bool):
        s0 = 0.5 * (pf_values.shape[1] - n[0]) - 0.5 * (n[2] - 1) * ds0
        s1 = 0.5 * (pf_values.shape[0] - n[1]) - 0.5 * (n[2] - 1) * ds1

        s0s = torch.arange(n[2]) * ds0 + s0
        j0s = torch.floor(s0s).long()  # (n[2], )
        s1s = torch.arange(n[2]) * ds1 + s1
        j1s = torch.floor(s1s).long()  # (n[2], )
        t0s = s0s - j0s  # (n[2], )
        t1s = s1s - j1s  # (n[2], )
        w0s = w * (1. - t0s) * (1. - t1s)  # (n[2], )
        w1s = w * t0s * (1. - t1s)  # (n[2], )
        w2s = w * (1. - t0s) * t1s  # (n[2], )
        w3s = w * t0s * t1s  # (n[2], )

        pd = pd.permute(2, 1, 0)  # (n[0], n[1], n[2]) -> (n[2], n[1], n[0])
        if use_defocus_corr:
            dim_vec = torch.tensor(dim, dtype=torch.long)
            dz = -(0.5 * voxel_size / wave_num) * -project_matrix[dim_vec, 2]
            k0s = torch.arange(n[0]).view(1, 1, -1)  # (1, 1, n[0])
            k1s = torch.arange(n[1]).view(1, -1, 1)  # (1, n[1], 1)
            k2s = torch.arange(n[2]).view(-1, 1, 1)  # (n[2], 1, 1)
            zs = -0.5 * (n[0] - 1) * dz[0] \
                 + (k1s - 0.5 * (n[1] - 1)) * dz[1] \
                 + (k2s - 0.5 * (n[2] - 1)) * dz[2] \
                 + k0s * dz[0]  # (n[2], n[1], n[0])
            zs = torch.complex(torch.zeros_like(zs), zs)  # (n[2], n[1], n[0]), complex
            if use_imag_pot:
                pl = torch.view_as_complex(torch.stack([lap_pot_re, lap_pot_im], dim=-1)) \
                    .permute(2, 1, 0)  # (n[2], n[1], n[0])
            else:
                pl = torch.view_as_complex(torch.stack([lap_pot_re, torch.zeros_like(lap_pot_re)], dim=-1)) \
                    .permute(2, 1, 0)  # (n[2], n[1], n[0])
            for k2 in range(0, n[2]):
                j0 = j0s[k2]
                j1 = j1s[k2]
                pdz = pd[k2] + zs[k2] * pl[k2]  # (n[1], n[0])
                pf_values[j1: j1 + n[1], j0: j0 + n[0]] += w0s[k2] * pdz
                pf_values[j1: j1 + n[1], j0 + 1: j0 + n[0] + 1] += w1s[k2] * pdz
                pf_values[j1 + 1: j1 + n[1] + 1, j0: j0 + n[0]] += w2s[k2] * pdz
                pf_values[j1 + 1: j1 + n[1] + 1, j0 + 1: j0 + n[0] + 1] += w3s[k2] * pdz
        else:
            for k2 in range(0, n[2]):
                j0 = j0s[k2]
                j1 = j1s[k2]
                pd_k2 = pd[k2]
                pf_values[j1: j1 + n[1], j0: j0 + n[0]] += w0s[k2] * pd_k2
                pf_values[j1: j1 + n[1], j0 + 1: j0 + n[0] + 1] += w1s[k2] * pd_k2
                pf_values[j1 + 1: j1 + n[1] + 1, j0: j0 + n[0]] += w2s[k2] * pd_k2
                pf_values[j1 + 1: j1 + n[1] + 1, j0 + 1: j0 + n[0] + 1] += w3s[k2] * pd_k2

        if not use_imag_pot:
            fph = math.sqrt(1 - famp * famp)
            pf_values = pf_values * torch.tensor(complex(fph, famp))
        return pf_values

    def project(self, project_matrix: torch.Tensor, position: torch.Tensor,
                potential_conv_factor: torch.Tensor,
                wave_num: torch.Tensor) -> VectorValuedFunction:
        if torch.abs(project_matrix[2, 2]) > torch.abs(project_matrix[0, 2]):  # z project > x project
            # z project > y project
            if torch.abs(project_matrix[2, 2]) > torch.abs(project_matrix[1, 2]):  # Projection line closest to z axis
                dim = [0, 1, 2]
            # z project <= y project
            else:  # Projection line closest to y axis
                dim = [0, 2, 1]
        else:  # z project <= x project
            if torch.abs(project_matrix[1, 2]) > torch.abs(project_matrix[0, 2]):  # y project > x project
                # Projection line closest to y axis
                dim = [0, 2, 1]
            else:  # y project <= x project,  projection line closest to x axis
                dim = [1, 2, 0]  # far - middle - near

        if self.use_imag_pot:
            pd = torch.view_as_complex(torch.stack([self.pot_re, self.pot_im], dim=-1))
        else:
            pd = self.pot_re

        # dim: far - middle - near and follow xyz order
        pd = pd.permute(2, 1, 0)
        pd = pd.permute(dim[0], dim[1], dim[2])
        n = pd.shape

        ds0 = -project_matrix[dim[0], 2] / project_matrix[dim[2], 2]
        ds1 = -project_matrix[dim[1], 2] / project_matrix[dim[2], 2]

        voxel_size = self.voxel_size
        w = voxel_size * torch.sqrt(1 + ds0 * ds0 + ds1 * ds1) * potential_conv_factor

        # pf_values: pixel_y, pixel_x, complex tensor
        pf_values = torch.zeros(torch.ceil(n[1] + n[2] * torch.abs(ds1) + 1).long(),
                                torch.ceil(n[0] + n[2] * torch.abs(ds0) + 1).long(), dtype=torch.cdouble)

        pf_basis_x = torch.stack([project_matrix[dim[0], 0], project_matrix[dim[0], 1]]) * voxel_size
        pf_basis_y = torch.stack([project_matrix[dim[1], 0], project_matrix[dim[1], 1]]) * voxel_size
        if self.use_defocus_corr:
            plr = self.lap_pot_re.permute(2, 1, 0)
            plr = plr.permute(dim[0], dim[1], dim[2])

            pli = self.lap_pot_im.permute(2, 1, 0)
            pli = pli.permute(dim[0], dim[1], dim[2])

            pf_values = self._calc_pf_values(pf_values, pd, w, ds0, ds1, wave_num, project_matrix,
                                             plr, pli,
                                             n, dim, voxel_size,
                                             self.famp, self.use_imag_pot, self.use_defocus_corr)
            _write_log(f'pf_values defocus_corr Grad: {pf_values.grad_fn}')
        else:
            placeholder = torch.zeros(1)
            pf_values = self._calc_pf_values(pf_values, pd, w, ds0, ds1, wave_num, project_matrix,
                                             placeholder, placeholder,
                                             n, dim, voxel_size,
                                             self.famp, self.use_imag_pot, self.use_defocus_corr)
            _write_log(f'pf_values Grad: {pf_values.grad_fn}')
        pf = VectorValuedFunction(pf_values, pf_basis_x, pf_basis_y, position.clone()[:2])

        return pf

    def _write_map(self, map_tensor: torch.Tensor, filepath: str):
        voxel_size = self.voxel_size
        map_tensor = change_axes(map_tensor, self.map_axis_order)
        format = self.map_file_format
        if format == BinaryFileFormat.RAW:
            write_tensor_to_binary(map_tensor, filepath, endian=self.map_file_byte_order)
        elif format == BinaryFileFormat.MRC:
            write_tensor_to_mrc(map_tensor, filepath, voxel_size, as_type=torch.float32)
        else:  # mrc-int
            map_tensor = map_tensor * (POTENTIAL_UNIT / INT_FILE_POT_UNIT)
            write_tensor_to_mrc(map_tensor, filepath, voxel_size, as_type=torch.int16)
        _write_log(f"Particle map written to file {filepath}.")

    @staticmethod
    def _gen_random_particle(nx: int, ny: int, nz: int, contrast: float, smoothness: float, positive: bool):
        num_elements = nx * ny * nz
        b0 = 1. + 2. / (1. + smoothness)
        b1 = 0.1
        n = [nx // 2 + 1, ny, nz]
        dxi = (nx + ny + nz) / (3.0 * nx)
        dxj = (nx + ny + nz) / (3.0 * ny)
        dxk = (nx + ny + nz) / (3.0 * nz)
        xks = torch.arange(n[2])
        xks = torch.minimum(xks, n[2] - xks)
        xks *= dxk
        xks = xks.reshape(1, 1, n[2])
        xjs = torch.arange(n[1])
        xjs = torch.minimum(xjs, n[1] - xjs)
        xjs *= dxj
        xjs = xjs.reshape(1, n[1], 1)
        xis = torch.arange(n[0]) * dxi
        xis = xis.reshape(n[0], 1, 1)
        s = torch.pow(xis * xis + xjs * xjs + xks * xks + b0, -1. - smoothness)  # (n[0], n[1], n[2])
        x1 = torch.randn_like(s) * s
        x2 = torch.randn_like(s) * s
        transform = torch.stack([x1, x2], dim=-1)  # (n[0], n[1], n[2], 2)
        transform = torch.view_as_complex(transform)  # (n[0], n[1], n[2]), complex

        transform_ = torch.fft.rfftn(transform.real, norm="backward")
        # TODO: Testing
        xk = torch.arange(n[2])  # (n[2],)
        xk = torch.minimum(xk, n[2] - xk) * dxk
        xj = torch.arange(n[1])  # (n[1],)
        xj = torch.minimum(xj, n[1] - xj) * dxj
        xi = torch.arange(n[0])  # (n[0])
        xi = torch.minimum(xi, n[0] - xi) * dxi

        xk = xk.view(1, 1, -1)  # (1, 1, n[2])
        xj = xj.view(1, -1, 1)  # (1, n[1], 1)
        xi = xi.view(-1, 1, 1)  # (n[0], 1, 1)
        s = torch.pow(xi ** 2 + xj ** 2 + xk ** 2 + b0, -1. - smoothness)  # (n[0], n[1], n[2])
        transform_.real = torch.randn_like(s) * s
        transform_.imag = torch.randn_like(s) * s
        if positive:
            transform_[0, 0, 0] = math.pow(b0, -1. - smoothness)
        transform_ = torch.fft.irfftn(transform_, norm="forward")
        matrix = transform_.real
        c = b1 * math.sqrt((matrix ** 2).sum() / num_elements)
        c2 = c * c
        matrix = torch.sqrt(matrix ** 2 + c2) - c
        ic = 0.5 * (nx - 1)
        jc = 0.5 * (ny - 1)
        kc = 0.5 * (nz - 1)
        xks = torch.arange(nz) / kc - 1
        xks = xks.reshape(1, 1, nz)
        xjs = torch.arange(ny) / jc - 1
        xjs = xjs.reshape(1, ny, 1)
        xis = torch.arange(nx) / ic - 1
        xis = xis.reshape(nx, 1, 1)
        r = xis * xis + xjs * xjs + xks * xks
        r_le_1 = r < 1
        matrix *= r_le_1.double() * (1.0 + torch.cos(r * r * PI))
        s = torch.abs(matrix).sum()
        matrix *= contrast * num_elements * PI / (6.0 * s)
        return matrix

    def _unwrap_required_parameter(self, parameter: Parameters) -> Any:
        parameter_value = self._parameters[parameter]
        if parameter_value is None:
            raise RequiredParameterNotSpecifiedException(parameter)
        return parameter_value

    @property
    def device(self):
        assert self._initialized, "Initialize particle first"
        return self.pot_re.device

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def source(self) -> str:
        """
        Defines what kind of source of information is used to obtain a potential map.
        Possible sources of the potential map are data in a PDB file,
        reading a map directly from an MRC file or a raw data file, or generating a random map.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.SOURCE)

    @property
    def use_imag_pot(self) -> bool:
        """
        Controls whether a separate map of an \"imaginary\" absorption potential
        should be used in the simulation along with the real valued electrostatic potential.
        If a separate map is not used, the imaginary potential is taken to be a multiple of the real potential, given by the parameter famp.
        When using a PDB file as source, \"use_imag_pot\" should be set to yes in order to make the simulation as correct as possible.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.USE_IMAG_POT)

    @property
    def famp(self) -> float:
        """
        If use_imag_pot is set to no, the same map is used to define both the real and the imaginary potential.
        The imaginary potential is taken as the map multiplied by famp, and the real potential as the map multiplied by sqrt(1-famp^2).

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.FAMP)

    @property
    def use_defocus_corr(self) -> bool:
        """
        A thick sample is always split into a number of slices to account for the varying defocus within the sample.
        This parameter tells the simulator to also correct for the varying defocus within each slice
        when simulating the interaction with this particle type.
        The computational work increases and the effect on the end result is usually negligible.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.USE_DEFOCUS_CORR)

    @property
    def voxel_size(self) -> float:
        """
        The voxel size in nm of the map of the particle potential used in the simulations.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.VOXEL_SIZE)

    @property
    def nx(self) -> int:
        """
        The number of voxels along the x-axis in the map of the particle potential.
        The size of the particle map must be specified if the map is to be read from a raw data file (without header) or randomly generated.
        It can optionally be specified if the map is generated from a PDB file.
        If the map is read from an MRC file the size is determined by the file header.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.NX)

    @property
    def ny(self) -> int:
        """
        The number of voxels along the y-axis in the map of the particle potential.
        The size of the particle map must be specified if the map is to be read from a raw data file (without header) or randomly generated.
        It can optionally be specified if the map is generated from a PDB file.
        If the map is read from an MRC file the size is determined by the file header.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.NY)

    @property
    def nz(self) -> int:
        """
        The number of voxels along the z-axis in the map of the particle potential.
        The size of the particle map must be specified if the map is to be read from a raw data file (without header) or randomly generated.
        It can optionally be specified if the map is generated from a PDB file.
        If the map is read from an MRC file the size is determined by the file header.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.NZ)

    @property
    def map_file_re_in(self) -> str:
        """
        The name of a file from which the potential map is read if source is set to \"map\".

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.MAP_FILE_RE_IN)

    @property
    def map_file_im_in(self) -> str:
        """
        The name of a file from which the imaginary absorption potential map is read if source is set to \"map\" and use_imag_pot is set to yes.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.MAP_FILE_IM_IN)

    @property
    def map_file_re_out_(self) -> str:
        """
        The name of a file to which the potential map is written if \"source\" is set to \"pdb\" or \"random\".
        If the parameter is not given, the potential map is not written to file.
        Unchecked accessor

        :return: None or file path string
        """
        return self._parameters[Particle.Parameters.MAP_FILE_RE_OUT]

    @property
    def map_file_re_out(self) -> str:
        """
        The name of a file to which the potential map is written if \"source\" is set to \"pdb\" or \"random\".
        If the parameter is not given, the potential map is not written to file.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.MAP_FILE_RE_OUT)

    @property
    def map_file_im_out_(self) -> str:
        """
        The name of a file to which the imaginary absorption potential map is written
        if source is set to \"pdb\" or \"random\" and use_imag_pot is set to yes.
        If the parameter is not given, the imaginary potential map is not written to file.
        Unchecked accessor

        :return: None or file path string
        """
        return self._parameters[Particle.Parameters.MAP_FILE_IM_OUT]

    @property
    def map_file_im_out(self) -> str:
        """
        The name of a file to which the imaginary absorption potential map is written
        if source is set to \"pdb\" or \"random\" and use_imag_pot is set to yes.
        If the parameter is not given, the imaginary potential map is not written to file.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.MAP_FILE_IM_OUT)

    @property
    def map_file_format(self) -> str:
        """
        The format of files used for input and output of potential maps.
        \"mrc\" means MRC file with 4-byte floating point data.
        \"mrc-int\" means MRC file with 2-byte integer data.
        \"raw\" is the same format as \"mrc\" but without the file header.
        For input files, \"mrc\" and \"mrc-int\" are equivalent, the type of MRC file is determined by the file header.
        The potential map is assumed to be in V for floating point file formats, and mV for integer file formats.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.MAP_FILE_FORMAT)

    @property
    def map_file_byte_order(self) -> ByteOrder:
        """
        Controls if input and output binary files are in little endian or big endian format.
        For MRC input files the parameter is ignored, instead the format is determined by the file header.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.MAP_FILE_BYTE_ORDER)

    @property
    def map_axis_order(self) -> str:
        """
        Controls the order in which voxel values appear in input and output map files.
        The first letter is the fastest varying index, and the last letter is the slowest varying.
        \"xyz\" means that the simulator's use of x, y, and z coordinates is consistent with the MRC file convention.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.MAP_AXIS_ORDER)

    @property
    def pdb_file_in(self) -> str:
        """
        The name of a PDB file used to generate a potential map if source is set to \"pdb\".

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.PDB_FILE_IN)

    @property
    def pdb_transf_file_in(self) -> str:
        """
        The name of a text file which defines a number of geometric transformations applied to the atomic coordinates in the PDB file.
        This can be used to define particles built up of several identical subunits.
        If the parameter is not given, the atomic coordinates are used just as they appear in the PDB file without any transformation applied.
        See the manual for details on the file format.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.PDB_TRANSF_FILE_IN)

    @property
    def add_hydrogen(self) -> bool:
        """
        Tells the simulator to add hydrogen atoms, which are usually not specified in PDB files,
        at the appropriate locations in the amino acid residuals.
        The hydrogen atoms are added at the coordinates of the atom to which they are bonded.
        This is of course not really correct, but it is a good enough approximation for many purposes.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.ADD_HYDROGEN)

    @property
    def contrast_re(self) -> float:
        """
        The overall contrast level of a randomly generated potential map.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.CONTRAST_RE)

    @property
    def contrast_im(self) -> float:
        """
        The overall contrast level of a randomly generated imaginary absorption potential map.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.CONTRAST_IM)

    @property
    def smoothness(self) -> float:
        """
        The smoothness of randomly generated maps.
        Higher values means smoother maps.
        The numerical value does not have any obvious physical interpretation.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.SMOOTHNESS)

    @property
    def make_positive(self) -> bool:
        """
        Tells the simulator to make random maps always have values above the background level,
        rather than randomly going above and below the background level.
        (Values always below the background value can be accomplished by setting \"make_positive\" to \"yes\" and making \"contrast_re\"
        and \"contrast_im\" negative.)

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particle.Parameters.MAKE_POSITIVE)

    def reset(self):
        """
        Reset values in Particle
        :return: self
        """
        if self._initialized:
            self.pot_re = UNINITIALIZED
            self.pot_im = UNINITIALIZED
            self.lap_pot_im = UNINITIALIZED
            self.lap_pot_re = UNINITIALIZED
            self._lock_parameters = False
            self._initialized = False
        return self

    def _set_parameter(self, parameter: Parameters, value):
        if isinstance(value, parameter.type()):
            if self._lock_parameters:
                raise ParameterLockedException(parameter)
            else:
                self._parameters[parameter] = value
        else:
            raise WrongArgumentTypeException(value, parameter.type())

    @source.setter
    def source(self, source: str):
        if source in self.SOURCE_OPTIONS:
            self._set_parameter(self.Parameters.SOURCE, source)
        else:
            raise Exception(f"Invalid source = {source}, "
                            f"valid options = {self.SOURCE_OPTIONS}")

    @use_imag_pot.setter
    def use_imag_pot(self, use_imag_pot: bool):
        self._set_parameter(self.Parameters.USE_IMAG_POT, use_imag_pot)

    @famp.setter
    def famp(self, famp: float):
        self._set_parameter(self.Parameters.FAMP, famp)

    @use_defocus_corr.setter
    def use_defocus_corr(self, use_defocus_corr: bool):
        self._set_parameter(self.Parameters.USE_DEFOCUS_CORR, use_defocus_corr)

    @voxel_size.setter
    def voxel_size(self, voxel_size: float):
        self._set_parameter(self.Parameters.VOXEL_SIZE, voxel_size)

    @nx.setter
    def nx(self, nx: int):
        self._set_parameter(self.Parameters.NX, nx)
        assert nx > 0

    @ny.setter
    def ny(self, ny: int):
        self._set_parameter(self.Parameters.NY, ny)
        assert ny > 0

    @nz.setter
    def nz(self, nz: int):
        self._set_parameter(self.Parameters.NZ, nz)
        assert nz > 0

    @map_file_re_in.setter
    def map_file_re_in(self, map_file_re_in: str):
        self._set_parameter(self.Parameters.MAP_FILE_RE_IN, map_file_re_in)

    @map_file_im_in.setter
    def map_file_im_in(self, map_file_im_in: str):
        self._set_parameter(self.Parameters.MAP_FILE_IM_IN, map_file_im_in)

    @map_file_re_out.setter
    def map_file_re_out(self, map_file_re_out: str):
        self._set_parameter(self.Parameters.MAP_FILE_RE_OUT, map_file_re_out)

    @map_file_im_out.setter
    def map_file_im_out(self, map_file_im_out: str):
        self._set_parameter(self.Parameters.MAP_FILE_IM_OUT, map_file_im_out)

    @map_file_format.setter
    def map_file_format(self, map_file_format: str):
        try:
            self._set_parameter(self.Parameters.MAP_FILE_FORMAT, BinaryFileFormat(map_file_format))
        except ValueError:
            raise Exception(f"Invalid map_file_format = {map_file_format}, "
                            f"valid options = {BinaryFileFormat.options()}")

    @map_file_byte_order.setter
    def map_file_byte_order(self, map_file_byte_order: str):
        try:
            self._set_parameter(self.Parameters.MAP_FILE_BYTE_ORDER, ByteOrder(map_file_byte_order))
        except ValueError:
            raise Exception(f"Invalid map_file_format = {map_file_byte_order}, "
                            f"valid options = {ByteOrder.options()}")

    @map_axis_order.setter
    def map_axis_order(self, map_file_axis_order: str):
        if map_file_axis_order in self.MAP_AXIS_ORDER_OPTIONS:
            self._set_parameter(self.Parameters.MAP_AXIS_ORDER, map_file_axis_order)
        else:
            raise Exception(f"Invalid map_file_format = {map_file_axis_order}, "
                            f"valid options = {self.MAP_AXIS_ORDER_OPTIONS}")

    @pdb_file_in.setter
    def pdb_file_in(self, pdb_file_in: str):
        self._set_parameter(self.Parameters.PDB_FILE_IN, pdb_file_in)

    @pdb_transf_file_in.setter
    def pdb_transf_file_in(self, pdb_transf_file_in: str):
        self._set_parameter(self.Parameters.PDB_TRANSF_FILE_IN, pdb_transf_file_in)

    @add_hydrogen.setter
    def add_hydrogen(self, add_hydrogen: bool):
        self._set_parameter(self.Parameters.ADD_HYDROGEN, add_hydrogen)

    @contrast_re.setter
    def contrast_re(self, contrast_re: float):
        self._set_parameter(self.Parameters.CONTRAST_RE, contrast_re)

    @contrast_im.setter
    def contrast_im(self, contrast_im: float):
        self._set_parameter(self.Parameters.CONTRAST_IM, contrast_im)

    @smoothness.setter
    def smoothness(self, smoothness: float):
        self._set_parameter(self.Parameters.SMOOTHNESS, smoothness)

    @make_positive.setter
    def make_positive(self, make_positive: bool):
        self._set_parameter(self.Parameters.MAKE_POSITIVE, make_positive)

    def log_parameters(self, logger: logging.Logger):
        logger.info(f"Particle:\n   {self._parameters}")
