import math
from copy import deepcopy
from enum import Enum
import logging
from typing import Any, Optional, Union, List, Dict

import torch
from . import UNINITIALIZED
from .constants.io_units import POTENTIAL_UNIT, INT_FILE_POT_UNIT
from .particle import Particle
from .utils.exceptions import ParameterLockedException, WrongArgumentTypeException, \
    RequiredParameterNotSpecifiedException
from .utils.input_processing import convert_bool
from .utils.parameter import Parameter
from .utils.helper_functions import check_tensor
from .utils.tensor_io import write_tensor_to_binary, ByteOrder
from .utils.tensor_ops import tensor_indexing
from .utils.mrc import write_tensor_to_mrc
from .utils.tensor_ops import change_axes
from .utils import BinaryFileFormat

_logger = logging.getLogger(__name__)


def _write_log(string, *logging_strings):
    strings = [string, *logging_strings]
    if len(strings) > 1:
        logging_string = "\n    ".join(strings)
        _logger.debug(f"{_logger.name}:\n    {logging_string}", stacklevel=2)
    else:
        _logger.debug(f"{_logger.name}:  {strings[0]}", stacklevel=2)


class Volume:
    """
    The volume component plays no role in the actual simulation of micrographs.
    It is used to export to a file a map of the entire sample or a sub-volume.
    This map can be used as a reference, for example when evaluating reconstructions made from the tilt series.
    """

    class Parameters(Enum):
        VOXEL_SIZE = Parameter("voxel_size", float, description="The size of voxels in the volume map, measured in nm.")
        N = Parameter("nx, ny, nz", int, description="The number of voxels along the x/y/z axis of the volume map.")
        OFFSET = Parameter("offset_x, offset_y, offset_z", float,
                           description="The x/y/z coordinate of the center of the volume map, measured in nm.")
        MAP_FILE_RE_OUT = Parameter("map_file_re_out", str,
                                    description="The name of a file "
                                                "to which the real part of the volume map is written.")
        MAP_FILE_IM_OUT = Parameter("map_file_im_out", str,
                                    description="The name of a file "
                                                "to which the imaginary part of the volume map is written. "
                                                "If not defined, the imaginary part is not written to file.")
        MAP_FILE_FORMAT = Parameter("map_file_format", BinaryFileFormat,
                                    description="The format of exported map files. "
                                                "\"mrc\" means MRC file with 4-byte floating point data, "
                                                "\"mrc-int\" means MRC file with 2-byte integer data. "
                                                "\"raw\" is the same format as \"mrc\" but without the file header.")
        MAP_FILE_BYTE_ORDER = Parameter("map_file_byte_order", ByteOrder,
                                        description="Controls if exported map files are "
                                                    "in little endian or big endian format.")
        MAP_AXIS_ORDER = Parameter("map_axis_order", str,
                                   description="Controls the order in which voxel values appear in exported map files. "
                                               "The first letter is the fastest varying index, "
                                               "and the last letter is the slowest varying. "
                                               "\"xyz\" means that the simulator's use of x, y, and z coordinates "
                                               "is consistent with the MRC file convention.")

        def type(self):
            return self.value.type

        @staticmethod
        def parameter_names():
            return [p.value.name for p in Volume.Parameters]

    REQUIRED_PARAMETERS = [Parameters.VOXEL_SIZE, Parameters.N, Parameters.MAP_FILE_RE_OUT]

    MAP_AXIS_ORDER_OPTIONS = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]

    PARAMETER_DEFAULTS = {
        Parameters.OFFSET: torch.zeros(3),
        Parameters.MAP_FILE_FORMAT: BinaryFileFormat.MRC,
        Parameters.MAP_FILE_BYTE_ORDER: ByteOrder.NATIVE,
        Parameters.MAP_AXIS_ORDER: MAP_AXIS_ORDER_OPTIONS[0]
    }

    def __init__(self):
        self._pot_re: torch.Tensor = UNINITIALIZED
        self._pot_im: torch.Tensor = UNINITIALIZED
        parameters = {}
        for p in self.Parameters:
            if p in self.PARAMETER_DEFAULTS:
                parameters[p] = self.PARAMETER_DEFAULTS[p]
            else:
                parameters[p] = UNINITIALIZED
        self._parameters = parameters
        self._initialized = False
        self._lock_parameters = False

    def to(self, device: torch.device):
        """
        Out-of-place move to a device, creating new copy of self whose data is on the device

        :param device: device that holds data
        :return: a new copy of self
        """
        assert self._initialized, "Initialize volume first"
        volume = Volume()
        volume._pot_re = self._pot_re.to(device)
        volume._pot_im = self._pot_im.to(device)
        volume._parameters = deepcopy(self._parameters)
        volume.offset = self.offset.to(device)
        volume.n = self.n.to(device)
        volume._initialized = self._initialized
        volume._lock_parameters = self._lock_parameters
        return volume

    def to_(self, device: torch.device):
        """
        In-place move to a device

        :param device: device that holds data
        """
        assert self._initialized, "Initialize volume first"
        self._lock_parameters = False
        self._pot_re = self._pot_re.to(device)
        self._pot_im = self._pot_im.to(device)
        self.offset = self.offset.to(device)
        self.n = self.n.to(device)
        self._lock_parameters = True

    @staticmethod
    def construct_from_parameters(parameters: Union[List[Dict[str, str]], Dict[str, str]],
                                  initialize: Optional[bool] = True):
        """
        Construct instances of Volume given parameters

        :param parameters: list of parameter dictionaries or a dictionary containing parameters
        :param initialize: whether to initialize instances, default: True
        :return: list of instances or one instance
        """
        if isinstance(parameters, list):
            volumes = [Volume._from_parameters(parameter_table) for parameter_table in parameters]
            if initialize:
                for v in volumes:
                    v.finish_init_()
            return volumes
        elif isinstance(parameters, dict):
            volume = Volume._from_parameters(parameters)
            if initialize:
                volume.finish_init_()
            return volume
        else:
            raise Exception(f"Invalid parameter of type {type(parameters)}")

    @staticmethod
    def _from_parameters(parameters: Dict[str, str]):
        volume = Volume()
        if "offset_x" in parameters and "offset_y" in parameters and "offset_z" in parameters:
            offset_x = float(parameters["offset_x"])
            offset_y = float(parameters["offset_y"])
            offset_z = float(parameters["offset_z"])
            volume.offset = torch.tensor([offset_x, offset_y, offset_z])
        if "nx" in parameters and "ny" in parameters and "nz" in parameters:
            nx = int(parameters["nx"])
            ny = int(parameters["ny"])
            nz = int(parameters["nz"])
            volume.n = torch.tensor([nx, ny, nz])
        for p in Volume.Parameters:
            if p != Volume.Parameters.OFFSET and p != Volume.Parameters.N:
                parameter_name = p.value.name
                parameter_type = p.type()
                if parameter_type == bool:
                    parameter_type = convert_bool
                if parameter_name in parameters:
                    parameter_val = parameter_type(parameters[parameter_name])
                    setattr(volume, parameter_name, parameter_val)
        return volume

    def finish_init_(self):
        """
        Finish initialization of Volume

        :return: self
        """
        if self._initialized:
            return self
        for parameter_type in self.REQUIRED_PARAMETERS:
            if self._parameters[parameter_type] is None:
                raise RequiredParameterNotSpecifiedException(parameter_type)
        self._pot_re = torch.zeros(*self.n)
        if self._parameters[self.Parameters.MAP_FILE_IM_OUT] is not None:
            self._pot_im = torch.zeros_like(self._pot_re)
        self._initialized = True
        self._lock_parameters = True
        _write_log("Volume component initialized.")
        return self

    def reset(self):
        """
        Reset Volume
        """
        if not self._initialized:
            return
        self._pot_re = None
        self._pot_im = None
        self._initialized = False
        self._lock_parameters = False

    def _write_map(self, map_tensor, file_path):
        voxel_size = self.voxel_size
        map_tensor = change_axes(map_tensor, self.map_axis_order)
        format = self.map_file_format
        if format == BinaryFileFormat.RAW:
            write_tensor_to_binary(map_tensor, file_path, endian=self.map_file_byte_order)
        elif format == BinaryFileFormat.MRC:
            write_tensor_to_mrc(map_tensor, file_path, voxel_size, as_type=torch.float32)
        else:  # mrc int
            map_tensor = map_tensor * (POTENTIAL_UNIT / INT_FILE_POT_UNIT)
            write_tensor_to_mrc(map_tensor, file_path, voxel_size, as_type=torch.int16)
        _write_log(f"Volume map written to file {file_path}")

    def write_maps(self):
        if not self._initialized:
            raise Exception("Volume not initialized")
        self._write_map(self._pot_re, self.map_file_re_out)
        if self._parameters[self.Parameters.MAP_FILE_IM_OUT] is not None:
            self._write_map(self._pot_im, self._parameters[self.Parameters.MAP_FILE_IM_OUT])

    def calculate_potential(self, sim):
        self.finish_init_()
        for particle_set in sim.particle_sets:
            particle_set.finish_init_(sim)
        self._pot_re.fill_(1e-10)
        if self._parameters[self.Parameters.MAP_FILE_IM_OUT] is not None:
            self._pot_im.fill_(1e-10)
        for particle_set in sim.particle_sets:
            particle_type = particle_set.particle_type
            particle = sim.get_particle(particle_type)
            if particle is not None:
                particle.finish_init_(sim)
            else:
                raise Exception(f"Particle {particle_type} not found.\n")
            for i in range(particle_set.coordinates.shape[0]):
                particle_rotate_matrix, particle_pos = particle_set.get_particle_coord(i)
                self._add_particle(particle_rotate_matrix, particle_pos, particle)
        self._add_background(sim)

    def _add_particle(self, rotate_matrix: torch.Tensor, particle_pos: torch.Tensor, particle: Particle):
        assert self._initialized and particle.initialized
        device = rotate_matrix.device
        check_tensor(rotate_matrix, shape=(3, 3))
        check_tensor(particle_pos, dimensionality=1, shape=(3,), device=device)
        ic = 0.5 * (torch.tensor(list(particle.pot_re.shape)) - 1)
        x = -self.offset.clone()
        voxel_size_v = self.voxel_size
        try:
            use_imag_pot_v = self.map_file_im_out is not None
        except RequiredParameterNotSpecifiedException:
            use_imag_pot_v = False
        d0 = x / voxel_size_v
        x += particle_pos
        r = torch.sum(ic.reshape(1, 3) * torch.abs(rotate_matrix), dim=1)
        y = x + r
        x -= r
        v_pot_re_shape_vec = torch.tensor(list(self._pot_re.shape), device=device)
        imin = torch.maximum(-torch.ones(1),
                             torch.floor(x / voxel_size_v + 0.5 * (v_pot_re_shape_vec - 1))).int()
        imax = torch.minimum(1 + v_pot_re_shape_vec,
                             torch.ceil(y / voxel_size_v + 0.5 * (v_pot_re_shape_vec - 1))).int()
        if torch.any(imin >= imax):
            return
        d0 += 0.5 * (v_pot_re_shape_vec - 1) - imin
        pot_shape = imax - imin
        pot_re = torch.zeros(*pot_shape, device=device)
        weight = torch.zeros(*pot_shape, device=device)
        if use_imag_pot_v:
            pot_im = torch.zeros(*pot_shape, device=device)
        if particle.use_imag_pot:
            famp = 1.0
            fph = 1.0
        else:
            famp = particle.famp
            fph = math.sqrt(1.0 - famp ** 2)

        voxel_size_p = particle.voxel_size
        particle_pot_re_shape = particle.pot_re.shape
        i = torch.arange(particle_pot_re_shape[0]).view(-1, 1, 1).repeat(1, particle_pot_re_shape[1],
                                                                         particle_pot_re_shape[2])
        j = torch.arange(particle_pot_re_shape[1]).view(1, -1, 1).repeat(particle_pot_re_shape[0], 1,
                                                                         particle_pot_re_shape[2])
        k = torch.arange(particle_pot_re_shape[2]).view(1, 1, -1).repeat(particle_pot_re_shape[0],
                                                                         particle_pot_re_shape[1], 1)
        ijk = torch.stack([i, j, k], dim=-1)  # shape(*particle_pot_re_shape, 3)
        y = voxel_size_p * (ijk - ic.reshape(1, 1, 1, 3))  # shape(*particle_pot_re_shape, 3)
        x = particle_pos.reshape(1, 1, 1, 3).repeat(y.shape[0], y.shape[1], y.shape[2],
                                                    1)  # shape(*particle_pot_re_shape, 3)
        x += torch.squeeze(rotate_matrix @ y.unsqueeze(-1), dim=-1)  # essentially rotate_matrix @ y
        x /= voxel_size_v
        x += d0.reshape(1, 1, 1, 3)
        d = torch.floor(x)  # shape(*particle_pot_re_shape, 3)
        w_1 = x - d
        w_0 = 1.0 - w_1
        w = torch.stack([w_0, w_1], dim=-1)  # shape(*particle_pot_re_shape, 3, 2)
        pot_shape = pot_shape.reshape(1, 1, 1, 3)
        in_range_mask = ((d >= 0) & (d < pot_shape)) & ((d + 1 >= 0) & (d + 1 < pot_shape))
        in_range_mask_indices = torch.where(in_range_mask)
        valid_d = d[in_range_mask_indices]  # shape(number of valid d, 3)
        pdr = pot_re[in_range_mask_indices]  # shape (number of valid d, )
        if use_imag_pot_v:
            pdi = pot_im[in_range_mask_indices]  # shape (number of valid d, )
        else:
            pdi = pot_re[in_range_mask_indices]

        index_offsets = torch.tensor([[0, 0, 0],
                                      [0, 0, 1],
                                      [0, 1, 0],
                                      [0, 1, 1],
                                      [1, 0, 0],
                                      [1, 0, 1],
                                      [1, 1, 0],
                                      [1, 1, 1]], dtype=torch.int, device=device)
        w_indices = list(zip(torch.tensor([0, 1, 2], device=device, dtype=torch.long).reshape(1, 3).repeat(8, 1),
                             index_offsets.long()))
        # todo: double check below lines, not emergent
        re = fph * pdr  # shape (number of valid d, )
        if use_imag_pot_v:
            im = famp * pdi  # shape (number of valid d, )
            for w_index, index_offset in zip(w_indices, index_offsets):  # accessing neighbors
                wp = w[w_index]
                wp = wp[0] * wp[1] * wp[2]
                index_offset = index_offset.view(1, 3)
                valid_d_with_offset = index_offset + valid_d  # shape(number of valid d, 3)
                indices_with_offset = tensor_indexing(pot_re, valid_d_with_offset, 3, only_indices=True)
                pot_re[indices_with_offset] += wp * re  # vdr[c] is pot_re[indices_with_offset]
                pot_im[indices_with_offset] += wp * im  # vdi[c] is pot_im[indices_with_offset]
                weight[indices_with_offset] += wp  # wd[c] is weight[indices_with_offset]
        else:
            for w_index, index_offset in zip(w_indices, index_offsets):  # accessing neighbors
                wp = w[w_index]
                wp = wp[0] * wp[1] * wp[2]
                index_offset = index_offset.view(1, 3)
                valid_d_with_offset = index_offset + valid_d  # shape(number of valid d, 3)
                indices_with_offset = tensor_indexing(pot_re, valid_d_with_offset, 3, only_indices=True)
                # vdr[c] is pot_re[indices_with_offset]
                pot_re[indices_with_offset] += wp * re
                # wd[c] is weight[indices_with_offset]
                weight[indices_with_offset] += wp

        weight_gt_1_indices = torch.where(weight > 0)
        weight[weight_gt_1_indices] = 1. / weight[weight_gt_1_indices]
        pot_re *= weight
        # todo: add_array_offset(), not emergent
        if use_imag_pot_v:
            pot_im *= weight

    def _add_background(self, simulation):
        # todo, not emergent
        pass

    def log_parameters(self, logger: logging.Logger):
        logger.info(f"Volume:\n   {self._parameters}")

    def _unwrap_required_parameter(self, parameter: Parameters) -> Any:
        parameter_value = self._parameters[parameter]
        if parameter_value is None:
            raise RequiredParameterNotSpecifiedException(parameter)
        return parameter_value

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def voxel_size(self) -> float:
        """
        The size of voxels in the volume map, measured in nm.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Volume.Parameters.VOXEL_SIZE)

    @property
    def device(self) -> torch.device:
        return self._pot_re.device

    @property
    def n(self) -> torch.Tensor:
        """
        The number of voxels along the x/y/z axis of the volume map.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Volume.Parameters.N)

    @property
    def offset(self) -> torch.Tensor:
        """
        The x/y/z coordinate of the center of the volume map, measured in nm.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Volume.Parameters.OFFSET)

    @property
    def map_file_re_out(self) -> str:
        """
        The name of a file to which the real part of the volume map is written.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Volume.Parameters.MAP_FILE_RE_OUT)

    @property
    def map_file_im_out(self) -> str:
        """
        The name of a file to which the imaginary part of the volume map is written.
        If not defined, the imaginary part is not written to file.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Volume.Parameters.MAP_FILE_IM_OUT)

    @property
    def map_file_format(self) -> BinaryFileFormat:
        """
        The format of exported map files.
        \"mrc\" means MRC file with 4-byte floating point data,
        \"mrc-int\" means MRC file with 2-byte integer data.
        \"raw\" is the same format as \"mrc\" but without the file header.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Volume.Parameters.MAP_FILE_FORMAT)

    @property
    def map_file_byte_order(self) -> ByteOrder:
        """
        Controls if exported map files are in little endian or big endian format.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Volume.Parameters.MAP_FILE_BYTE_ORDER)

    @property
    def map_axis_order(self) -> str:
        """
        Controls the order in which voxel values appear in exported map files.
        The first letter is the fastest varying index, and the last letter is the slowest varying.
        \"xyz\" means that the simulator's use of x, y, and z coordinates is consistent with the MRC file convention.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Volume.Parameters.MAP_AXIS_ORDER)

    def _set_parameter(self, parameter: Parameters, value):
        if isinstance(value, parameter.type()):
            if self._lock_parameters:
                raise ParameterLockedException(parameter)
            else:
                self._parameters[parameter] = value
        else:
            raise WrongArgumentTypeException(value, parameter.type())

    @voxel_size.setter
    def voxel_size(self, voxel_size: float):
        self._set_parameter(self.Parameters.VOXEL_SIZE, voxel_size)
        assert voxel_size > 0.

    @map_file_re_out.setter
    def map_file_re_out(self, filepath: str):
        self._set_parameter(self.Parameters.MAP_FILE_RE_OUT, filepath)

    @map_file_im_out.setter
    def map_file_im_out(self, filepath: str):
        self._set_parameter(self.Parameters.MAP_FILE_IM_OUT, filepath)

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

    @n.setter
    def n(self, n: torch.Tensor):
        check_tensor(n, dimensionality=1, shape=(3,), dtype=torch.int)
        assert torch.all(n > 0)
        self._parameters[self.Parameters.N] = torch.clone(n)

    @offset.setter
    def offset(self, offset: torch.Tensor):
        check_tensor(offset, dimensionality=1, shape=(3,), dtype=torch.double)
        self._parameters[self.Parameters.OFFSET] = torch.clone(offset)
