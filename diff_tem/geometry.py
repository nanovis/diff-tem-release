import logging
import math
from copy import deepcopy
from enum import Enum
from typing import Any, Tuple, Union, List, Dict, Optional

import torch

from . import UNINITIALIZED
from .constants import PI
from .utils.conversion import degree_to_radian, radian_to_degree
from .utils.exceptions import *
from .utils.helper_functions import check_tensor
from .utils.input_processing import convert_bool
from .utils.parameter import Parameter
from .utils.tensor_io import write_matrix_text_with_headers, read_matrix_text
from .utils.transformations import rotate_3d_rows

_logger = logging.getLogger(__name__)


def _write_log(string, *logging_strings):
    strings = [string, *logging_strings]
    if len(strings) > 1:
        logging_string = "\n    ".join(strings)
        _logger.debug(f"{_logger.name}:\n    {logging_string}", stacklevel=2)
    else:
        _logger.debug(f"{_logger.name}:  {strings[0]}", stacklevel=2)


class Geometry:
    """
    The geometry component controls how the sample is moved from image to image in a tilt series.
    The movement can be a simple rotation around a tilt axis,
    but much more general tilt geometries can be accomplished by reading information from a file.
    It is also possible to impose random errors on the tilt geometry.
    A geometry component is required for simulation of micrographs.
    """

    class TiltModes(Enum):
        TILT_SERIES = "tiltseries"
        SINGLE_PARTICLE = "single_particle"

    class Parameters(Enum):
        GEN_TILT_DATA = Parameter("gen_tilt_data", bool,
                                  description="Controls whether the tilt geometry "
                                              "should be automatically generated or read from a file. "
                                              "If the tilt geometry consists of rotations around a fixed tilt axis "
                                              "perpendicular to the optical axis it can be automatically generated "
                                              "by specifying the parameters tilt_axis, "
                                              "theta_start, and theta_incr. "
                                              "For more complicated geometries "
                                              "it is necessary to read input from a text file.")
        N_TILTS = Parameter("ntilts", int,
                            description="The number of images in the tilt series. "
                                        "If ntilts is not specified and the tilt geometry is read from a file, "
                                        "the number of images is instead determined by the number of lines in the file.")
        TILT_AXIS = Parameter("tilt_axis", float,
                              description="The angle in degrees between the x-axis and the tilt axis, "
                                          "if the tilt geometry is automatically generated.")
        THETA_START = Parameter("theta_start", float,
                                description="The tilt angle in degrees of the first image, "
                                            "if the tilt geometry is automatically generated.")
        THETA_INCR = Parameter("theta_incr", float,
                               description="The increment of the tilt angle in degrees between successive images, "
                                           "if the tilt geometry is automatically generated.")
        TILT_MODE = Parameter("tilt_mode", str,
                              description="Controls how the tilt geometry is applied to the sample. "
                                          "\"tiltseries\" means that rotations are applied to the entire sample "
                                          "and the particle positions. "
                                          "\"single_particle\" means that the sample and particle positions stay fixed, "
                                          "while individual particles are rotated around their center.")
        GEOM_FILE_IN = Parameter("geom_file_in", str,
                                 description="The name of a text file from "
                                             "which the tilt geometry is read if gen_tilt_data is set to no.")
        GEOM_FILE_OUT = Parameter("geom_file_out", str,
                                  description="The name of a text file to which the generated tilt geometry is written "
                                              "if gen_tilt_data is set to yes.")
        GEOM_ERRORS = Parameter("geom_errors", str,
                                description="Controls whether geometry errors should be included, "
                                            "and if the errors should be random or read from a file."
                                            "If the errors are random, their magnitude is controlled by the parameters"
                                            "\"tilt_err\", \"align_err_rot\", and \"align_err_tr\".")
        TILT_ERR = Parameter("tilt_err", float,
                             description="If \"geom_errors\" is set to \"random\", "
                                         "the sample is rotated a random angle "
                                         "around a random line perpendicular to the optical axis. "
                                         "\"tilt_err\" is the standard deviation "
                                         "of the rotation angle measured in degrees.")
        ALIGN_ERR_ROT = Parameter("align_err_rot", float,
                                  description="If \"geom_errors\" is set to \"random\", "
                                              "the sample is rotated a random angle around the optical axis. "
                                              "\"align_err_rot\" is the standard deviation "
                                              "of the rotation angle measured in degrees.")
        ALIGN_ERR_TR = Parameter("align_err_tr", float,
                                 description="If \"geom_errors\" is set to \"random\", "
                                             "the sample is translated a random distance "
                                             "perpendicular to the optical axis. "
                                             "\"align_err_tr\" is the standard deviation of "
                                             "the translation measured in nm.")
        ERROR_FILE_IN = Parameter("error_file_in", str,
                                  description="The name of a text file "
                                              "from which geometry errors are read if geom_errors is set to \"file\".")
        ERROR_FILE_OUT = Parameter("error_file_out", str,
                                   description="The name of a text file "
                                               "to which the random geometry errors are written "
                                               "if geom_errors is set to \"random\".")

        def type(self):
            return self.value.type

        @staticmethod
        def parameter_names():
            return [p.value.name for p in Geometry.Parameters]

    GEOM_ERR_OPTIONS = ["none", "random", "file"]

    REQUIRED_PARAMETERS = [Parameters.GEN_TILT_DATA, Parameters.GEOM_ERRORS]
    PARAMETER_DEFAULTS = {
        Parameters.TILT_AXIS: 0.,
        Parameters.TILT_MODE: TiltModes.TILT_SERIES,
        Parameters.TILT_ERR: 0.,
        Parameters.ALIGN_ERR_ROT: 0.,
        Parameters.ALIGN_ERR_TR: 0.,
    }
    LEARNABLE_PARAMETER_NAMES = ["tilt_err", "align_err_rot", "align_err_tr"]

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self._data: torch.Tensor = UNINITIALIZED
        self._errors: torch.Tensor = UNINITIALIZED
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
        assert self._initialized, "Initialize geometry first"
        g = Geometry(self.name)
        g._data = self._data.to(device)
        g._errors = self._errors.to(device)
        g._parameters = deepcopy(self._parameters)
        g._initialized = self._initialized
        g._lock_parameters = self._lock_parameters
        return g

    def to_(self, device: torch.device):
        """
        In-place move to a device

        :param device: device that holds data
        """
        assert self._initialized, "Initialize geometry first"
        self._lock_parameters = False
        self._data = self._data.to(device)
        self._errors = self._errors.to(device)
        self._lock_parameters = True

    @staticmethod
    def construct_from_parameters(parameters: Union[List[Dict[str, str]], Dict[str, str]],
                                  initialize: Optional[bool] = True):
        """
        Construct instances of Geometry given parameters

        :param parameters: list of parameter dictionaries or a dictionary containing parameters
        :param initialize: whether to initialize instances, default: True
        :return: list of instances or one instance
        """
        if isinstance(parameters, list):
            geometries = [Geometry._from_parameters(parameter_table) for parameter_table in parameters]
            if initialize:
                for g in geometries:
                    g.finish_init_()
            return geometries
        elif isinstance(parameters, dict):
            geometry = Geometry._from_parameters(parameters)
            if initialize:
                geometry.finish_init_()
            return geometry
        else:
            raise Exception(f"Invalid parameter of type {type(parameters)}")

    @staticmethod
    def _from_parameters(parameters: Dict[str, str]):
        geometry = Geometry()
        if "tilt_mode" in parameters:
            geometry.tilt_mode = Geometry.TiltModes(parameters["tilt_mode"])
        for p in Geometry.Parameters:
            if p != Geometry.Parameters.TILT_MODE:
                parameter_name = p.value.name
                parameter_type = p.type()
                if parameter_type == bool:
                    parameter_type = convert_bool
                if parameter_name in parameters:
                    parameter_val = parameter_type(parameters[parameter_name])
                    setattr(geometry, parameter_name, parameter_val)
        return geometry

    def _unwrap_required_parameter(self, parameter: Parameters) -> Any:
        parameter_value = self._parameters[parameter]
        if parameter_value is None:
            raise RequiredParameterNotSpecifiedException(parameter)
        return parameter_value

    def finish_init_(self):
        """
        Finish initialization of Geometry

        :return: self
        """
        if self._initialized:
            return self
        # check parameters, CORRESPOND: geometry_check_input
        for parameter_type in self.REQUIRED_PARAMETERS:
            if self._parameters[parameter_type] is None:
                raise RequiredParameterNotSpecifiedException(parameter_type)

        if self.gen_tilt_data:
            theta_incr = self.theta_incr
            tilt_num = self.ntilts
            # geometry_generate_tilt_data
            matrix = torch.ones(tilt_num, 3, dtype=torch.double)
            theta = degree_to_radian(self.theta_start)
            theta_incr = degree_to_radian(theta_incr)
            tilt_axis = degree_to_radian(self.tilt_axis)
            for i in range(tilt_num):
                matrix[i, 0] = -tilt_axis
                matrix[i, 1] = theta
                matrix[i, 2] = tilt_axis
                theta += theta_incr

            self._data = matrix

            geom_file_out = self._parameters[self.Parameters.GEOM_FILE_OUT]
            if geom_file_out is not None:
                # geometry_write_data_to_file
                write_matrix = radian_to_degree(matrix)
                write_matrix_text_with_headers(write_matrix, geom_file_out, "psi", "theta", "phi")
        else:
            # geometry_read_data_from_file
            matrix = read_matrix_text(self.geom_file_in)
            matrix = degree_to_radian(matrix)
            tilt_num = self._parameters[self.Parameters.N_TILTS]
            if tilt_num is not None:
                assert tilt_num == matrix.shape[0]
            else:
                self.ntilts = matrix.shape[0]
                _write_log(f"Number of tilts read from geometry data file: {matrix.shape[0]}")

        errors = self.geom_errors
        if errors == self.GEOM_ERR_OPTIONS[1]:  # random
            tilt_err = self.tilt_err
            align_err_rot = self.align_err_rot
            align_err_tr = self.align_err_tr
            # geometry_generate_random_errors
            tilt_num = self.ntilts
            align_err_rot = degree_to_radian(align_err_rot)
            align_err_tr = math.sqrt(0.5) * align_err_tr
            tilt_err = degree_to_radian(tilt_err)
            error_matrix = torch.zeros(tilt_num, 5)
            error_matrix[:, 0] = torch.randn(tilt_num) * align_err_rot
            if tilt_err != 0.:
                tilt_axis = torch.rand(tilt_num) * PI
                error_matrix[:, 1] = tilt_axis
            error_matrix[:, 2] = torch.randn(tilt_num) * tilt_err
            error_matrix[:, 3] = torch.randn(tilt_num) * align_err_tr
            error_matrix[:, 4] = torch.randn(tilt_num) * align_err_tr
            self._errors = error_matrix

        elif errors == self.GEOM_ERR_OPTIONS[2]:  # file
            # geometry_read_errors_from_file
            error_matrix = read_matrix_text(self.error_file_in)
            error_matrix[:, :3] = degree_to_radian(error_matrix[:, :3])
            check_tensor(error_matrix, shape=(self.ntilts, 5))
            self._errors = error_matrix

        else:  # none
            self._errors = torch.zeros(self.ntilts, 5)

        assert self.tilt_mode is not None
        self._initialized = True
        self._lock_parameters = True
        _write_log("Geometry component initialized.")

    def get_particle_geom(self, particle_set, i: int, tilt: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the mapping from particle coordinate system to microscope coordinate system
        for a given particle and a given tilt.
        :param particle_set: particle set which the particle belongs to
        :param i: index of that particle
        :param tilt: tilt index of the image in tilt series.
        :return: transformation matrix and particle position
        """
        # CORRESPOND: get_particle_geom()
        assert self._initialized, "Geometry not initialized"
        assert particle_set.initialized, "Particleset not initialized"
        assert 0 <= tilt < self._data.shape[0], "Error in get_particle_geom: tilt number out of range."
        assert 0 <= i < particle_set.coordinates.shape[0]
        proj_matrix = self.get_proj_matrix(tilt)
        x = particle_set.get_particle_pos(i)
        pos = torch.tensor([self._errors[tilt, 3], self._errors[tilt, 4], 0.0], dtype=torch.double)
        if self.tilt_mode == self.TiltModes.TILT_SERIES:
            pos = pos + (x.unsqueeze(0) @ proj_matrix).squeeze()
        else:
            pos = pos + x

        particle_rotate_matrix = particle_set.get_particle_rotation_matrix(i)
        matrix = particle_rotate_matrix @ proj_matrix
        return matrix, pos

    def get_proj_matrix(self, tilt_idx: int) -> torch.Tensor:
        """
        Get projection matrix mapping the sample coordinate system to the microscope coordinate system for a given tilt.
        :param tilt_idx:
        :return: rotation matrix
        """
        # CORRESPOND: set_proj_matrix
        # NOTE: instead of modifying an input matrix via pointer, we return a new matrix
        assert self._initialized, "Geometry not initialized"
        assert self._data.shape[0] > tilt_idx >= 0, "Tilt idx out of bound"
        x_axis = torch.tensor([1., 0., 0.], dtype=torch.double)
        z_axis = torch.tensor([0., 0., 1.], dtype=torch.double)
        error_vec = self._errors[tilt_idx]
        r0 = rotate_3d_rows(z_axis, -error_vec[0])
        v = error_vec[1]
        r1 = rotate_3d_rows(torch.tensor([torch.cos(v), torch.sin(v), 0.]), -error_vec[2])
        data_vec = self._data[tilt_idx]
        r2 = rotate_3d_rows(z_axis, -data_vec[2])
        r3 = rotate_3d_rows(x_axis, -data_vec[1])
        r4 = rotate_3d_rows(z_axis, -data_vec[0])
        rotation_matrix = r4 @ r3 @ r2 @ r1 @ r0
        return rotation_matrix

    def reset(self):
        """
        Reset Geometry
        """
        if not self._initialized:
            return self
        self._data = UNINITIALIZED
        self._errors = UNINITIALIZED
        self._lock_parameters = False
        self._initialized = False

    def log_parameters(self, logger: logging.Logger):
        logger.info(f"Geometry:\n   {self._parameters}")

    def get_sample_geom(self, sample, tilt) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get projection matrix and position of a sample in one tilt
        :param sample: Sample
        :param tilt: tilt idx
        """
        assert self._initialized
        if self.tilt_mode == self.TiltModes.TILT_SERIES:
            project_matrix = self.get_proj_matrix(tilt)
        else:
            project_matrix = torch.eye(3)

        x = torch.tensor([sample.offset_x, sample.offset_y]).view(2, 1)
        error = torch.tensor([self._errors[tilt, 3], self._errors[tilt, 4], 0.])
        pos = (project_matrix[:, :2] @ x).squeeze(-1) + error
        return project_matrix, pos

    @property
    def device(self):
        assert self._initialized, "Initialized geometry first"
        return self._data.device

    @property
    def data(self):
        assert self._initialized, "Initialized geometry first"
        return self._data

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def gen_tilt_data(self) -> bool:
        """
        Controls whether the tilt geometry should be automatically generated or read from a file.
        If the tilt geometry consists of rotations around a fixed tilt axis
        perpendicular to the optical axis it can be automatically generated
        by specifying the parameters tilt_axis, theta_start, and theta_incr.
        For more complicated geometries it is necessary to read input from a text file.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Geometry.Parameters.GEN_TILT_DATA)

    @property
    def ntilts(self) -> int:
        """
        The number of images in the tilt series.
        If ntilts is not specified and the tilt geometry is read from a file,
        the number of images is instead determined by the number of lines in the file.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Geometry.Parameters.N_TILTS)

    @property
    def tilt_axis(self) -> float:
        """
        The angle in degrees between the x-axis and the tilt axis, if the tilt geometry is automatically generated.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Geometry.Parameters.TILT_AXIS)

    @property
    def theta_start(self) -> float:
        """
        The tilt angle in degrees of the first image, if the tilt geometry is automatically generated.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Geometry.Parameters.THETA_START)

    @property
    def theta_incr(self) -> float:
        """
        The increment of the tilt angle in degrees between successive images, if the tilt geometry is automatically generated.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Geometry.Parameters.THETA_INCR)

    @property
    def tilt_mode(self) -> TiltModes:
        """
        Controls how the tilt geometry is applied to the sample.
        \"tiltseries\" means that rotations are applied to the entire sample and the particle positions.
        \"single_particle\" means that the sample and particle positions stay fixed, while individual particles are rotated around their center.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Geometry.Parameters.TILT_MODE)

    @property
    def geom_file_in(self) -> str:
        """
        The name of a text file from which the tilt geometry is read if gen_tilt_data is set to no.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Geometry.Parameters.GEOM_FILE_IN)

    @property
    def geom_file_out(self) -> str:
        """
        The name of a text file to which the generated tilt geometry is written if gen_tilt_data is set to yes.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Geometry.Parameters.GEOM_FILE_OUT)

    @property
    def geom_errors(self) -> str:
        """
        Controls whether geometry errors should be included, and if the errors should be random or read from a file.
        If the errors are random, their magnitude is controlled by the parameters \"tilt_err\", \"align_err_rot\", and \"align_err_tr\".

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Geometry.Parameters.GEOM_ERRORS)

    @property
    def tilt_err(self) -> float:
        """

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Geometry.Parameters.TILT_ERR)

    @property
    def align_err_rot(self) -> float:
        """

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Geometry.Parameters.ALIGN_ERR_ROT)

    @property
    def align_err_tr(self) -> float:
        """

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Geometry.Parameters.ALIGN_ERR_TR)

    @property
    def error_file_in(self) -> str:
        """

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Geometry.Parameters.ERROR_FILE_IN)

    @property
    def error_file_out(self) -> str:
        """

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Geometry.Parameters.ERROR_FILE_OUT)

    @property
    def errors(self) -> torch.Tensor:
        return self._errors

    def _set_parameter(self, parameter: Parameters, value):
        if isinstance(value, parameter.type()):
            if self._lock_parameters:
                raise ParameterLockedException(parameter)
            else:
                self._parameters[parameter] = value
        else:
            raise WrongArgumentTypeException(value, parameter.type())

    @gen_tilt_data.setter
    def gen_tilt_data(self, gen_tilt_data: bool):
        self._set_parameter(self.Parameters.GEN_TILT_DATA, gen_tilt_data)

    @ntilts.setter
    def ntilts(self, ntilts: int):
        self._set_parameter(self.Parameters.N_TILTS, ntilts)
        assert ntilts > 0

    @tilt_axis.setter
    def tilt_axis(self, tilt_axis: float):
        self._set_parameter(self.Parameters.TILT_AXIS, tilt_axis)

    @theta_start.setter
    def theta_start(self, theta_start: float):
        self._set_parameter(self.Parameters.THETA_START, theta_start)

    @theta_incr.setter
    def theta_incr(self, theta_incr: float):
        self._set_parameter(self.Parameters.THETA_INCR, theta_incr)

    @tilt_mode.setter
    def tilt_mode(self, tilt_mode: TiltModes):
        if isinstance(tilt_mode, Geometry.TiltModes):
            self._parameters[self.Parameters.TILT_MODE] = tilt_mode
        else:
            raise WrongArgumentTypeException(tilt_mode, Geometry.TiltModes)

    @geom_file_in.setter
    def geom_file_in(self, geom_file_in: str):
        self._set_parameter(self.Parameters.GEOM_FILE_IN, geom_file_in)

    @geom_file_out.setter
    def geom_file_out(self, geom_file_out: str):
        self._set_parameter(self.Parameters.GEOM_FILE_OUT, geom_file_out)

    @geom_errors.setter
    def geom_errors(self, geom_errors: str):
        self._set_parameter(self.Parameters.GEOM_ERRORS, geom_errors)
        assert geom_errors in self.GEOM_ERR_OPTIONS, f"Invalid value, got = {geom_errors}, expect one of {self.GEOM_ERR_OPTIONS}"

    @tilt_err.setter
    def tilt_err(self, tilt_err: float):
        self._set_parameter(self.Parameters.TILT_ERR, tilt_err)

    @align_err_rot.setter
    def align_err_rot(self, align_err_rot: float):
        self._set_parameter(self.Parameters.ALIGN_ERR_ROT, align_err_rot)
        assert align_err_rot >= 0.

    @align_err_tr.setter
    def align_err_tr(self, align_err_tr: float):
        self._set_parameter(self.Parameters.ALIGN_ERR_TR, align_err_tr)
        assert align_err_tr >= 0.

    @error_file_in.setter
    def error_file_in(self, error_file_in: str):
        self._set_parameter(self.Parameters.ERROR_FILE_IN, error_file_in)

    @error_file_out.setter
    def error_file_out(self, error_file_out: str):
        self._set_parameter(self.Parameters.ERROR_FILE_OUT, error_file_out)
