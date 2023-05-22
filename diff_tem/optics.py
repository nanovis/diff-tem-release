from copy import deepcopy
from enum import Enum
from typing import Optional, Any, Union, List, Dict

import torch
import logging
from . import UNINITIALIZED
from .utils.input_processing import convert_bool
from .utils.parameter import Parameter
from .utils.exceptions import *
from .utils.tensor_io import read_matrix_text, write_matrix_text_with_headers
from .constants.io_units import APERTURE_UNIT, FOCAL_LENGTH_UNIT, COND_AP_ANGLE_UNIT, CC_UNIT, CS_UNIT, \
    DEFOCUS_UNIT

_logger = logging.getLogger(__name__)


def _write_log(string, *logging_strings):
    strings = [string, *logging_strings]
    if len(strings) > 1:
        logging_string = "\n    ".join(strings)
        _logger.debug(f"{_logger.name}:\n    {logging_string}", stacklevel=2)
    else:
        _logger.debug(f"{_logger.name}:  {strings[0]}", stacklevel=2)


class Optics:
    """
    The optics component specifies the characteristics of the optical system in the microscope.
    Most of the optical parameters stay the same in all the images of a tilt series,
    but the defocus can be varied from image to image.
    An optics component is required for simulation of micrographs.
    """

    class Parameters(Enum):
        MAGNIFICATION = Parameter("magnification", float,
                                  description="The magnification of the microscope.")
        GEN_DEFOCUS = Parameter("gen_defocus", bool,
                                description="Controls if the defocus in each "
                                            "image should be automatically generated "
                                            "as random variations around a nominal value, "
                                            "or if an arbitrary sequence of defocus values "
                                            "should be read from a file.")
        DEFOCUS_NOMINAL = Parameter("defocus_nominal", float,
                                    description="The nominal defocus value used "
                                                "if gen_defocus is set to yes. "
                                                "Defocus is measured in micrometer "
                                                "and a positive value is interpreted as underfocus.")
        DEFOCUS_SYST_ERR = Parameter("defocus_syst_error", float,
                                     description="Standard deviation of a random error applied to the defocus values "
                                                 "if gen_defocus is set to yes. "
                                                 "The same error is used for all images in the tilt series. "
                                                 "Measured in micrometer.")
        DEFOCUS_NONSYST_ERR = Parameter("defocus_nonsyst_error", float,
                                        description="Standard deviation of a random error applied "
                                                    "to the defocus values if gen_defocus is set to yes. "
                                                    "A new random error is computed for each image. "
                                                    "Measured in micrometer.")
        CS = Parameter("cs", float, description="The spherical aberration measured in mm.")
        CC = Parameter("cc", float, description="The chromatic aberration measured in mm.")
        APERTURE = Parameter("aperture", float,
                             description="The diameter of the aperture in the focal plane, measured in micrometer.")
        FOCAL_LENGTH = Parameter("focal_length", float,
                                 description="The focal length of the primary lens measured in mm.")
        COND_AP_ANGLE = Parameter("cond_ap_angle", float,
                                  description="The aperture angle of the condenser lens, measured in milliradian.")
        DEFOCUS_FILE_IN = Parameter("defocus_file_in", str,
                                    description="A text file which the defocus of "
                                                "each image in the tilt series is read from "
                                                "if gen_defocus is set to no.")
        DEFOCUS_FILE_OUT = Parameter("defocus_file_out", str,
                                     description="A text file which the defocus of each image "
                                                 "in the tilt series is written to if gen_defocus is set to yes. "
                                                 "If the parameter is not specified, "
                                                 "defocus values are not written to file.")

        def type(self):
            return self.value.type

        @staticmethod
        def parameter_names():
            return [p.value.name for p in Optics.Parameters]

    REQUIRED_PARAMETERS = [Parameters.MAGNIFICATION, Parameters.GEN_DEFOCUS, Parameters.CS, Parameters.CC,
                           Parameters.APERTURE, Parameters.FOCAL_LENGTH, Parameters.COND_AP_ANGLE]

    LEARNABLE_PARAMETER_NAMES = ["defocus_nominal", "defocus_syst_error", "defocus_nonsyst_error"]

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self._defocus: torch.Tensor = UNINITIALIZED
        self._parameters = {parameter_type: None for parameter_type in self.Parameters}
        self._lock_parameters = False
        # defaults
        self.defocus_syst_err = 0.
        self.defocus_nonsyst_err = 0.
        self._initialized = False

    def to(self, device: torch.device):
        """
        Out-of-place move to a device, creating new copy of self whose data is on the device

        :param device: device that holds data
        :return: a new copy of self
        """
        assert self._initialized, "Initialized optics first"
        optics = Optics(self.name)
        optics._defocus = self._defocus.to(device)
        optics._parameters = deepcopy(self._parameters)
        optics._initialized = self._initialized
        optics._lock_parameters = self._lock_parameters
        return optics

    def to_(self, device: torch.device):
        """
        In-place move to a device

        :param device: device that holds data
        """
        assert self._initialized, "Initialized optics first"
        self._lock_parameters = False
        self._defocus = self._defocus.to(device)
        self._lock_parameters = True

    @staticmethod
    def prototype_from_parameters(parameters: Union[List[Dict[str, str]], Dict[str, str]],
                                  initialize: Optional[bool] = False,
                                  tilt_num: Optional[int] = None):
        """
        Construct instances of Optics given parameters

        :param parameters: list of parameter dictionaries or a dictionary containing parameters
        :param initialize: whether to initialize instances, default: False
        :param tilt_num: default None, but if initialize is set to True, it must be a valid integer
        :return: list of instances or one instance
        """
        if isinstance(parameters, list):
            optics = [Optics._from_parameters(parameter_table) for parameter_table in parameters]
            if initialize:
                assert tilt_num is not None, "When require init, tilt num must be provided"
                for o in optics:
                    o.finish_init_(tilt_num)
            return optics
        elif isinstance(parameters, dict):
            optics = Optics._from_parameters(parameters)
            if initialize:
                assert tilt_num is not None, "When require init, tilt num must be provided"
                optics.finish_init_(tilt_num)
            return optics
        else:
            raise Exception(f"Invalid parameter of type {type(parameters)}")

    @staticmethod
    def _from_parameters(parameters: Dict[str, str]):
        optics = Optics()
        for p in Optics.Parameters:
            parameter_name = p.value.name
            parameter_type = p.type()
            if parameter_type == bool:
                parameter_type = convert_bool
            if parameter_name in parameters:
                parameter_val = parameter_type(parameters[parameter_name])
                setattr(optics, parameter_name, parameter_val)
        return optics

    def _gen_random_defocus(self, tilt_num: int):
        # CORRESPOND: optics_generate_random_defocus
        if tilt_num <= 0:
            raise Exception(f"Invalid tilt number = {tilt_num}")
        def_avg = torch.randn(1) * self.defocus_syst_err + self.defocus_nominal
        defocus = torch.randn(tilt_num) * self.defocus_nonsyst_err + def_avg
        return defocus

    def finish_init_(self, tilt_num: int):
        """
        Finish initialization of Optics
        :param tilt_num: number of tilts
        :return: self
        """
        # CORRESPOND: optics_init
        if self._initialized:
            return self
        # check parameters, CORRESPOND: optics_check_input
        for parameter_type in self.REQUIRED_PARAMETERS:
            if self._parameters[parameter_type] is None:
                raise RequiredParameterNotSpecifiedException(parameter_type)
        if self._parameters[self.Parameters.GEN_DEFOCUS]:
            self._defocus = self._gen_random_defocus(tilt_num)
            defocus_file_out = self._parameters[self.Parameters.DEFOCUS_FILE_OUT]
            if defocus_file_out is not None:
                # CORRESPOND: optics_write_defocus_to_file
                write_matrix_text_with_headers(self._defocus.reshape(-1, 1), defocus_file_out, "defocus")
        else:
            defocus_file = self.defocus_file_in
            defocus = read_matrix_text(defocus_file)
            self._defocus = defocus

        self._initialized = True
        self._lock_parameters = True
        _write_log("Optics component initialized.")
        return self

    def get_defocus_nanometer(self, tilt: int) -> torch.Tensor:
        """
        Get the defocus in image number tilt, converted to standard length unit.

        :param tilt: index of image in tilt series, zero-based
        :return: Defocus converted in nanometer
        """
        return DEFOCUS_UNIT * self._defocus[tilt]

    def log_parameters(self, logger: logging.Logger):
        logger.info(f"Optics:\n   {self._parameters}")

    def _unwrap_required_parameter(self, parameter: Parameters) -> Any:
        parameter_value = self._parameters[parameter]
        if parameter_value is None:
            raise RequiredParameterNotSpecifiedException(parameter)
        return parameter_value

    @property
    def device(self):
        assert self._initialized, "Initialized optics first"
        return self._defocus.device

    @property
    def defocus(self) -> torch.Tensor:
        assert self._initialized, "Initialized optics first"
        return self._defocus

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def magnification(self) -> float:
        """
        The magnification of the microscope.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set.
        """
        return self._unwrap_required_parameter(Optics.Parameters.MAGNIFICATION)

    @property
    def gen_defocus(self) -> bool:
        """
        Controls if the defocus in each image should be automatically generated as random variations around a nominal value,
        or if an arbitrary sequence of defocus values should be read from a file.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set.
        """
        return self._unwrap_required_parameter(Optics.Parameters.GEN_DEFOCUS)

    @property
    def defocus_nominal(self) -> float:
        """
        The nominal defocus value used if gen_defocus is set to yes.
        Defocus is measured in micrometer and a positive value is interpreted as under focus.
        :raises RequiredParameterNotSpecifiedException: if this parameter is not set.
        """
        return self._unwrap_required_parameter(Optics.Parameters.DEFOCUS_NOMINAL)

    @property
    def defocus_syst_err(self) -> float:
        """
        Standard deviation of a random error applied to the defocus values if gen_defocus is set to yes.
        The same error is used for all images in the tilt series. Measured in micrometer.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set.
        """
        return self._unwrap_required_parameter(Optics.Parameters.DEFOCUS_SYST_ERR)

    @property
    def defocus_nonsyst_err(self) -> float:
        """
        Standard deviation of a random error applied to the defocus values if gen_defocus is set to yes.
        A new random error is computed for each image. Measured in micrometer.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set.
        """
        return self._unwrap_required_parameter(Optics.Parameters.DEFOCUS_NONSYST_ERR)

    @property
    def cs(self) -> float:
        """
        The spherical aberration measured in mm.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set.
        """
        return self._unwrap_required_parameter(Optics.Parameters.CS)

    @property
    def cc(self) -> float:
        """
        The chromatic aberration measured in mm.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set.
        """
        return self._unwrap_required_parameter(Optics.Parameters.CC)

    @property
    def aperture(self) -> float:
        """
        The diameter of the aperture in the focal plane, measured in micrometer.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set.
        """
        return self._unwrap_required_parameter(Optics.Parameters.APERTURE)

    @property
    def focal_length(self) -> float:
        """
        The focal length of the primary lens measured in mm.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set.
        """
        return self._unwrap_required_parameter(Optics.Parameters.FOCAL_LENGTH)

    @property
    def cond_ap_angle(self) -> float:
        """
        The aperture angle of the condenser lens, measured in milliradian.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set.
        """
        return self._unwrap_required_parameter(Optics.Parameters.COND_AP_ANGLE)

    @property
    def defocus_file_in(self) -> str:
        """
        A text file which the defocus of each image in the tilt series is read from if gen_defocus is set to no.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set.
        """
        return self._unwrap_required_parameter(Optics.Parameters.DEFOCUS_FILE_IN)

    @property
    def defocus_file_out(self) -> str:
        """
        A text file which the defocus of each image in the tilt series is written to if gen_defocus is set to yes.
        If the parameter is not specified, defocus values are not written to file.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set.
        """
        return self._unwrap_required_parameter(Optics.Parameters.DEFOCUS_FILE_OUT)

    @property
    def aperture_nanometer(self) -> float:
        """
        The diameter of the aperture in the focal plane, measured in nanometer.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set.
        """
        return APERTURE_UNIT * self.aperture

    @property
    def focal_length_nanometer(self) -> float:
        """
        The focal length of the primary lens measured in nanometer.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set.
        """
        return FOCAL_LENGTH_UNIT * self.focal_length

    @property
    def cond_ap_angle_radian(self) -> float:
        """
        The aperture angle of the condenser lens, measured in radian.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set.
        """
        return COND_AP_ANGLE_UNIT * self.cond_ap_angle

    @property
    def cs_nanometer(self) -> float:
        """
        The spherical aberration measured in nanometer.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set.
        """
        return CS_UNIT * self.cs

    @property
    def cc_nanometer(self) -> float:
        """
        The chromatic aberration measured in nanometer.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set.
        """
        return CC_UNIT * self.cc

    def _set_parameter(self, parameter: Parameters, value):
        if isinstance(value, parameter.type()):
            if self._lock_parameters:
                raise ParameterLockedException(parameter)
            else:
                self._parameters[parameter] = value
        else:
            raise WrongArgumentTypeException(value, parameter.type())

    @magnification.setter
    def magnification(self, magnification: float):
        self._set_parameter(self.Parameters.MAGNIFICATION, magnification)

    @gen_defocus.setter
    def gen_defocus(self, gen_defocus: bool):
        self._set_parameter(self.Parameters.GEN_DEFOCUS, gen_defocus)

    @defocus_nominal.setter
    def defocus_nominal(self, defocus_norminal: float):
        self._set_parameter(self.Parameters.DEFOCUS_NOMINAL, defocus_norminal)

    @defocus_syst_err.setter
    def defocus_syst_err(self, defocus_syst_error: float):
        self._set_parameter(self.Parameters.DEFOCUS_SYST_ERR, defocus_syst_error)

    @defocus_nonsyst_err.setter
    def defocus_nonsyst_err(self, defocus_nonsyst_error: float):
        self._set_parameter(self.Parameters.DEFOCUS_NONSYST_ERR, defocus_nonsyst_error)

    @cs.setter
    def cs(self, cs: float):
        self._set_parameter(self.Parameters.CS, cs)

    @cc.setter
    def cc(self, cc: float):
        self._set_parameter(self.Parameters.CC, cc)

    @aperture.setter
    def aperture(self, aperture: float):
        self._set_parameter(self.Parameters.APERTURE, aperture)

    @focal_length.setter
    def focal_length(self, focal_length: float):
        self._set_parameter(self.Parameters.FOCAL_LENGTH, focal_length)

    @cond_ap_angle.setter
    def cond_ap_angle(self, cond_ap_angle: float):
        self._set_parameter(self.Parameters.COND_AP_ANGLE, cond_ap_angle)

    @defocus_file_in.setter
    def defocus_file_in(self, defocus_file_in: str):
        self._set_parameter(self.Parameters.DEFOCUS_FILE_IN, defocus_file_in)

    @defocus_file_out.setter
    def defocus_file_out(self, defocus_file_out: str):
        self._set_parameter(self.Parameters.DEFOCUS_FILE_OUT, defocus_file_out)
