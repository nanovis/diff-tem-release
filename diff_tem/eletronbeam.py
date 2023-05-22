from copy import deepcopy
from typing import Optional, Any, Union, List, Dict

import torch
import torch.nn.functional as F
import logging
from enum import Enum
from . import UNINITIALIZED
from .constants.physical_constants import *
from .constants import PI
from .constants.io_units import POTENTIAL_UNIT, ENERGY_SPREAD_UNIT, ACC_ENERGY_UNIT
from .constants.derived_units import ONE_ELECTRONVOLT
from .utils.exceptions import *
from .utils.input_processing import convert_bool
from .utils.parameter import Parameter
from .utils.helper_functions import eprint
from .utils.tensor_io import write_matrix_text_with_headers, read_matrix_text

_logger = logging.getLogger(__name__)


def _write_log(string, *logging_strings):
    strings = [string, *logging_strings]
    if len(strings) > 1:
        logging_string = "\n    ".join(strings)
        _logger.debug(f"{_logger.name}:\n    {logging_string}", stacklevel=2)
    else:
        _logger.debug(f"{_logger.name}:  {strings[0]}", stacklevel=2)


class Electronbeam:
    """
    The electronbeam component controls the properties of the electron beam.
    An electronbeam component is required for simulation of micrographs.
    """

    class Parameters(Enum):
        ACC_VOLTAGE = Parameter("acc_voltage", float, description="Acceleration voltage of the microscope in kV.")
        ENERGY_SPREAD = Parameter("energy_spread", float, description="Energy spread of the beam in eV.")
        GEN_DOSE = Parameter("gen_dose", bool,
                             description="Controls whether the electron dose for each image "
                                         "should be automatically generated or read from a file.")
        TOTAL_DOSE = Parameter("total_dose", float,
                               description="If the dose in each image is to be "
                                           "automatically generated, "
                                           "either total_dose or dose_per_im must be provided. "
                                           "total_dose sets the total dose in all the images, "
                                           "measured in electrons per nm^2.")
        DOSE_PER_IM = Parameter("dose_per_im", float,
                                description="If the dose in each image is to be "
                                            "automatically generated, either total_dose "
                                            "or dose_per_im must be provided. "
                                            "dose_per_im sets the average dose in each image, "
                                            "measured in electrons per nm^2.")
        DOSE_SD = Parameter("dose_sd", float,
                            description="Standard deviation of the average dose used in each image, "
                                        "relative to the average dose per image, if the dose "
                                        "in each image is automatically generated.")
        DOSE_FILE_IN = Parameter("dose_file_in", str,
                                 description="A text file from which to read the dose of each image, "
                                             "if it is not to be automatically generated.")
        DOSE_FILE_OUT = Parameter("dose_file_out", str,
                                  description="A text file to which the dose in each image is written "
                                              "if it has been automatically generated.")

        def type(self):
            return self.value.type

        @staticmethod
        def parameter_names():
            return [p.value.name for p in Electronbeam.Parameters]

    REQUIRED_PARAMETERS = [Parameters.ACC_VOLTAGE, Parameters.ENERGY_SPREAD, Parameters.GEN_DOSE]
    LEARNABLE_PARAMETER_NAMES = ["total_dose", "dose_per_im", "dose_sd", "energy_spread", "acc_voltage"]

    def __init__(self, name: Optional[str] = None):
        self.name = name
        self._dose: torch.Tensor = UNINITIALIZED  # (num_tilt, )
        self._initialized = False
        self._parameters = {parameter_type: None for parameter_type in Electronbeam.Parameters}
        self._lock_parameters = False
        self.dose_sd = 0.  # default

    def to(self, device: torch.device):
        """
        Out-of-place move to a device, creating new copy of self whose data is on the device

        :param device: device that holds data
        :return: a new copy of self
        """
        assert self._initialized, "Initialize electronbeam first"
        electron_beam = Electronbeam(self.name)
        electron_beam._dose = self._dose.to(device)
        electron_beam._initialized = self._initialized
        electron_beam._parameters = deepcopy(self._parameters)
        electron_beam.acc_voltage = self.acc_voltage.to(device)
        electron_beam.dose_sd = self.dose_sd
        electron_beam._lock_parameters = self._lock_parameters
        return electron_beam

    def to_(self, device: torch.device):
        """
        In-place move to a device

        :param device: device that holds data
        """
        assert self._initialized, "Initialize electronbeam first"
        self._lock_parameters = False
        self._dose = self._dose.to(device)
        self.acc_voltage = self.acc_voltage.to(device)
        self._lock_parameters = True

    @staticmethod
    def prototype_from_parameters(parameters: Union[List[Dict[str, str]], Dict[str, str]],
                                  initialize: Optional[bool] = False,
                                  tilt_num: Optional[int] = None):
        """
        Construct instances of Electronbeam given parameters

        :param parameters: list of parameter dictionaries or a dictionary containing parameters
        :param initialize: whether to initialize instances, default: False
        :param tilt_num: default None, but if initialize is set to True, it must be a valid integer
        :return: list of instances or one instance
        """
        if isinstance(parameters, list):
            electronbeams = [Electronbeam._from_parameters(parameter_table) for parameter_table in parameters]
            if initialize:
                assert tilt_num is not None, "When require init, tilt num must be provided"
                for e in electronbeams:
                    e.finish_init_(tilt_num)
            return electronbeams
        elif isinstance(parameters, dict):
            electronbeam = Electronbeam._from_parameters(parameters)
            if initialize:
                assert tilt_num is not None, "When require init, tilt num must be provided"
                electronbeam.finish_init_(tilt_num)
            return electronbeam
        else:
            raise Exception(f"Invalid parameter of type {type(parameters)}")

    @staticmethod
    def _from_parameters(parameters: Dict[str, str]):
        electronbeam = Electronbeam()
        for p in Electronbeam.Parameters:
            parameter_name = p.value.name
            parameter_type = p.type()
            if parameter_type == bool:
                parameter_type = convert_bool
            if parameter_name in parameters:
                parameter_val = parameter_type(parameters[parameter_name])
                setattr(electronbeam, parameter_name, parameter_val)
        return electronbeam

    def _calc_dose(self, tilt_num):
        # CORRESPOND: electronbeam_gen_dose
        if tilt_num <= 0:
            raise Exception(f"Invalid tilt number = {tilt_num}")
        total_dose = self._parameters[self.Parameters.TOTAL_DOSE]
        if total_dose is not None:
            avg_dose = total_dose / tilt_num
        else:
            avg_dose = self.dose_per_im
        dose = F.relu(avg_dose * (1.0 + torch.randn(tilt_num) * self.dose_sd))
        return dose

    def finish_init_(self, tilt_num: int):
        """
        Finish initialization of Electronbeam
        :param tilt_num: tilt number for calculating dose
        :return: self
        """
        # CORRESPOND: electronbeam_init
        if self._initialized:
            return self
        else:
            # CORRESPOND: electronbeam_check_input
            # check required parameters
            for parameter_type in self.REQUIRED_PARAMETERS:
                if self._parameters[parameter_type] is None:
                    raise RequiredParameterNotSpecifiedException(parameter_type)
            if self._parameters[self.Parameters.GEN_DOSE]:
                total_dose = self._parameters[self.Parameters.TOTAL_DOSE]
                dose_per_im = self._parameters[self.Parameters.DOSE_PER_IM]
                if total_dose is None and dose_per_im is None:
                    raise Exception(f"You've set {self.Parameters.GEN_DOSE} "
                                    f"but not specified "
                                    f"one of {self.Parameters.TOTAL_DOSE} and {self.Parameters.DOSE_PER_IM}")
                if total_dose is not None and dose_per_im is not None:
                    eprint("WARNING: Both total dose and dose per image specified. "
                           "Total dose value will be used in simulation.")
                self._dose = self._calc_dose(tilt_num)
                if self._parameters[self.Parameters.DOSE_FILE_OUT] is not None:
                    # CORRESPOND: electronbeam_write_dose_to_file
                    write_matrix_text_with_headers(self._dose.reshape(-1, 1),
                                                   self._parameters[self.Parameters.DOSE_FILE_OUT], "dose")
            else:
                dose_file = self._parameters[self.Parameters.DOSE_FILE_IN]
                if dose_file is None:
                    raise Exception(f"You've set {self.Parameters.GEN_DOSE} "
                                    f"but not specified {self.Parameters.DOSE_FILE_IN}")
                else:
                    # CORRESPOND: electronbeam_read_dose_from_file
                    dose = read_matrix_text(dose_file).squeeze()
                    if dose.shape[0] != tilt_num:
                        raise Exception(f"Unmatched dose configuration in file {dose_file}, "
                                        f"require {tilt_num} doses, "
                                        f"got {dose.shape[0]}")
            # all set
            self._initialized = True
            self._lock_parameters = True
            _write_log("Electronbeam component initialized.")
            return self

    def reset(self):
        """
        Reset Electronbeam's values
        :return:
        """
        # CORRESPOND: electronbeam_reset
        # note: not needed, since only used when handling invalid parameters that will throw exceptions in Python
        if self._initialized:
            self._dose = None
            self._initialized = False
            self._parameters = {parameter_type: None for parameter_type in Electronbeam.Parameters}
            self._lock_parameters = False

    def log_parameters(self, logger: logging.Logger):
        logger.info(f"Electronbeam:\n   {self._parameters}")

    def _unwrap_required_parameter(self, parameter: Parameters) -> Any:
        parameter_value = self._parameters[parameter]
        if parameter_value is None:
            raise RequiredParameterNotSpecifiedException(parameter)
        return parameter_value

    @property
    def device(self):
        assert self._initialized, "Initialize electronbeam first"
        return self._dose.device

    @property
    def doses(self) -> torch.Tensor:
        assert self._initialized, "Initialize electronbeam first"
        return self._dose

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def energy_spread_with_unit(self) -> float:
        """
        Energy spread of the beam with unit

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return ENERGY_SPREAD_UNIT * self.energy_spread

    @property
    def acceleration_energy(self) -> torch.Tensor:
        """
        Get acceleration energy of electrons as determined by acceleration voltage read from input.

        :raises RequiredParameterNotSpecifiedException:: if this parameter is not set
        """
        return ACC_ENERGY_UNIT * self.acc_voltage

    @property
    def acc_voltage(self) -> torch.Tensor:
        """
        Acceleration voltage of the microscope in kV.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Electronbeam.Parameters.ACC_VOLTAGE)

    @property
    def energy_spread(self) -> float:
        """
        Energy spread of the beam in eV.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Electronbeam.Parameters.ENERGY_SPREAD)

    @property
    def gen_dose(self) -> bool:
        """
        Controls whether the electron dose for each image should be automatically generated or read from a file.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Electronbeam.Parameters.GEN_DOSE)

    @property
    def total_dose(self) -> float:
        """
        If the dose in each image is to be automatically generated,
        either total_dose or dose_per_im must be provided.
        total_dose sets the total dose in all the images, measured in electrons per nm^2.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Electronbeam.Parameters.TOTAL_DOSE)

    @property
    def dose_per_im(self) -> float:
        """
        If the dose in each image is to be automatically generated,
        either total_dose or dose_per_im must be provided.
        dose_per_im sets the average dose in each image, measured in electrons per nm^2.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Electronbeam.Parameters.DOSE_PER_IM)

    @property
    def dose_sd(self) -> float:
        """
        Standard deviation of the average dose used in each image,
        relative to the average dose per image,
        if the dose in each image is automatically generated.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Electronbeam.Parameters.DOSE_SD)

    @property
    def dose_file_in(self) -> str:
        """
        A text file from which to read the dose of each image, if it is not to be automatically generated.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Electronbeam.Parameters.DOSE_FILE_IN)

    @property
    def dose_file_out(self) -> str:
        """
        A text file to which the dose in each image is written if it has been automatically generated.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Electronbeam.Parameters.DOSE_FILE_OUT)

    def _set_parameter(self, parameter: Parameters, value, convert_to_tensor=False, requires_grad=False):
        if isinstance(value, parameter.type()):
            if self._lock_parameters:
                raise ParameterLockedException(parameter)
            else:
                if convert_to_tensor:
                    self._parameters[parameter] = torch.tensor(value, requires_grad=requires_grad)
                else:
                    self._parameters[parameter] = value
        else:
            raise WrongArgumentTypeException(value, parameter.type())

    @acc_voltage.setter
    def acc_voltage(self, acc_voltage: float):
        self._set_parameter(self.Parameters.ACC_VOLTAGE, acc_voltage, convert_to_tensor=True)

    @energy_spread.setter
    def energy_spread(self, energy_spread: float):
        self._set_parameter(self.Parameters.ENERGY_SPREAD, energy_spread)

    @gen_dose.setter
    def gen_dose(self, gen_dose: bool):
        self._set_parameter(self.Parameters.GEN_DOSE, gen_dose)

    @total_dose.setter
    def total_dose(self, total_dose: float):
        self._set_parameter(self.Parameters.TOTAL_DOSE, total_dose)

    @dose_per_im.setter
    def dose_per_im(self, dose_per_im: float):
        self._set_parameter(self.Parameters.DOSE_PER_IM, dose_per_im)

    @dose_sd.setter
    def dose_sd(self, dose_sd: float):
        self._set_parameter(self.Parameters.DOSE_SD, dose_sd)

    @dose_file_in.setter
    def dose_file_in(self, dose_file_in: str):
        self._set_parameter(self.Parameters.DOSE_FILE_IN, dose_file_in)

    @dose_file_out.setter
    def dose_file_out(self, dose_file_out: str):
        self._set_parameter(self.Parameters.DOSE_FILE_OUT, dose_file_out)


def wave_number(acceleration_energy: torch.Tensor) -> torch.Tensor:
    """
    Compute wave number of electron wave for given acceleration energy.

    :param acceleration_energy: Acceleration energy of electrons
    :return: Wave number is standard units
    """
    return 2 * PI / (SPEED_OF_LIGHT * PLANCKS_CONSTANT) * \
           torch.sqrt(acceleration_energy * (acceleration_energy + 2 * ELEC_REST_ENERGY))


def potential_conv_factor(acceleration_energy: torch.Tensor) -> torch.Tensor:
    """
    Compute factor by which projected potential should be multiplied to obtain phase shift of electron wave.

    :param acceleration_energy: Acceleration energy of electrons
    :return: Multiplicative factor
    """
    return 4 * PI * PI * ELECTRON_MASS * ELEMENTARY_CHARGE / PLANCKS_CONSTANT_SQ * POTENTIAL_UNIT \
           * (1.0 + acceleration_energy / ELEC_REST_ENERGY) / wave_number(acceleration_energy)


def differential_cross_section(acceleration_energy: torch.Tensor,
                               atomic_number: torch.Tensor,
                               scattering_angle: torch.Tensor) -> torch.Tensor:
    """
    Compute differential cross section for elastic and inelastic scattering of electrons
    from a single neutral atom according to model in Ludwig Reimer, Transmission electron microscopy:
    physics of image formation and microanalysis.

    :param acceleration_energy: Acceleration energy of electrons
    :param atomic_number: Atomic number
    :param scattering_angle: Scattering angle
    :return: Differential cross section
    """
    # CORRESPOND: diff_cross_sec
    return differential_elastic_cross_section_scattering(acceleration_energy, atomic_number, scattering_angle) \
           + differential_inelastic_cross_section_scattering(acceleration_energy, atomic_number, scattering_angle)


def total_cross_section(acceleration_energy: torch.Tensor, atomic_number: torch.Tensor) -> torch.Tensor:
    """
    Compute total cross section for elastic and inelastic scattering of electrons from a single neutral atom
    according to model in Ludwig Reimer, Transmission electron microscopy: physics of image formation and microanalysis.

    :param acceleration_energy: Acceleration energy of electrons.
    :param atomic_number: Atomic number.
    :return: Cross section.
    """
    # CORRESPOND: cross_sec
    return elastic_cross_section(acceleration_energy, atomic_number) \
           + inelastic_cross_section(acceleration_energy, atomic_number)


def differential_elastic_cross_section_scattering(acceleration_energy: torch.Tensor,
                                                  atomic_number: torch.Tensor,
                                                  scattering_angle: torch.Tensor) -> torch.Tensor:
    """
    Compute differential cross section for elastic scattering of electrons from a single neutral atom according to model
    in Ludwig Reimer, Transmission electron microscopy: physics of image formation and microanalysis.

    :param acceleration_energy: Acceleration energy of electrons.
    :param atomic_number: Atomic number.
    :param scattering_angle: Scattering angle.
    :return: Differential cross section.
    """
    # CORRESPOND: diff_el_cross_sec
    k = wave_number(acceleration_energy)
    theta02 = torch.pow(atomic_number, 1. / 3.0) / (k * BOHR_RADIUS)
    theta02 *= theta02
    relang2 = scattering_angle * scattering_angle / theta02
    a = BOHR_RADIUS * (1 + acceleration_energy / ELEC_REST_ENERGY) / (1 + relang2)
    return 4 * torch.pow(atomic_number, 2. / 3.) * a * a


def differential_inelastic_cross_section_scattering(acceleration_energy: torch.Tensor,
                                                    atomic_number: torch.Tensor,
                                                    scattering_angle: torch.Tensor) -> torch.Tensor:
    """
    Compute differential cross section for inelastic scattering
    of electrons from a single neutral atom according to model
    in Ludwig Reimer, Transmission electron microscopy: physics of image formation and microanalysis.

    :param acceleration_energy: Acceleration energy of electrons.
    :param atomic_number: Atomic number.
    :param scattering_angle: Scattering angle.
    :return: Differential cross section.
    """
    # CORRESPOND: diff_inel_cross_sec
    k = wave_number(acceleration_energy)
    thetaE2 = 13.5 * ONE_ELECTRONVOLT * atomic_number / (4 * acceleration_energy)
    thetaE2 *= thetaE2
    theta02 = pow(atomic_number, 1.0 / 3.0) / (k * BOHR_RADIUS)
    theta02 *= theta02
    theta2 = scattering_angle * scattering_angle
    a = 1 + theta2 / theta02
    b = (1 + acceleration_energy / ELEC_REST_ENERGY) / (k * k * BOHR_RADIUS * (theta2 + thetaE2))
    return 4 * atomic_number * b * b * (1.0 - 1.0 / (a * a))


def elastic_cross_section(acceleration_energy: torch.Tensor, atomic_number: torch.Tensor) -> torch.Tensor:
    """
    Compute total cross section for elastic scattering of
    electrons from a single neutral atom according to model in
    Ludwig Reimer, Transmission electron microscopy: physics of image formation and microanalysis.

    :param acceleration_energy: Acceleration energy of electrons.
    :param atomic_number: Atomic number.
    :return: Cross section.
    """
    # CORRESPOND: el_cross_sec
    k = wave_number(acceleration_energy)
    a = (1 + acceleration_energy / ELEC_REST_ENERGY) / k
    return 4 * PI * torch.pow(atomic_number, 4.0 / 3.0) * a * a


def inelastic_cross_section(acceleration_energy: torch.Tensor, atomic_number: torch.Tensor) -> torch.Tensor:
    """
    Compute total cross section for inelastic scattering of
    electrons from a single neutral atom according to model in
    Ludwig Reimer, Transmission electron microscopy: physics of image formation and microanalysis.

    :param acceleration_energy: Acceleration energy of electrons.
    :param atomic_number: Atomic number.
    :return: Cross section.
    """
    # CORRESPOND: inel_cross_sec
    k = wave_number(acceleration_energy)
    # 13.5: the mean ionization energy of the atom 13.5 * Z, Z - atomic number
    thetaE2 = 13.5 * ONE_ELECTRONVOLT * atomic_number / (4 * acceleration_energy)
    thetaE2 *= thetaE2
    theta02 = torch.pow(atomic_number, 1. / 3.0) / (k * BOHR_RADIUS)
    theta02 *= theta02
    a = theta02 - thetaE2
    b = (1 + acceleration_energy / ELEC_REST_ENERGY) / (k * k * BOHR_RADIUS * a)
    return 4 * PI * atomic_number * b * b / a \
           * (-3 * theta02 * theta02 + 4 * theta02 * thetaE2 - thetaE2 * thetaE2 + 2 * theta02 * theta02 *
              (torch.log(theta02) - torch.log(thetaE2)))
