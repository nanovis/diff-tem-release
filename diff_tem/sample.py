import logging
from typing import Union, Optional, List, Dict
import torch

from .constants import PI
from .constants.derived_units import ONE_ANGSTROM
from .constants.physical_constants import PLANCKS_CONSTANT_SQ, ELECTRON_MASS, ELEMENTARY_CHARGE, ICE_DENS, \
    ELEC_REST_ENERGY, CARBON_DENS
from .constants.io_units import POTENTIAL_UNIT
from .eletronbeam import wave_number, total_cross_section
from .pdb import get_scattering_parameters

_logger = logging.getLogger(__name__)


def _write_log(string, *logging_strings):
    strings = [string, *logging_strings]
    if len(strings) > 1:
        logging_string = "\n    ".join(strings)
        _logger.debug(f"{_logger.name}:\n    {logging_string}", stacklevel=2)
    else:
        _logger.debug(f"{_logger.name}:  {strings[0]}", stacklevel=2)


class Sample:
    """
    The sample component defines the shape of the sample,
    which is a round hole in a supporting slab filled with ice.
    A sample component is required for simulation of micrographs and for generating a volume map.
    """

    PARAMETER_DESCRIPTIONS = {
        "diameter": "The diameter of the hole measured in nm.",
        "thickness_center": "The thickness of the ice at the center of the hole, measured in nm.",
        "thickness_edge": "The thickness of the ice at the egde of the hole, "
                          "and also the thickness of the supporting slab, measured in nm.",
        "offset_x": "The x coordinate of the center of the hole, measured in nm.",
        "offset_y": "The y coordinate of the center of the hole, measured in nm."
    }

    def __init__(self,
                 diameter: float,
                 thickness_center: float,
                 thickness_edge: float,
                 name: str = None,
                 offset_x: Optional[float] = 0.,
                 offset_y: Optional[float] = 0.):
        assert diameter > 0.
        assert thickness_edge > 0.
        assert thickness_edge > 0.
        self.name = name
        self.diameter = diameter
        self.thickness_center = thickness_center
        self.thickness_edge = thickness_edge
        self.offset_x = offset_x
        self.offset_y = offset_y
        _write_log("Sample component initialized.\n")

    @staticmethod
    def construct_from_parameters(parameters: Union[List[Dict[str, str]], Dict[str, str]]):
        """
        Construct initialized instances of Sample given parameters

        :param parameters: list of parameter dictionaries or a dictionary containing parameters
        :return: list of instances or one instance
        """
        if isinstance(parameters, list):
            return [Sample._from_parameters(parameter_table) for parameter_table in parameters]
        elif isinstance(parameters, dict):
            return Sample._from_parameters(parameters)
        else:
            raise Exception(f"Invalid parameter of type {type(parameters)}")

    @staticmethod
    def _from_parameters(parameters: Dict[str, str]):
        diameter = float(parameters["diameter"])
        thickness_center = float(parameters["thickness_center"])
        thickness_edge = float(parameters["thickness_edge"])
        offset_x = 0.
        offset_y = 0.
        if "offset_x" in parameters:
            offset_x = float(parameters["offset_x"])
        if "offset_y" in parameters:
            offset_y = float(parameters["offset_y"])
        return Sample(diameter, thickness_center, thickness_edge, offset_x=offset_x, offset_y=offset_y)

    def log_parameters(self, logger: logging.Logger):
        parameters = {
            "Diameter": self.diameter,
            "ThicknessCenter": self.thickness_center,
            "ThicknessEdge": self.thickness_edge,
            "Offset_x": self.offset_x,
            "Offset_y": self.offset_y
        }
        logger.info(f"Detector:\n   {parameters}")


def sample_ice_pot() -> torch.Tensor:
    a, _ = get_scattering_parameters(8)
    s = a.sum()
    a, _ = get_scattering_parameters(1)
    s += 2 * a.sum()
    s *= ONE_ANGSTROM
    s *= PLANCKS_CONSTANT_SQ / (2.0 * PI * ELECTRON_MASS * ELEMENTARY_CHARGE) * ICE_DENS
    return s


def sample_ice_abs_pot(acc_en: Union[float, torch.Tensor]) -> torch.Tensor:
    if isinstance(acc_en, float):
        acc_en = torch.tensor(acc_en)
    return PLANCKS_CONSTANT_SQ / (8. * PI * PI * ELECTRON_MASS * ELEMENTARY_CHARGE * POTENTIAL_UNIT) * wave_number(
        acc_en) / (1 + acc_en / ELEC_REST_ENERGY) * ICE_DENS * \
           (total_cross_section(acc_en, torch.tensor(8)) + 2. * total_cross_section(acc_en, torch.tensor(1)))


def sample_support_pot() -> torch.Tensor:
    a, _ = get_scattering_parameters(6)
    s = a.sum()
    s *= ONE_ANGSTROM
    s *= PLANCKS_CONSTANT_SQ / (2 * PI * ELECTRON_MASS * ELEMENTARY_CHARGE) * CARBON_DENS
    return s


def sample_support_abs_pot(acc_en: Union[float, torch.Tensor]) -> torch.Tensor:
    if isinstance(acc_en, float):
        acc_en = torch.tensor(acc_en)
    return PLANCKS_CONSTANT_SQ / (8. * PI * PI * ELECTRON_MASS * ELEMENTARY_CHARGE * POTENTIAL_UNIT) * wave_number(
        acc_en) / (1 + acc_en / ELEC_REST_ENERGY) * CARBON_DENS * total_cross_section(acc_en, torch.tensor(6))
