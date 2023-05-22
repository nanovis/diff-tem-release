import logging
from copy import deepcopy
from enum import Enum
from typing import Any, Optional, Union, List, Dict, Tuple

import mrcfile
import torch
from taichi._snode.snode_tree import SNodeTree

from . import UNINITIALIZED
from .constants.io_units import DET_PIXEL_UNIT
from .utils import BinaryFileFormat
from .utils.exceptions import ParameterLockedException, WrongArgumentTypeException, \
    RequiredParameterNotSpecifiedException
from .utils.helper_functions import check_tensor
from .utils.input_processing import convert_bool
from .utils.mrc import write_tensor_to_mrc
from .utils.parameter import Parameter
from .utils.tensor_io import ByteOrder
from .utils.tensor_ops import absolute
from .vector_valued_function import VectorValuedFunction
from .wavefunction import WaveFunction
import taichi as ti
from torch.distributions import Poisson

_logger = logging.getLogger(__name__)


def _write_log(string, *logging_strings):
    strings = [string, *logging_strings]
    if len(strings) > 1:
        logging_string = "\n    ".join(strings)
        _logger.debug(f"{_logger.name}:\n    {logging_string}", stacklevel=2)
    else:
        _logger.debug(f"{_logger.name}:  {strings[0]}", stacklevel=2)


class Detector:
    """
    The detector component simulates the formation of an image once the electron wave reaches the scintillator
    and writes the images to a file.
    One or several detectors with different properties can be used in the same simulation.
    """

    class Parameters(Enum):
        DET_PIX = Parameter("det_pix", int, description="The number of detector pixels in x/y directions")
        PADDING = Parameter("padding", int, description="Number of pixels of padding added "
                                                        "on each side of the detector in the internal computations.")
        PIXEL_SIZE = Parameter("pixel_size", float, description="Physical size of the detector pixels in micrometer.")
        GAIN = Parameter("gain", float,
                         description="The average number of counts produced by each electron striking the detector.")
        USE_QUANTIZATION = Parameter("use_quantization", bool,
                                     description="Controls whether the electron wave is "
                                                 "quantized into discrete electrons. "
                                                 "This is the primary source of noise in a real detector.")
        MTF_A = Parameter("mtf_a", float, description="One of the parameters controlling the "
                                                      "modulation transfer function of the detector.")
        MTF_B = Parameter("mtf_b", float, description="One of the parameters controlling the "
                                                      "modulation transfer function of the detector.")
        MTF_C = Parameter("mtf_c", float, description="One of the parameters controlling the "
                                                      "modulation transfer function of the detector.")
        MTF_ALPHA = Parameter("mtf_alpha", float, description="One of the parameters controlling "
                                                              "the modulation transfer function of the detector.")
        MTF_BETA = Parameter("mtf_beta", float, description="One of the parameters controlling "
                                                            "the modulation transfer function of the detector.")
        MTF_P = Parameter("mtf_p", float, description="One of the parameters controlling the "
                                                      "modulation transfer function of the detector.")
        MTF_Q = Parameter("mtf_q", float, description="One of the parameters controlling the "
                                                      "modulation transfer function of the detector.")
        IMAGE_FILE_OUT = Parameter("image_file_out", str, description="Name of the file to which images are written.")
        IMAGE_FILE_FORMAT = Parameter("image_file_format", BinaryFileFormat,
                                      description="File format to use for output images.")
        IMAGE_AXIS_ORDER = Parameter("image_axis_order", str, description="Controls the order in which "
                                                                          "pixel values are written in the output file. "
                                                                          "\"xy\" means that x is the "
                                                                          "fastest varying index.")
        IMAGE_FILE_BYTE_ORDER = Parameter("image_file_byte_order", ByteOrder,
                                          description="Controls if the output file should "
                                                      "be big endian or little endian.")

        def type(self):
            return self.value.type

        @staticmethod
        def parameter_names():
            return [p.value.name for p in Detector.Parameters]

    REQUIRED_PARAMETERS = [Parameters.DET_PIX, Parameters.PIXEL_SIZE, Parameters.GAIN, Parameters.USE_QUANTIZATION,
                           Parameters.MTF_A, Parameters.MTF_B, Parameters.MTF_C,
                           Parameters.MTF_ALPHA, Parameters.MTF_BETA,
                           Parameters.IMAGE_FILE_OUT]

    LEARNABLE_PARAMETER_NAMES = ["mtf_a", "mtf_b", "mtf_c", "mtf_alpha", "mtf_beta", "mtf_p", "mtf_q",
                                 "gain"]

    IMAGE_AXIS_ORDER_OPTIONS = ["xy", "yx"]

    PARAMETER_DEFAULTS = {
        Parameters.PADDING: 20,
        Parameters.MTF_P: 1.0,
        Parameters.MTF_Q: 1.0,
        Parameters.IMAGE_FILE_FORMAT: BinaryFileFormat.MRC,
        Parameters.IMAGE_AXIS_ORDER: IMAGE_AXIS_ORDER_OPTIONS[0],
        Parameters.IMAGE_FILE_BYTE_ORDER: ByteOrder.NATIVE
    }

    def __init__(self,
                 parameters_require_grad: bool = False,
                 device: Union[str, torch.device] = "cpu"):
        self.pixel: float = UNINITIALIZED
        parameters = {}
        for p in self.Parameters:
            if p in self.PARAMETER_DEFAULTS:
                parameters[p] = self.PARAMETER_DEFAULTS[p]
            else:
                parameters[p] = UNINITIALIZED
        if isinstance(device, str):
            device = torch.device(device)
        if parameters_require_grad:
            parameters[Detector.Parameters.MTF_P] = torch.tensor(parameters[Detector.Parameters.MTF_P],
                                                                 requires_grad=True, device=device)
            parameters[Detector.Parameters.MTF_Q] = torch.tensor(parameters[Detector.Parameters.MTF_Q],
                                                                 requires_grad=True, device=device)

        self.vvf_x_basis: torch.Tensor = UNINITIALIZED
        self.vvf_y_basis: torch.Tensor = UNINITIALIZED
        self.vvf_offset: torch.Tensor = UNINITIALIZED
        self.vvf_nx: int = UNINITIALIZED
        self.vvf_ny: int = UNINITIALIZED
        self._parameters: Dict[Detector.Parameters, Any] = parameters
        self._lock_parameters: bool = False
        self._initialized: bool = False
        self._parameters_requires_grad: bool = parameters_require_grad
        self._device: torch.device = device

    def to(self, device: torch.device):
        """
        Out-of-place move to a device, creating new copy of self whose data is on the device

        :param device: device that holds data
        :return: a new copy of self
        """
        assert self._initialized, "Initialized detector first"
        d = Detector(self._parameters_requires_grad, device=device)
        d.pixel = self.pixel
        for parameter in self._parameters:
            parameter_value = self._parameters[parameter]
            if isinstance(parameter_value, torch.Tensor):
                d._parameters[parameter] = parameter_value.clone().to(device)
            else:
                d._parameters[parameter] = deepcopy(parameter_value)
        d.vvf_x_basis = self.vvf_x_basis.to(device)
        d.vvf_y_basis = self.vvf_y_basis.to(device)
        d.vvf_offset = self.vvf_offset.to(device)
        d.vvf_nx = self.vvf_nx
        d.vvf_ny = self.vvf_ny
        d.det_pix = self.det_pix.to(device)
        d._lock_parameters = self._lock_parameters
        d._initialized = self._initialized
        return d

    def to_(self, device: torch.device):
        """
        In-place move to a device

        :param device: device that holds data
        """
        assert self._initialized, "Initialized detector first"
        self._lock_parameters = False
        self.det_pix = self.det_pix.to(device)
        for parameter in self._parameters:
            parameter_value = self._parameters[parameter]
            if isinstance(parameter_value, torch.Tensor):
                self._parameters[parameter] = parameter_value.to(device)
        self.vvf_x_basis = self.vvf_x_basis.to(device)
        self.vvf_y_basis = self.vvf_y_basis.to(device)
        self.vvf_offset = self.vvf_offset.to(device)
        self._lock_parameters = True

    @staticmethod
    def prototype_from_parameters(parameters: Union[List[Dict[str, str]], Dict[str, str]]):
        """
        Construct **uninitialized** instances of Detector given parameters

        :param parameters: list of parameter dictionaries or a dictionary containing parameters
        :return: list of instances or one instance
        """
        if isinstance(parameters, list):
            return [Detector._from_parameters(parameter_table) for parameter_table in parameters]
        elif isinstance(parameters, dict):
            return Detector._from_parameters(parameters)
        else:
            raise Exception(f"Invalid parameter of type {type(parameters)}")

    @staticmethod
    def _from_parameters(parameters: Dict[str, str]):
        detector = Detector()
        if "det_pix_x" in parameters and "det_pix_y" in parameters:
            det_pix_x = int(parameters["det_pix_x"])
            det_pix_y = int(parameters["det_pix_y"])
            detector.det_pix = torch.tensor([det_pix_x, det_pix_y], dtype=torch.int)
        for p in Detector.Parameters:
            if p != Detector.Parameters.DET_PIX:
                parameter_name = p.value.name
                parameter_type = p.type()
                if parameter_type == bool:
                    parameter_type = convert_bool
                if parameter_name in parameters:
                    parameter_val = parameter_type(parameters[parameter_name])
                    setattr(detector, parameter_name, parameter_val)
        return detector

    def finish_init_(self, simulation):
        """
        Finish initialization of Detector

        :param simulation: Simulation instance
        :return: self
        """
        # CORRESPOND: detector_init_share
        if self._initialized:
            return self
        # check parameters, CORRESPOND: detector_check_input
        for parameter_type in self.REQUIRED_PARAMETERS:
            if self._parameters[parameter_type] is None:
                raise RequiredParameterNotSpecifiedException(parameter_type)
        pixel_size = self.pixel_size_nanometer
        self.vvf_x_basis = torch.tensor([pixel_size, 0.], device=self._device)
        self.vvf_y_basis = torch.tensor([0., pixel_size], device=self._device)
        self.vvf_offset = torch.zeros(2, device=self._device)
        padding = self.padding
        det_pix = self.det_pix
        self.vvf_nx = det_pix[0] + 2 * padding
        self.vvf_ny = det_pix[1] + 2 * padding
        self.create_image_file(simulation)
        self._initialized = True
        self._lock_parameters = True
        _write_log("Detector component initialized.")
        return self

    def create_vvf(self, count: Optional[torch.Tensor] = None) -> VectorValuedFunction:
        """
        Create a VectorValuedFunction based on parameters of the Detector

        :param count: optional values in VVF
        :return: a new VectorValuedFunction
        """
        assert self._initialized
        nx = self.vvf_nx
        ny = self.vvf_ny
        if count is None:
            count = torch.zeros(ny, nx, device=self._device, dtype=torch.cdouble)
        else:
            check_tensor(count, shape=(ny, nx), device=self._device, dtype=torch.cdouble)
        values = count
        vvf = VectorValuedFunction(values, self.vvf_x_basis.clone(), self.vvf_y_basis.clone(), self.vvf_offset.clone())
        return vvf

    def create_image_file(self, simulation):
        # CORRESPOND: detector_create_image_file
        filepath = self.image_file_out
        format = self.image_file_format
        geometry = None
        if format == BinaryFileFormat.MRC or format == BinaryFileFormat.MRC_INT:
            optics = simulation.optics
            assert optics is not None, "Optics required for initialization of detector.\n"
            geometry = simulation.geometry
            assert geometry is not None, "Geometry required for initialization of detector.\n"
            geometry.finish_init_()
            optics.finish_init_(geometry.ntilts)
        i = 1 if self.image_axis_order == self.IMAGE_AXIS_ORDER_OPTIONS[1] else 0
        with mrcfile.new(filepath, overwrite=True) as mrc:
            det_pix = self.det_pix
            if i == 0:
                mrc.header.nx = det_pix[0]
                mrc.header.ny = det_pix[1]
            else:
                mrc.header.ny = det_pix[0]
                mrc.header.nx = det_pix[1]
            mrc.header.nz = 0
            if format == BinaryFileFormat.MRC or BinaryFileFormat.MRC_INT:
                if geometry.gen_tilt_data:
                    data = geometry.data.numpy()
                    tilt_data_file_path = filepath + ".tilt"
                    data.tofile(tilt_data_file_path, sep="\n")

    def write_image(self, vvf_values_in_tilt_series: List[torch.Tensor]):
        """
        Add an output image to the image file and update file header if necessary.

        :param vvf_values_in_tilt_series: values of vector valued functions of a tilt series
        """
        # CORRESPOND: detector_write_image
        assert self._initialized, "Detector not initialized"
        image_format = self.image_file_format

        if image_format == BinaryFileFormat.MRC_INT:
            data_type = torch.int16
        elif image_format == BinaryFileFormat.MRC:
            data_type = torch.float32
        else:
            data_type = None  # default
        values = torch.stack(vvf_values_in_tilt_series, dim=0)
        # TODO: check axis order and order of values
        # values shape: nz, ny, nx
        values = values[:, self.padding: self.det_pix[1] + self.padding, self.padding:self.det_pix[0] + self.padding]
        write_tensor_to_mrc(values.real.detach(), self.image_file_out,
                            as_type=data_type,
                            axis_order=self.image_axis_order + "z")

    @staticmethod
    def apply_quantization(vvf_values: torch.Tensor) -> torch.Tensor:
        """
        Apply quantization noise to the detector image being processed.

        :param vvf_values: values of input VectorValuedFunction
        :return: a tensor containing new values of a new VectorValuedFunction(because other infor in new VVF is useless)
        """
        # CORRESPOND: detector_apply_quantization
        check_tensor(vvf_values, dimensionality=2)
        vvf_values = torch.complex(torch.poisson(vvf_values.real), vvf_values.imag)
        return vvf_values

    @staticmethod
    def apply_quantization_differentiable(vvf_values: torch.Tensor, sample_num: int) \
            -> Tuple[torch.Tensor, torch.Tensor, Poisson]:
        """
        Apply quantization noise to the detector image being processed.

        :param vvf_values: values of input VectorValuedFunction, containing VVF values of shape (x, y)
        :param sample_num: number of samples
        :return: a tensor containing new values of a new VectorValuedFunction(because other infor in new VVF is useless)
        shape = (sample_num, x, y), log probabilities and Poisson instance
        """
        assert sample_num > 0, "sample_num must > 0"
        check_tensor(vvf_values, dimensionality=2)
        poisson = Poisson(torch.abs(vvf_values.real))
        new_vvf_values_real = poisson.sample((sample_num,))
        new_vvf_values_imag = torch.stack([vvf_values.imag] * sample_num)
        log_probs = poisson.log_prob(new_vvf_values_real)
        vvf_values = torch.complex(new_vvf_values_real, new_vvf_values_imag)
        return vvf_values, log_probs, poisson

    @staticmethod
    def value_to_device(x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        else:
            return x

    def apply_mtf(self, vvf_values: torch.Tensor) -> torch.Tensor:
        """
        Apply modulation transfer function (detector blurring) to the detector image being processed.

        :param vvf_values: the tensor of a VectorValuedFunction
        :return: a new tensor of a new VectorValuedFunction(because other infor in new VVF is useless)
        """
        # CORRESPOND: detector_apply_mtf
        assert self._initialized
        check_tensor(vvf_values, dimensionality=2)
        device = vvf_values.device
        m = vvf_values.shape[1]  # nx
        n = vvf_values.shape[0]  # ny
        dxi = 1. / m
        dxj = 1. / n
        gain = self.value_to_device(self.gain, device)
        mtf_a = self.value_to_device(self.mtf_a, device)
        mtf_b = self.value_to_device(self.mtf_b, device)
        mtf_c = self.value_to_device(self.mtf_c, device)
        mtf_p = self.value_to_device(self.mtf_p, device)
        mtf_q = self.value_to_device(self.mtf_q, device)
        mtf_alpha = self.value_to_device(self.mtf_alpha, device)
        mtf_beta = self.value_to_device(self.mtf_beta, device)

        g = gain / (mtf_a + mtf_b + mtf_c) / (m * n)
        two = torch.tensor(2., device=device)
        two_p = torch.pow(two, mtf_p)
        two_q = torch.pow(two, mtf_q)

        count_ft = torch.fft.rfft2(vvf_values.real, norm="backward")
        j = torch.arange(count_ft.shape[0], device=device)
        j = torch.minimum(j, n - j)  # (count_ft.shape[0], )
        xj = dxj * j
        i = torch.arange(count_ft.shape[1], device=device)
        i = torch.minimum(i, m - i)  # (count_ft.shape[1], )
        xi = dxi * i

        xj = xj.view(-1, 1)  # (count_ft.shape[0], 1)
        xi = xi.view(1, -1)  # (1, count_ft.shape[1])
        x2 = xi * xi + xj * xj  # (count_ft.shape[0], count_ft.shape[1])

        y = 1. + mtf_alpha * x2  # (count_ft.shape[0], count_ft.shape[1])
        y_pow_minus_p = torch.pow(y, -mtf_p)
        mtf = mtf_c + mtf_a * two_p * y_pow_minus_p / (2. + (two_p - 2.) * y_pow_minus_p)

        y = 1. + mtf_beta * x2  # (count_ft.shape[0], count_ft.shape[1])
        y_pow_minus_q = torch.pow(y, -mtf_q)
        mtf = mtf_b * two_q * y_pow_minus_q / (2. + (two_q - 2.) * y_pow_minus_q) + mtf
        mtf = torch.abs(g * mtf)

        count_ft = torch.complex(count_ft.real * mtf, count_ft.imag * mtf)
        new_vvf_values = torch.complex(torch.fft.irfft2(count_ft, norm="forward"), vvf_values.imag)
        _write_log(f"apply_mtf grad_fn = {new_vvf_values.grad_fn}")
        return new_vvf_values

    def add_intensity_from_wave_function_to_vvf(self,
                                                vvf: VectorValuedFunction,
                                                wf: WaveFunction,
                                                tilt: int) -> Union[VectorValuedFunction, List[SNodeTree]]:
        """
        Get electron wave intensity in the detector plane from wave function object.
        """
        # CORRESPOND: detector_get_intensity
        assert self._initialized
        assert 0 <= tilt < wf.electronbeam.doses.shape[0]
        dose_per_pixel = (self.pixel_size_nanometer / wf.optics.magnification) ** 2
        dose_per_pixel = wf.electronbeam.doses[tilt] * dose_per_pixel
        snode_trees = []
        new_vvf = vvf.spawn_with(vvf.values.clone())
        task = new_vvf.add_(wf.intens)
        virtual_fields = task.send(None)
        concrete_fields = []
        if virtual_fields is not None:
            field_builder = ti.FieldsBuilder()
            for vf in virtual_fields:
                concrete_fields.append(vf.concretize(field_builder))
            snode1 = field_builder.finalize()
            snode_trees.append(snode1)

        virtual_fields = task.send(concrete_fields)
        concrete_fields = []
        if virtual_fields is not None:
            field_builder = ti.FieldsBuilder()
            for vf in virtual_fields:
                concrete_fields.append(vf.concretize(field_builder))
            snode2 = field_builder.finalize()
            snode_trees.append(snode2)

        try:
            task.send(concrete_fields)
        except StopIteration:
            _write_log("finish")
            pass

        new_vvf.values = torch.complex(new_vvf.values.real * dose_per_pixel,
                                       new_vvf.values.imag * dose_per_pixel)
        _write_log(f'count.values Grad add_intensity_from_wave_function_ = {new_vvf.values.grad_fn}')
        return new_vvf, snode_trees

    def log_parameters(self, logger: logging.Logger):
        logger.info(f"Detector:\n   {self._parameters}")

    def reset(self):
        """
        Reset detector's values
        """
        if self._initialized:
            self._lock_parameters = False
            self._initialized = False

    def _unwrap_required_parameter(self, parameter: Parameters) -> Any:
        parameter_value = self._parameters[parameter]
        if parameter_value is None:
            raise RequiredParameterNotSpecifiedException(parameter)
        return parameter_value

    @property
    def device(self):
        assert self._initialized, "Initialized detector first"
        return self._device

    @property
    def pixel_size_nanometer(self) -> float:
        """
        Physical size of the detector pixels in nanometer.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return DET_PIXEL_UNIT * self.pixel_size

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def det_pix(self) -> torch.Tensor:
        """
        The number of detector pixels in x/y directions

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.DET_PIX)

    @property
    def padding(self) -> int:
        """
        Number of pixels of padding added on each side of the detector in the internal computations.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.PADDING)

    @property
    def pixel_size(self) -> float:
        """
        Physical size of the detector pixels in micrometer.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.PIXEL_SIZE)

    @property
    def gain(self) -> Union[float, torch.Tensor]:
        """
        The average number of counts produced by each electron striking the detector.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return absolute(self._unwrap_required_parameter(Detector.Parameters.GAIN))

    @property
    def _gain(self) -> Union[float, torch.Tensor]:
        """
        The average number of counts produced by each electron striking the detector.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.GAIN)

    @property
    def use_quantization(self) -> bool:
        """
        Controls whether the electron wave is quantized into discrete electrons.
        This is the primary source of noise in a real detector.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.USE_QUANTIZATION)

    @property
    def mtf_a(self) -> Union[float, torch.Tensor]:
        """
        One of the parameters controlling the modulation transfer function of the detector.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.MTF_A)

    @property
    def _mtf_a(self) -> Union[float, torch.Tensor]:
        """
        One of the parameters controlling the modulation transfer function of the detector.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.MTF_A)

    @property
    def mtf_b(self) -> Union[float, torch.Tensor]:
        """
        One of the parameters controlling the modulation transfer function of the detector.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.MTF_B)

    @property
    def _mtf_b(self) -> Union[float, torch.Tensor]:
        """
        One of the parameters controlling the modulation transfer function of the detector.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.MTF_B)

    @property
    def mtf_c(self) -> Union[float, torch.Tensor]:
        """
        One of the parameters controlling the modulation transfer function of the detector.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.MTF_C)

    @property
    def _mtf_c(self) -> Union[float, torch.Tensor]:
        """
        One of the parameters controlling the modulation transfer function of the detector.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.MTF_C)

    @property
    def mtf_alpha(self) -> Union[float, torch.Tensor]:
        """
        One of the parameters controlling the modulation transfer function of the detector.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.MTF_ALPHA)

    @property
    def _mtf_alpha(self) -> Union[float, torch.Tensor]:
        """
        One of the parameters controlling the modulation transfer function of the detector.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.MTF_ALPHA)

    @property
    def mtf_beta(self) -> Union[float, torch.Tensor]:
        """
        One of the parameters controlling the modulation transfer function of the detector.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.MTF_BETA)

    @property
    def _mtf_beta(self) -> Union[float, torch.Tensor]:
        """
        One of the parameters controlling the modulation transfer function of the detector.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.MTF_BETA)

    @property
    def mtf_p(self) -> Union[float, torch.Tensor]:
        """
        One of the parameters controlling the modulation transfer function of the detector.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return absolute(self._unwrap_required_parameter(Detector.Parameters.MTF_P))

    @property
    def _mtf_p(self) -> Union[float, torch.Tensor]:
        """
        One of the parameters controlling the modulation transfer function of the detector.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.MTF_P)

    @property
    def mtf_q(self) -> Union[float, torch.Tensor]:
        """
        One of the parameters controlling the modulation transfer function of the detector.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return absolute(self._unwrap_required_parameter(Detector.Parameters.MTF_Q))

    @property
    def _mtf_q(self) -> Union[float, torch.Tensor]:
        """
        One of the parameters controlling the modulation transfer function of the detector.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.MTF_Q)

    @property
    def image_file_out(self) -> str:
        """
        Name of the file to which images are written.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.IMAGE_FILE_OUT)

    @property
    def image_file_format(self) -> BinaryFileFormat:
        """
        File format to use for output images.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.IMAGE_FILE_FORMAT)

    @property
    def image_axis_order(self) -> str:
        """
        Controls the order in which pixel values are written in the output file.
        \"xy\" means that x is the fastest varying index.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.IMAGE_AXIS_ORDER)

    @property
    def image_file_byte_order(self) -> ByteOrder:
        """
        Controls if the output file should be big endian or little endian.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Detector.Parameters.IMAGE_FILE_BYTE_ORDER)

    @property
    def parameters_require_grad(self) -> bool:
        """
        Whether the parameters of Detector require gradients
        :return: bool
        """
        return self._parameters_requires_grad

    def _set_parameter(self, parameter: Parameters, value):
        if isinstance(value, parameter.type()):
            if self._lock_parameters:
                raise ParameterLockedException(parameter)
            else:
                self._parameters[parameter] = value
        else:
            raise WrongArgumentTypeException(value, parameter.type())

    @det_pix.setter
    def det_pix(self, det_pix: torch.Tensor):
        check_tensor(det_pix, dimensionality=1, shape=(2,), dtype=torch.int)
        assert torch.all(det_pix > 0)
        self._parameters[self.Parameters.DET_PIX] = torch.clone(det_pix)

    @padding.setter
    def padding(self, padding: int):
        self._set_parameter(self.Parameters.PADDING, padding)
        assert padding > 0

    @pixel_size.setter
    def pixel_size(self, pixel_size: float):
        self._set_parameter(self.Parameters.PIXEL_SIZE, pixel_size)
        assert pixel_size > 0.

    @gain.setter
    def gain(self, gain: Union[float, torch.Tensor]):
        assert gain > 0., f"Gain must > 0., got {gain}"
        parameter = Detector.Parameters.GAIN
        if self._parameters_requires_grad:
            if isinstance(gain, float):
                gain = torch.tensor(gain, requires_grad=True)
            elif isinstance(gain, torch.Tensor):
                gain = gain.clone()
                gain.requires_grad_(True)
            else:
                raise Exception(f"Not support type {type(gain)}")
            self._parameters[parameter] = gain
        else:
            self._set_parameter(parameter, gain)

    @use_quantization.setter
    def use_quantization(self, use_quantization: bool):
        self._set_parameter(self.Parameters.USE_QUANTIZATION, use_quantization)

    @mtf_a.setter
    def mtf_a(self, mtf_a: Union[float, torch.Tensor]):
        parameter = Detector.Parameters.MTF_A
        if self._parameters_requires_grad:
            if isinstance(mtf_a, float):
                mtf_a = torch.tensor(mtf_a, requires_grad=True)
            elif isinstance(mtf_a, torch.Tensor):
                check_tensor(mtf_a, dimensionality=0)
                mtf_a = mtf_a.clone()
                mtf_a.requires_grad_(True)
            else:
                raise Exception(f"Not support type {type(mtf_a)}")
            self._parameters[parameter] = mtf_a
        else:
            self._set_parameter(parameter, mtf_a)

    @mtf_b.setter
    def mtf_b(self, mtf_b: Union[float, torch.Tensor]):
        parameter = Detector.Parameters.MTF_B
        if self._parameters_requires_grad:
            if isinstance(mtf_b, float):
                mtf_b = torch.tensor(mtf_b, requires_grad=True)
            elif isinstance(mtf_b, torch.Tensor):
                check_tensor(mtf_b, dimensionality=0)
                mtf_b = mtf_b.clone()
                mtf_b.requires_grad_(True)
            else:
                raise Exception(f"Not support type {type(mtf_b)}")
            self._parameters[parameter] = mtf_b
        else:
            self._set_parameter(parameter, mtf_b)

    @mtf_c.setter
    def mtf_c(self, mtf_c: Union[float, torch.Tensor]):
        parameter = Detector.Parameters.MTF_C
        if self._parameters_requires_grad:
            if isinstance(mtf_c, float):
                mtf_c = torch.tensor(mtf_c, requires_grad=True)
            elif isinstance(mtf_c, torch.Tensor):
                check_tensor(mtf_c, dimensionality=0)
                mtf_c = mtf_c.clone()
                mtf_c.requires_grad_(True)
            else:
                raise Exception(f"Not support type {type(mtf_c)}")
            self._parameters[parameter] = mtf_c
        else:
            self._set_parameter(parameter, mtf_c)

    @mtf_alpha.setter
    def mtf_alpha(self, mtf_alpha: Union[float, torch.Tensor]):
        assert mtf_alpha > 0.
        parameter = Detector.Parameters.MTF_ALPHA
        if self._parameters_requires_grad:
            if isinstance(mtf_alpha, float):
                mtf_alpha = torch.tensor(mtf_alpha, requires_grad=True)
            elif isinstance(mtf_alpha, torch.Tensor):
                check_tensor(mtf_alpha, dimensionality=0)
                mtf_alpha = mtf_alpha.clone()
                mtf_alpha.requires_grad_(True)
            else:
                raise Exception(f"Not support type {type(mtf_alpha)}")
            self._parameters[parameter] = mtf_alpha
        else:
            self._set_parameter(parameter, mtf_alpha)

    @mtf_beta.setter
    def mtf_beta(self, mtf_beta: Union[float, torch.Tensor]):
        assert mtf_beta > 0.
        parameter = Detector.Parameters.MTF_BETA
        if self._parameters_requires_grad:
            if isinstance(mtf_beta, float):
                mtf_beta = torch.tensor(mtf_beta, requires_grad=True)
            elif isinstance(mtf_beta, torch.Tensor):
                check_tensor(mtf_beta, dimensionality=0)
                mtf_beta = mtf_beta.clone()
                mtf_beta.requires_grad_(True)
            else:
                raise Exception(f"Not support type {type(mtf_beta)}")
            self._parameters[parameter] = mtf_beta
        else:
            self._set_parameter(parameter, mtf_beta)

    @mtf_p.setter
    def mtf_p(self, mtf_p: Union[float, torch.Tensor]):
        assert 0. < mtf_p < 10.
        parameter = Detector.Parameters.MTF_P
        if self._parameters_requires_grad:
            if isinstance(mtf_p, float):
                mtf_p = torch.tensor(mtf_p, requires_grad=True)
            elif isinstance(mtf_p, torch.Tensor):
                check_tensor(mtf_p, dimensionality=0)
                mtf_p = mtf_p.clone()
                mtf_p.requires_grad_(True)
            else:
                raise Exception(f"Not support type {type(mtf_p)}")
            self._parameters[parameter] = mtf_p
        else:
            self._set_parameter(parameter, mtf_p)

    @mtf_q.setter
    def mtf_q(self, mtf_q: Union[float, torch.Tensor]):
        assert 0. < mtf_q < 10.
        parameter = Detector.Parameters.MTF_Q
        if self._parameters_requires_grad:
            if isinstance(mtf_q, float):
                mtf_q = torch.tensor(mtf_q, requires_grad=True)
            elif isinstance(mtf_q, torch.Tensor):
                check_tensor(mtf_q, dimensionality=0)
                mtf_q = mtf_q.clone()
                mtf_q.requires_grad_(True)
            else:
                raise Exception(f"Not support type {type(mtf_q)}")
            self._parameters[parameter] = mtf_q
        else:
            self._set_parameter(parameter, mtf_q)

    @image_file_out.setter
    def image_file_out(self, image_file_out: str):
        self._set_parameter(self.Parameters.IMAGE_FILE_OUT, image_file_out)

    @image_file_format.setter
    def image_file_format(self, image_file_format: str):
        try:
            self._set_parameter(self.Parameters.IMAGE_FILE_FORMAT, BinaryFileFormat(image_file_format))
        except ValueError:
            raise Exception(f"Invalid {image_file_format=}, valid options = {BinaryFileFormat.options()}")

    @image_axis_order.setter
    def image_axis_order(self, image_axis_order: str):
        self._set_parameter(self.Parameters.IMAGE_AXIS_ORDER, image_axis_order)
        assert image_axis_order in self.IMAGE_AXIS_ORDER_OPTIONS

    @image_file_byte_order.setter
    def image_file_byte_order(self, image_file_byte_order: str):
        try:
            self._set_parameter(self.Parameters.IMAGE_FILE_BYTE_ORDER, ByteOrder(image_file_byte_order))
        except ValueError:
            raise Exception(f"Invalid image_file_byte_order = {image_file_byte_order}, "
                            f"valid options = {ByteOrder.options()}")

    @parameters_require_grad.setter
    def parameters_require_grad(self, require_grad: bool):
        assert isinstance(require_grad, bool)
        lock = self._lock_parameters
        self._lock_parameters = False
        if self._parameters_requires_grad and not require_grad:
            self._parameters_requires_grad = False
            for parameter_name in self._parameters:
                value = self._parameters[parameter_name]
                if isinstance(value, torch.Tensor):
                    name = parameter_name.value.name
                    value.requires_grad_(False)
                    setattr(self, name, value.item())  # use setter to convert to float
        elif not self._parameters_requires_grad and require_grad:
            self._parameters_requires_grad = True
            for parameter_name in self._parameters:
                name = parameter_name.value.name
                value = self._parameters[parameter_name]
                setattr(self, name, value)  # use setter to convert to tensor
        else:
            pass
        self._lock_parameters = lock
