from copy import deepcopy
from typing import Union, Any, Tuple, List, Dict

import torch
import logging
from enum import Enum
from . import UNINITIALIZED
from .constants import PI
from .constants.io_units import VOL_CONC_UNIT, SURF_CONC_UNIT
from .utils.conversion import radian_to_degree, degree_to_radian
from .utils.exceptions import RequiredParameterNotSpecifiedException, ParameterLockedException, \
    WrongArgumentTypeException
from .utils.input_processing import convert_bool
from .utils.tensor_io import read_matrix_text, write_matrix_text_with_headers
from .utils.parameter import Parameter
from .utils.helper_functions import check_tensor, eprint
from .utils.transformations import rotate_3d_rows, rand_orientation

_logger = logging.getLogger(__name__)


def _write_log(string, *logging_strings):
    strings = [string, *logging_strings]
    if len(strings) > 1:
        logging_string = "\n    ".join(strings)
        _logger.debug(f"{_logger.name}:\n    {logging_string}", stacklevel=2)
    else:
        _logger.debug(f"{_logger.name}:  {strings[0]}", stacklevel=2)


class Particleset:
    """
    A particleset component specifies the locations of one or several identical particles in the sample.
    The particle itself is defined by a particle component. A simulation of micrographs can use one or several
    particleset components, or possibly none at all if an empty sample is to be simulated.
    """

    class Parameters(Enum):
        PARTICLE_TYPE = Parameter("particle_type", str,
                                  description="The name of the particle component used in the particleset.")
        PARTICLE_COORDS = Parameter("particle_coords", str,
                                    description="Specifies how the particle coordinates should be generated. "
                                                "\"file\" means that positions and orientations are read from a text file. "
                                                "\"rand\" means that positions and orientations are randomly generated. "
                                                "\"grid\" means that the positions are on a regular grid, and the orientations are random.")
        WHERE = Parameter("where", str,
                          description="Possible locations of randomly placed particles. "
                                      "Available choices are the entire sample volume, "
                                      "near both surfaces of the sample, "
                                      "near the top surface, or near the bottom surface.")
        NUM_PARTICLES = Parameter("num_particles", int,
                                  description="When generating random particle positions, "
                                              "one of the parameters num_particles, particle_conc, or occupancy "
                                              "must be defined. "
                                              "num_particles defines the number of particles to place in the sample. "
                                              "The parameter is optional if positions are read from a file, "
                                              "and defines the number of particle positions in the file to use.")
        PARTICLE_CONC = Parameter("particle_conc", float,
                                  description="When generating random particle positions, "
                                              "one of the parameters num_particles, "
                                              "particle_conc, or occupancy must be defined. "
                                              "particle_conc defines the number of particles per cubic micrometer "
                                              "of the sample if \"where\" = \"volume\", "
                                              "or the number of particles per square micrometer "
                                              "on the surface otherwise.")
        OCCUPANCY = Parameter("occupancy", float,
                              description="When generating random particle positions, "
                                          "one of the parameters num_particles, "
                                          "particle_conc, or occupancy must be defined. "
                                          "occupancy defines the fraction of the sample volume occupied by particles "
                                          "if where = \"volume\", "
                                          "or the fraction of the sample surface occupied in other cases.")
        N = Parameter("nx,ny,nz", float,
                      description="The size of the grid in the x/y/z axis direction "
                                  "when particle positions are on a regular grid.")
        PARTICLE_DIST = Parameter("particle_dist", float,
                                  description="The distance in nanometer "
                                              "between the centers of adjacent particles "
                                              "when particle positions are on a regular grid.")
        COORD_FILE_IN = Parameter("coord_file_in", str,
                                  description="The name of a text file "
                                              "from which the positions of particles are read "
                                              "if they are not randomly generated. "
                                              "See the manual for details about the file format.")
        COORD_FILE_OUT = Parameter("coord_file_out", str,
                                   description="The name of a text file "
                                               "which randomly generated particle positions are written to. "
                                               "See the manual for details about the file format.")
        OFFSET = Parameter("offset", float,
                           description="The x/y/z coordinate of the grid center "
                                       "when particle positions are on a regular grid.")

        def type(self):
            return self.value.type

    PARTICLE_COORD_OPTIONS = ["rand", "file", "grid"]
    WHERE_OPTIONS = ["volume", "surface", "top", "bottom"]

    REQUIRED_PARAMETERS = [Parameters.PARTICLE_TYPE, Parameters.PARTICLE_COORDS]
    LEARNABLE_PARAMETER_NAMES = ["coordinates", ]  # possibly offset and particle dist?

    def __init__(self):
        self.coordinates: torch.Tensor = UNINITIALIZED
        self._initialized: bool = False
        self._lock_parameters: bool = False
        self._parameters = {parameter_type: None for parameter_type in self.Parameters}
        self.offset = torch.zeros(3, dtype=torch.double)  # Default

    def to(self, device: torch.device):
        """
        Out-of-place move to a device, creating new copy of self whose data is on the device

        :param device: device that holds data
        :return: a new copy of self
        """
        assert self._initialized, "Initialize particle set first"
        particle_set = Particleset()
        particle_set.coordinates = self.coordinates.to(device)
        particle_set._initialized = self._initialized
        particle_set._parameters = deepcopy(self._parameters)
        particle_set.offset = self.offset.to(device)
        particle_set.n = self.n.to(device)
        particle_set._lock_parameters = self._lock_parameters
        return particle_set

    def to_(self, device: torch.device):
        """
        In-place move to a device

        :param device: device that holds data
        """
        assert self._initialized, "Initialize particle set first"
        self._lock_parameters = False
        self.offset = self.offset.to(device)
        self.n = self.n.to(device)
        self._lock_parameters = True

    @staticmethod
    def prototype_from_parameters(parameters: Union[List[Dict[str, str]], Dict[str, str]]):
        """
        Construct **uninitialized** instances of Particleset given parameters

        :param parameters: list of parameter dictionaries or a dictionary containing parameters
        :return: list of instances or one instance
        """
        if isinstance(parameters, list):
            return [Particleset._from_parameters(parameter_table) for parameter_table in parameters]
        elif isinstance(parameters, dict):
            return Particleset._from_parameters(parameters)
        else:
            raise Exception(f"Invalid parameter of type {type(parameters)}")

    @staticmethod
    def _from_parameters(parameters: Dict[str, str]):
        particle_set = Particleset()
        if "offset_x" in parameters and "offset_y" in parameters and "offset_z" in parameters:
            offset_x = float(parameters["offset_x"])
            offset_y = float(parameters["offset_y"])
            offset_z = float(parameters["offset_z"])
            particle_set.offset = torch.tensor([offset_x, offset_y, offset_z])
        if "nx" in parameters and "ny" in parameters and "nz" in parameters:
            nx = int(parameters["nx"])
            ny = int(parameters["ny"])
            nz = int(parameters["nz"])
            particle_set.n = torch.tensor([nx, ny, nz])
        for p in Particleset.Parameters:
            if p != Particleset.Parameters.OFFSET and p != Particleset.Parameters.N:
                parameter_name = p.value.name
                parameter_type = p.type()
                if parameter_type == bool:
                    parameter_type = convert_bool
                if parameter_name in parameters:
                    parameter_val = parameter_type(parameters[parameter_name])
                    setattr(particle_set, parameter_name, parameter_val)
        return particle_set

    def finish_init_(self, simulation):
        """
        Finish initialization of Particleset

        :param simulation: Simulation instance
        :return: self
        """
        if self._initialized:
            return self
        else:
            for parameter_type in self.REQUIRED_PARAMETERS:
                if self._parameters[parameter_type] is None:
                    raise RequiredParameterNotSpecifiedException(parameter_type)

            particle_coords = self.particle_coords
            if particle_coords == self.PARTICLE_COORD_OPTIONS[0]:  # rand
                IN_VOLUME = self.WHERE_OPTIONS[0]
                ON_SURFACES = self.WHERE_OPTIONS[1]
                ON_TOP_SURFACE = self.WHERE_OPTIONS[2]
                particle = simulation.get_particle(self.particle_type)
                if particle is None:
                    raise Exception(f"Particle {particle} not found")
                particle.finish_init_(simulation)
                sample = simulation.sample
                if sample is None:
                    raise Exception("Sample required for init particleset")
                where = self.where
                thickness_center = sample.thickness_center
                thickness_edge = sample.thickness_edge
                diameter = sample.diameter
                offset_x = sample.offset_x
                offset_y = sample.offset_y
                voxel_size = particle.voxel_size
                pot_re_shape = particle.pot_re.shape
                num_particles = self._parameters[self.Parameters.NUM_PARTICLES]
                particle_conc = self._parameters[self.Parameters.PARTICLE_CONC]
                occupancy = self._parameters[self.Parameters.OCCUPANCY]
                # estimate of particle radius
                r = (pot_re_shape[0] + pot_re_shape[1] + pot_re_shape[2]) * voxel_size / 6.0
                if num_particles is not None:
                    num = num_particles
                elif particle_conc is not None:
                    if where == IN_VOLUME:
                        vol = 0.125 * PI * diameter * diameter
                        num = round(vol * VOL_CONC_UNIT * particle_conc)
                    else:
                        area = 0.25 * PI * diameter * diameter
                        if where == ON_SURFACES:
                            area *= 2
                        num = round(area * SURF_CONC_UNIT * particle_conc)
                elif occupancy is not None:
                    if where == IN_VOLUME:
                        vol = 0.125 * PI * diameter * diameter * (thickness_edge + thickness_center)
                        num = round(vol * 3 / (4 * PI * r * r * r) * occupancy)
                    else:
                        area = 0.25 * PI * diameter * diameter
                        if where == ON_SURFACES:
                            area *= 2
                        num = round(area / (PI * r * r) * occupancy)
                else:
                    raise Exception(f"To generate random particle coordinates, one of the parameters "
                                    f"{self.Parameters.NUM_PARTICLES}, "
                                    f"{self.Parameters.PARTICLE_CONC} or "
                                    f"{self.Parameters.OCCUPANCY} must be specified")

                if 2. * r > min(thickness_center, diameter):
                    eprint("Error in generate_random_coordinates: particle to big to fit in sample")
                rad = 0.5 * diameter - r
                rad2 = rad * rad
                a = 2 * (thickness_edge - thickness_center) / (diameter * diameter)
                b = 0.5 * thickness_center - r
                zmax = a * rad2 + b
                xy = torch.rand(num, 2) * 2. * rad - rad
                zs = a * (xy ** 2).sum(dim=-1) + b
                if where == IN_VOLUME:
                    z = torch.rand(num, 1) * 2. * zmax - zmax
                elif where == ON_SURFACES:
                    z = torch.bernoulli(torch.ones(num, 1) * 0.5) * 2.0 - 1.0  # random 1 or -1
                    z = z * zs
                elif where == ON_TOP_SURFACE:
                    z = -zs
                else:  # ON_BOTTOM_SURFACE
                    z = zs

                for i in range(num):
                    check_pass = False
                    while not check_pass:
                        u = (xy[i] ** 2).sum()
                        if u >= rad2:  # if out of bound, generate new one
                            xy[i] = torch.rand(2) * 2. * rad - rad
                            continue
                        zs = a * u + b
                        if where == IN_VOLUME:
                            z_val = z[i]
                            if abs(z_val) > zs:
                                xy[i] = torch.rand(2) * 2. * rad - rad
                                z[i] = z = torch.rand(1) * 2. * zmax - zmax
                                continue
                        elif where == ON_SURFACES:
                            z[i] = torch.sign(z[i]) * zs
                        elif where == ON_TOP_SURFACE:
                            z[i] = -zs
                        else:  # ON_BOTTOM_SURFACE
                            z[i] = zs
                        check_pass = True

                xy += torch.tensor([offset_x, offset_y]).reshape(1, 2)
                random_orientations = rand_orientation(num)
                self.coordinates = torch.cat([xy, z, random_orientations], dim=-1)

            elif particle_coords == self.PARTICLE_COORD_OPTIONS[1]:  # file
                # CORRESPOND: particleset_read_coord
                coords = read_matrix_text(self.coord_file_in)
                coords[:, 3:] = degree_to_radian(coords[:, 3:])
                num_particles = self._parameters[self.Parameters.NUM_PARTICLES]
                if num_particles is not None:
                    assert len(coords) == num_particles
                self.coordinates = coords

            else:  # grid
                n = self.n
                x = n[0]
                y = n[1]
                z = n[2]
                offset = self.offset.reshape(1, 3)
                x_indices = torch.arange(x).reshape(x, 1, 1).repeat(1, y, z)
                y_indices = torch.arange(y).reshape(1, y, 1).repeat(x, 1, z)
                z_indices = torch.arange(z).reshape(1, 1, z).repeat(x, y, 1)
                indices = torch.stack([x_indices, y_indices, z_indices], dim=-1).reshape(-1, 3)
                coordinates = (0.5 * indices - 0.5) * self.particle_dist + offset
                # rand_orientation
                random_orientations = rand_orientation(coordinates.shape[0])
                self.coordinates = torch.cat([coordinates, random_orientations], dim=-1)

            particle_coords_file_out = self._parameters[self.Parameters.COORD_FILE_OUT]
            if particle_coords_file_out is not None and particle_coords != self.PARTICLE_COORD_OPTIONS[1]:
                # particleset_write_coord
                coords = self.coordinates.clone()
                coords[:, 3:] = radian_to_degree(coords[:, 3:])
                write_matrix_text_with_headers(coords, particle_coords_file_out,
                                               "x", "y", "z", "phi", "theta", "psi")

            self._initialized = True
            self._lock_parameters = True
            _write_log("Particleset component initialized.")

    def get_particle_rotation_matrix(self, idx: int) -> torch.Tensor:
        """
        Get particle rotation matrix given its index

        :param idx: particle index
        :return: tensor of rotation matrix
        """
        # CORRESPOND: rotate_particle
        # NOTE: instead of modifying the input matrix pm in C code, here we return a rotation matrix of particle x
        assert self._initialized, "Particleset not initialized"
        assert self.coordinates.shape[0] > idx >= 0, f"Idx = {idx} out of bound"
        rotate_matrix0 = rotate_3d_rows(torch.tensor([0., 0., 1.]), -self.coordinates[idx, 5])
        rotate_matrix1 = rotate_3d_rows(torch.tensor([1., 0., 0.]), -self.coordinates[idx, 4])
        rotate_matrix2 = rotate_3d_rows(torch.tensor([0., 0., 1.]), -self.coordinates[idx, 3])
        rotate_matrix = rotate_matrix2 @ rotate_matrix1 @ rotate_matrix0
        return rotate_matrix

    def get_particle_pos(self, idx: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        Get particle's position given its index

        :param idx: particle index
        :return: tensor of positions
        """
        assert self._initialized, "Particleset not initialized"
        if isinstance(idx, int):
            assert self.coordinates.shape[0] > idx >= 0, f"Idx = {idx} out of bound"
            pos = self.coordinates[idx, :3]
        else:
            assert torch.all((self.coordinates.shape[0] > idx) & (idx >= 0)), f"Some of idx = {idx} out of bound"
            check_tensor(idx, 1, dtype=torch.int)
            pos = self.coordinates[idx, :3]
        return pos

    def get_particle_coord(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get particle rotation matrix and position given its index

        :param idx: particle index
        :return: tensors of particle rotation matrix and position
        """
        # CORRESPOND: get_particle_coord
        # NOTE: instead of modifying the input matrix pm in C code, here we return a rotation matrix of particle x
        pos = self.get_particle_pos(idx)
        particle_rotate_matrix = self.get_particle_rotation_matrix(idx)
        return particle_rotate_matrix, pos

    def log_parameters(self, logger: logging.Logger):
        logger.info(f"Particleset:\n   {self._parameters}")

    def _unwrap_required_parameter(self, parameter: Parameters) -> Any:
        parameter_value = self._parameters[parameter]
        if parameter_value is None:
            raise RequiredParameterNotSpecifiedException(parameter)
        return parameter_value

    @property
    def device(self) -> torch.device:
        assert self._initialized, "Initialize particle set first"
        return self.coordinates.device

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def particle_type(self) -> str:
        """
        The name of the particle component used in the particleset.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particleset.Parameters.PARTICLE_TYPE)

    @property
    def particle_coords(self) -> str:
        """
        Specifies how the particle coordinates should be generated.
        \"file\" means that positions and orientations are read from a text file.
        \"rand\" means that positions and orientations are randomly generated.
        \"grid\" means that the positions are on a regular grid, and the orientations are random.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particleset.Parameters.PARTICLE_COORDS)

    @property
    def where(self) -> str:
        """
        Possible locations of randomly placed particles.
        Available choices are the entire sample volume, near both surfaces of the sample, near the top surface, or near the bottom surface.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particleset.Parameters.WHERE)

    @property
    def num_particles(self) -> int:
        """
        When generating random particle positions, one of the parameters num_particles, particle_conc, or occupancy must be defined.
        num_particles defines the number of particles to place in the sample.
        The parameter is optional if positions are read from a file, and defines the number of particle positions in the file to use.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particleset.Parameters.NUM_PARTICLES)

    @property
    def particle_conc(self) -> float:
        """
        When generating random particle positions, one of the parameters num_particles, particle_conc, or occupancy must be defined.
        particle_conc defines the number of particles per cubic micrometer of the sample if \"where\" = \"volume\",
        or the number of particles per square micrometer on the surface otherwise.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particleset.Parameters.PARTICLE_CONC)

    @property
    def occupancy(self) -> float:
        """
        When generating random particle positions, one of the parameters num_particles, particle_conc, or occupancy must be defined.
        occupancy defines the fraction of the sample volume occupied by particles if where = \"volume\", or the fraction of the sample surface occupied in other cases.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particleset.Parameters.OCCUPANCY)

    @property
    def n(self) -> torch.Tensor:
        """
        The size of the grid in the x/y/z axis direction when particle positions are on a regular grid.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particleset.Parameters.N)

    @property
    def particle_dist(self) -> float:
        """
        The distance in nanometer between the centers of adjacent particles when particle positions are on a regular grid.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particleset.Parameters.PARTICLE_DIST)

    @property
    def coord_file_in(self) -> str:
        """
        The name of a text file from which the positions of particles are read if they are not randomly generated.
        See the manual for details about the file format.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particleset.Parameters.COORD_FILE_IN)

    @property
    def coord_file_out(self) -> str:
        """
        The name of a text file which randomly generated particle positions are written to.
        See the manual for details about the file format.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particleset.Parameters.COORD_FILE_OUT)

    @property
    def offset(self) -> torch.Tensor:
        """
        The x/y/z coordinate of the grid center when particle positions are on a regular grid.

        :raises RequiredParameterNotSpecifiedException: if this parameter is not set
        """
        return self._unwrap_required_parameter(Particleset.Parameters.OFFSET)

    def _set_parameter(self, parameter: Parameters, value):
        if isinstance(value, parameter.type()):
            if self._lock_parameters:
                raise ParameterLockedException(parameter)
            else:
                self._parameters[parameter] = value
        else:
            raise WrongArgumentTypeException(value, parameter.type())

    @particle_type.setter
    def particle_type(self, particle_type: str):
        self._set_parameter(self.Parameters.PARTICLE_TYPE, particle_type)

    @particle_coords.setter
    def particle_coords(self, particle_coords: str):
        assert particle_coords in self.PARTICLE_COORD_OPTIONS, f"Invalid particle_coord = {particle_coords}, " \
                                                               f"valid options = {self.PARTICLE_COORD_OPTIONS}"
        self._set_parameter(self.Parameters.PARTICLE_COORDS, particle_coords)

    @offset.setter
    def offset(self, offset: torch.Tensor):
        check_tensor(offset, dimensionality=1, shape=(3,), dtype=torch.double)
        self._parameters[self.Parameters.OFFSET] = torch.clone(offset)

    @where.setter
    def where(self, where: str):
        assert where in self.WHERE_OPTIONS, f"Invalid where = {where}, valid options = {self.WHERE_OPTIONS}"
        self._set_parameter(self.Parameters.WHERE, where)

    @num_particles.setter
    def num_particles(self, num_particles: int):
        self._set_parameter(self.Parameters.NUM_PARTICLES, num_particles)

    @particle_conc.setter
    def particle_conc(self, particle_conc: float):
        self._set_parameter(self.Parameters.PARTICLE_CONC, particle_conc)

    @occupancy.setter
    def occupancy(self, occupancy: float):
        self._set_parameter(self.Parameters.OCCUPANCY, occupancy)

    @n.setter
    def n(self, n: torch.Tensor):
        check_tensor(n, dimensionality=1, shape=(3,), dtype=torch.int)
        assert torch.all(n > 0)
        self._parameters[self.Parameters.N] = torch.clone(n)

    @particle_dist.setter
    def particle_dist(self, particle_dist: float):
        self._set_parameter(self.Parameters.PARTICLE_DIST, particle_dist)

    @coord_file_in.setter
    def coord_file_in(self, coord_file_in: str):
        self._set_parameter(self.Parameters.COORD_FILE_IN, coord_file_in)

    @coord_file_out.setter
    def coord_file_out(self, coord_file_out: str):
        self._set_parameter(self.Parameters.COORD_FILE_OUT, coord_file_out)
