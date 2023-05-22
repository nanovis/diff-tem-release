import torch
import logging
import numpy as np
from typing import List, Union, Optional, Dict

from . import UNINITIALIZED
from .constants import PI
from .detector import Detector
from .eletronbeam import Electronbeam, wave_number
from .geometry import Geometry
from .optics import Optics
from .particle import Particle
from .particleset import Particleset
from .sample import Sample
from .utils.input_processing import read_input_file, convert_bool
from .utils.components import Components
from .volume import Volume
from .wavefunction import WaveFunction

_logger = logging.getLogger(__name__)


def _write_log(string, *logging_strings):
    strings = [string, *logging_strings]
    if len(strings) > 1:
        logging_string = "\n    ".join(strings)
        _logger.debug(f"{_logger.name}:\n    {logging_string}", stacklevel=2)
    else:
        _logger.debug(f"{_logger.name}:  {strings[0]}", stacklevel=2)


class Simulation:
    """
    The simulation component defines what should happen when
    the simulator is run. Three kinds of actions are possible: generate a series of
    simulated micrographs (the main purpose of the program), generate one or several
    volume maps of the sample used in the simulation, and generate maps of one or
    several individual particles.
    """

    def __init__(self,
                 generate_micrographs: Optional[bool] = False,
                 generate_volumes: Optional[bool] = False,
                 generate_particle_maps: Optional[bool] = False,
                 log_file: Optional[str] = None,
                 rand_seed: Optional[int] = None):
        self.gen_micrographs = generate_micrographs
        self.gen_volumes = generate_volumes
        self.gen_particle_maps = generate_particle_maps
        self.log_file = log_file
        self.rand_seed = rand_seed
        if rand_seed is not None:
            torch.random.manual_seed(rand_seed)
            torch.cuda.manual_seed(rand_seed)
            np.random.seed(rand_seed)

        self.detectors: List[Detector] = []
        self.particles: List[Particle] = []
        self.particle_sets: List[Particleset] = []
        self.volumes: List[Volume] = []

        self.sample: Sample = UNINITIALIZED
        self.electronbeam: Electronbeam = UNINITIALIZED
        self.geometry: Geometry = UNINITIALIZED
        self.optics: Optics = UNINITIALIZED

        self.parameter_logger = logging.getLogger("Simulation Parameter Logger")
        logging.info("Simulation parameters object initialized.\n")

    @staticmethod
    def construct_from_file(file_path: str):
        """
        Construct a simulation with all its components from a file
        :param file_path: file path to txt or json file
        :return: Ready-to-go Simulation instance
        """
        configs = read_input_file(file_path)
        assert Components.SIMULATION in configs, "Must specify a simulation in file"
        # components that have only one instance
        simulations = Simulation.construct_from_parameters(configs[Components.SIMULATION])
        assert len(simulations) == 1, f"Allow only 1 simulation in file, got {len(simulations)}"
        simulation = simulations[0]
        if Components.SAMPLE in configs:
            samples = Sample.construct_from_parameters(configs[Components.SAMPLE])
            assert len(samples) == 1, f"Allow only 1 sample in file, got {len(samples)}"
            simulation.sample = samples[0]
        if Components.ELECTRON_BEAM in configs:
            electron_beams = Electronbeam.prototype_from_parameters(configs[Components.ELECTRON_BEAM])
            assert len(electron_beams) == 1, f"Allow only 1 electronbeam in file, got {len(electron_beams)}"
            simulation.electronbeam = electron_beams[0]
        if Components.GEOMETRY in configs:
            geometries = Geometry.construct_from_parameters(configs[Components.GEOMETRY])
            assert len(geometries) == 1, f"Allow only 1 geometry in file, got {len(geometries)}"
            simulation.geometry = geometries[0]
        if Components.OPTICS in configs:
            optics = Optics.prototype_from_parameters(configs[Components.OPTICS])
            assert len(optics) == 1, f"Allow only 1 optics in file, got {len(optics)}"
            simulation.optics = optics[0]
        # components that may have multiple instances and at least 1
        if Components.DETECTOR in configs:
            simulation.detectors = Detector.prototype_from_parameters(configs[Components.DETECTOR])
            assert len(simulation.detectors) > 0
        if Components.PARTICLE in configs:
            simulation.particles = Particle.prototype_from_parameters(configs[Components.PARTICLE])
            assert len(simulation.particles) > 0
        if Components.PARTICLESET in configs:
            simulation.particle_sets = Particleset.prototype_from_parameters(configs[Components.PARTICLESET])
            assert len(simulation.particle_sets) > 0
        # optional components
        if Components.VOLUME in configs:
            simulation.volumes = Volume.construct_from_parameters(configs[Components.VOLUME])

        n_tilts = simulation.geometry.ntilts
        simulation.electronbeam.finish_init_(n_tilts)
        simulation.optics.finish_init_(n_tilts)
        for particle in simulation.particles:
            particle.finish_init_(simulation)
        for detector in simulation.detectors:
            detector.finish_init_(simulation, None)
        for particle_set in simulation.particle_sets:
            particle_set.finish_init_(simulation)

        return simulation

    @staticmethod
    def construct_from_parameters(parameters: Union[List[Dict[str, str]], Dict[str, str]]):
        """
        Construct instances of Simulation given parameters

        :param parameters: list of parameter dictionaries or a dictionary containing parameters
        :return: list of instances or one instance
        """
        if isinstance(parameters, list):
            return [Simulation._from_parameters(parameter_table) for parameter_table in parameters]
        elif isinstance(parameters, dict):
            return Simulation._from_parameters(parameters)
        else:
            raise Exception(f"Invalid parameter of type {type(parameters)}")

    @staticmethod
    def _from_parameters(parameters: Dict[str, str]):
        generate_micrographs = False
        generate_volumes = False
        generate_particle_maps = False
        log_file = None
        rand_seed = None
        if "generate_micrographs" in parameters:
            generate_micrographs = convert_bool(parameters["generate_micrographs"])
        if "generate_volumes" in parameters:
            generate_volumes = convert_bool(parameters["generate_volumes"])
        if "generate_particle_maps" in parameters:
            generate_particle_maps = convert_bool(parameters["generate_particle_maps"])
        if "log_file" in parameters:
            log_file = parameters["log_file"]
        if "rand_seed" in parameters:
            rand_seed = int(parameters["rand_seed"])
        return Simulation(generate_micrographs, generate_volumes, generate_particle_maps, log_file, rand_seed)

    def run(self):
        # CORRESPOND: simulation_run
        if self.gen_particle_maps:
            self.generate_particle_maps()
        if self.gen_volumes:
            self.generate_volumes()
        if self.gen_micrographs:
            self.generate_micrographs()

    def generate_micrographs(self):
        raise Exception("TODO: sync the code with simulation_grad.py")
        if len(self.detectors) == 0 or self.electronbeam is None or self.geometry is None \
                or self.optics is None or self.sample is None:
            raise Exception("Incomplete input data for generating micrographs.")
        self.geometry.finish_init_()
        n_tilts = self.geometry.ntilts
        self.electronbeam.finish_init_(n_tilts)
        self.optics.finish_init_(n_tilts)
        for particle_set in self.particle_sets:
            particle_set.finish_init_(self)

        pixel_size = self.detectors[0].pixel_size_nanometer
        xy_range = self.detectors[0].count.xy_range
        for d in self.detectors:
            pixel_size = min(pixel_size, d.pixel_size_nanometer)
            xy_range = torch.minimum(xy_range, d.count.xy_range)

        wf = WaveFunction(self.electronbeam, self.optics, pixel_size, xy_range)
        slice_th = wf.pixel_size ** 2 / (4 * PI) * wave_number(self.electronbeam.acceleration_energy)
        _write_log("Generating micrographs.\n")

        wave_functions = [wf.spawn() for _ in range(n_tilts)]

        for tilt in range(n_tilts):
            wf = wave_functions[tilt]
            wf.propagate_(self, slice_th, tilt)

        detector_standins = [[d.spawn() for _ in range(n_tilts)] for d in self.detectors]
        for detector_in_tilt_series in detector_standins:
            for tilt in range(n_tilts):
                nth_detector = detector_in_tilt_series[tilt]
                nth_wf = wave_functions[tilt]
                nth_detector.add_intensity_from_wave_function_to_vvf(nth_wf, tilt)
                if nth_detector.use_quantization:
                    nth_detector.apply_quantization()
                nth_detector.apply_mtf()

        for detector, detector_standins in zip(self.detectors, detector_standins):
            detector.write_image(detector_standins)

        _write_log("Simulation complete")

    def generate_volumes(self):
        if self.sample is None:
            raise Exception("Incomplete input data for generating volumes.")
        for particle_set in self.particle_sets:
            particle_set.finish_init_(self)

    def generate_particle_maps(self):
        for particle in self.particles:
            source = particle.source
            if (source == Particle.SOURCE_OPTIONS[1] or source == Particle.SOURCE_OPTIONS[2]) \
                    and (particle.map_file_re_out_ is not None
                         or particle.map_file_im_out_ is not None):  # pdb or random
                particle.finish_init_(self)
        _write_log("Generating volumes")
        for volume in self.volumes:
            volume.finish_init_()
            volume.calculate_potential(self)
            volume.write_maps()
            volume.reset()

    def log_parameters(self):
        # CORRESPOND: simulation_write_log
        self.electronbeam.log_parameters(self.parameter_logger)
        self.geometry.log_parameters(self.parameter_logger)
        self.optics.log_parameters(self.parameter_logger)
        self.sample.log_parameters(self.parameter_logger)
        for d in self.detectors:
            d.log_parameters(self.parameter_logger)
        for p in self.particles:
            p.log_parameters(self.parameter_logger)
        for v in self.volumes:
            v.log_parameters(self.parameter_logger)

    def get_particle(self, particle_name) -> Union[Particle, None]:
        i = self._find_particle(particle_name)
        if i >= 0:
            return self.particles[i]
        else:
            return None

    def _find_particle(self, name: str) -> int:
        if name == "":
            if len(self.particles) == 1:
                return 0
            return -1
        for idx, particle in enumerate(self.particles):
            if particle.name == name:
                return idx
        return -1
