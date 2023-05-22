from diff_tem.eletronbeam import Electronbeam
from diff_tem.geometry import Geometry
from diff_tem.simulation import Simulation
from diff_tem.utils.input_processing import *
from diff_tem.detector import Detector
from diff_tem.optics import Optics
from diff_tem.particle import Particle
from diff_tem.particleset import Particleset
from diff_tem.sample import Sample


def test_read_file():
    file_path = "./input.txt"
    components = read_input_file(file_path)
    assert len(components["detector"]) == 2


def test_detector_construction():
    file_path = "./input.txt"
    components = read_input_file(file_path)
    detector_parameter_tables = components["detector"]
    detectors = Detector.prototype_from_parameters(detector_parameter_tables)
    assert len(detectors) == 2


def test_sample_construction():
    file_path = "./input.txt"
    components = read_input_file(file_path)
    sample_parameter_tables = components["sample"]
    samples = Sample.construct_from_parameters(sample_parameter_tables)
    assert len(samples) == 1


def test_particle_construction():
    import taichi as ti
    ti.init(ti.cpu)
    file_path = "./input.txt"
    components = read_input_file(file_path)
    particle_parameter_tables = components["particle"]
    particles = Particle.prototype_from_parameters(particle_parameter_tables)
    assert len(particles) == 1


def test_particleset_construction():
    file_path = "./input.txt"
    components = read_input_file(file_path)
    particleset_parameter_tables = components["particleset"]
    particlesets = Particleset.prototype_from_parameters(particleset_parameter_tables)
    assert len(particlesets) == 1


def test_geometry_construction():
    file_path = "./input.txt"
    components = read_input_file(file_path)
    geometry_parameter_tables = components["geometry"]
    geometries = Geometry.construct_from_parameters(geometry_parameter_tables)
    assert len(geometries) == 1


def test_electronbeam_construction():
    file_path = "./input.txt"
    components = read_input_file(file_path)
    electronbeam_parameter_tables = components["electronbeam"]
    electronbeams = Electronbeam.prototype_from_parameters(electronbeam_parameter_tables)
    assert len(electronbeams) == 1


def test_optics_construction():
    file_path = "./input.txt"
    components = read_input_file(file_path)
    optics_parameter_tables = components["optics"]
    optics = Optics.prototype_from_parameters(optics_parameter_tables)
    assert len(optics) == 1


def test_simulation_construction():
    file_path = "./input.txt"
    components = read_input_file(file_path)
    simulation_parameter_tables = components["simulation"]
    simulation = Simulation.construct_from_parameters(simulation_parameter_tables)
    assert len(simulation) == 1


def test_json():
    file_path = "./input.txt"
    components = read_input_file(file_path)
    export_input_configs_json(components, "./input.json")
    components = read_input_file("./input.json")
    print(components)
