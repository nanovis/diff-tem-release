from enum import Enum

from ..detector import Detector
from ..eletronbeam import Electronbeam
from ..geometry import Geometry
from ..optics import Optics
from ..particle import Particle
from ..particleset import Particleset
from ..sample import Sample
from ..volume import Volume


class Components(Enum):
    SIMULATION = "simulation"
    SAMPLE = "sample"
    PARTICLE = "particle"
    PARTICLESET = "particleset"
    GEOMETRY = "geometry"
    OPTICS = "optics"
    DETECTOR = "detector"
    VOLUME = "volume"
    ELECTRON_BEAM = "electronbeam"

    def __hash__(self):
        return self.value.__hash__()

    def __eq__(self, other):
        if isinstance(other, Components):
            return self.value == other.value
        else:
            return self.value == other

    def get_description(self):
        if self == Components.SIMULATION:
            desc = "The simulation component defines what should happen when " \
                   "the simulator is run. Three kinds of actions are possible: generate a series of " \
                   "simulated micrographs (the main purpose of the program), generate one or several " \
                   "volume maps of the sample used in the simulation, and generate maps of one or " \
                   "several individual particles."
        elif self == Components.SAMPLE:
            desc = Sample.__doc__
        elif self == Components.PARTICLE:
            desc = Particle.__doc__
        elif self == Components.PARTICLESET:
            desc = Particleset.__doc__
        elif self == Components.GEOMETRY:
            desc = Geometry.__doc__
        elif self == Components.OPTICS:
            desc = Optics.__doc__
        elif self == Components.DETECTOR:
            desc = Detector.__doc__
        elif self == Components.VOLUME:
            desc = Volume.__doc__
        elif self == Components.ELECTRON_BEAM:
            desc = Electronbeam.__doc__
        else:
            raise Exception("Should not enter this branch")

        return desc

    def get_parameter_description(self, parameter_name: str):
        if self == Components.SIMULATION:
            if parameter_name == "generate_micrographs":
                desc = "Tells the simulator to generate " \
                       "a micrograph or a tilt series. Requires one each of the components sample, " \
                       "electronbeam, geometry, and optics, at least one detector, and for a meaningful " \
                       "simulation at least one particle and one particleset component."
            elif parameter_name == "generate_volumes":
                desc = "Tells the simulator to generate one " \
                       "or several volume maps of all or part of the sample, as specified by volume " \
                       "components. Requires one sample component and (to be meaningful) at least one " \
                       "particle and one particleset component. Might also require an electronbeam " \
                       "component, since the imaginary or absorption potential can depend on the " \
                       "accelertion energy."
            elif parameter_name == "generate_particle_maps":
                desc = "Tells the simulator to " \
                       "generate map files for all particle components. This also happens as a side " \
                       "effect whenever one of the other two simulation tasks are run, at least for " \
                       "particles used in a particleset."
            elif parameter_name == "log_file":
                desc = "The name of a log file which is continuously " \
                       "updated during the simulation. If the parameter is not defined, a unique name is " \
                       "automatically generated, based on the time when the simulation starts."
            elif parameter_name == "rand_seed":
                desc = "A number which is used to seed the random " \
                       "number generator at the start of a simulation. If the parameter is not defined, " \
                       "the random number generator is instead seeded with the current time."
            else:
                raise Exception(f"""Unknown parameter {parameter_name} in {self.value}.\n
                Acceptable Parameters = {["generate_micrographs", "generate_volumes", "generate_particle_maps", "log_file", "rand_seed"]}""")
        elif self == Components.SAMPLE:
            if parameter_name in Sample.PARAMETER_DESCRIPTIONS:
                desc = Sample.PARAMETER_DESCRIPTIONS[parameter_name]
            else:
                raise Exception(f"""Unknown parameter {parameter_name} in {self.value}.\n
                Acceptable Parameters = {Sample.PARAMETER_DESCRIPTIONS.keys()}""")
        elif self == Components.PARTICLE:
            all_parameter_names = Particle.Parameters.parameter_names()
            if parameter_name in all_parameter_names:
                parameter = Particle.Parameters(parameter_name)
                desc = parameter.value.description
            else:
                raise Exception(f"Unknown parameter {parameter_name} in {self.value}.\n"
                                f"Acceptable parameters = {all_parameter_names}")
        elif self == Components.PARTICLESET:
            all_parameter_names = Particleset.Parameters.parameter_names()
            if parameter_name in all_parameter_names:
                parameter = Particleset.Parameters(parameter_name)
                desc = parameter.value.description
            elif parameter_name in ["nx", "ny", "nz"]:
                desc = Particleset.Parameters.N.value.description
            elif parameter_name.startswith("offset"):
                desc = Particleset.Parameters.OFFSET.value.description
            else:
                raise Exception(f"Unknown parameter {parameter_name} in {self.value}.\n"
                                f"Acceptable parameters = {all_parameter_names}")
        elif self == Components.GEOMETRY:
            all_parameter_names = Geometry.Parameters.parameter_names()
            if parameter_name in all_parameter_names:
                parameter = Geometry.Parameters(parameter_name)
                desc = parameter.value.description
            else:
                raise Exception(f"Unknown parameter {parameter_name} in {self.value}.\n"
                                f"Acceptable parameters = {all_parameter_names}")
        elif self == Components.OPTICS:
            all_parameter_names = Optics.Parameters.parameter_names()
            if parameter_name in all_parameter_names:
                parameter = Optics.Parameters(parameter_name)
                desc = parameter.value.description
            else:
                raise Exception(f"Unknown parameter {parameter_name} in {self.value}.\n"
                                f"Acceptable parameters = {all_parameter_names}")
        elif self == Components.DETECTOR:
            all_parameter_names = Detector.Parameters.parameter_names()
            if parameter_name in all_parameter_names:
                parameter = Detector.Parameters(parameter_name)
                desc = parameter.value.description
            elif parameter_name.startswith("det_pix"):
                desc = Detector.Parameters.DET_PIX.value.description
            else:
                raise Exception(f"Unknown parameter {parameter_name} in {self.value}.\n"
                                f"Acceptable parameters = {all_parameter_names}")
        elif self == Components.VOLUME:
            all_parameter_names = Volume.Parameters.parameter_names()
            if parameter_name in all_parameter_names:
                parameter = Volume.Parameters(parameter_name)
                desc = parameter.value.description
            elif parameter_name in ["nx", "ny", "nz"]:
                desc = Volume.Parameters.N.value.description
            elif parameter_name.startswith("offset"):
                desc = Volume.Parameters.OFFSET.value.description
            else:
                raise Exception(f"Unknown parameter {parameter_name} in {self.value}.\n"
                                f"Acceptable parameters = {all_parameter_names}")
        elif self == Components.ELECTRON_BEAM:
            all_parameter_names = Electronbeam.Parameters.parameter_names()
            if parameter_name in all_parameter_names:
                parameter = Electronbeam.Parameters(parameter_name)
                desc = parameter.value.description
            else:
                raise Exception(f"Unknown parameter {parameter_name} in {self.value}.\n"
                                f"Acceptable parameters = {all_parameter_names}")
        else:
            raise Exception("Should not enter this branch")

        return desc
