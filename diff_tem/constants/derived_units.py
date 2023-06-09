from . import PI
from . import *
from . import _ELEMENTARY_CHARGE, _ONE_METER

# Derived Units
ONE_METER = _ONE_METER
ONE_DEGREE = PI / 180. * ONE_RADIAN
ONE_MILLIRADIAN = 1e-3 * ONE_RADIAN
ONE_KILOVOLT = 1e3 * ONE_VOLT
ONE_MILLIVOLT = 1e-3 * ONE_VOLT
ONE_KILOELECTRONVOLT = _ELEMENTARY_CHARGE * ONE_KILOVOLT
ONE_ELECTRONVOLT = _ELEMENTARY_CHARGE * ONE_VOLT
ONE_ANGSTROM = 0.1 * ONE_NANOMETER
ONE_ANGSTROM_SQ = ONE_ANGSTROM * ONE_ANGSTROM
ONE_ANGSTROM_CU = ONE_ANGSTROM * ONE_ANGSTROM_SQ
ONE_NANOMETER_SQ = ONE_NANOMETER * ONE_NANOMETER
ONE_NANOMETER_CU = ONE_NANOMETER * ONE_NANOMETER_SQ
ONE_MICROMETER = 1e3 * ONE_NANOMETER
ONE_PER_MICROMETER_SQ = 1.0 / (ONE_MICROMETER * ONE_MICROMETER)
ONE_PER_MICROMETER_CU = ONE_PER_MICROMETER_SQ / ONE_MICROMETER
ONE_MILLIMETER = 1e6 * ONE_NANOMETER
