"""
Stores useful constants

CORRESPOND: the part of constants in macros.h
"""
import math
from .basic_units import *

PI = math.pi
_ONE_METER = 1e9 * ONE_NANOMETER  # Derived unit that causes circular import issue

# Physical units that cause circular import issue
_ONE_COULOMB = (ONE_KILOGRAM * _ONE_METER * _ONE_METER / (ONE_SECOND * ONE_SECOND * ONE_VOLT))
_ELEMENTARY_CHARGE = 1.60217646e-19 * _ONE_COULOMB
