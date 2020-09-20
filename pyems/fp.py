"""
Prevent floating-point rounding errors.

The primary purpose of this file is to avoid openems floating-point
rounding errors when generating the CSXCAD structure.  Using this
mechanism correctly requires some care.  The basic idea is that we
modify floating-point values with full precision and then round them
right before handing them off to CSXCAD.  This should ensure that any
floating-point rounding errors accumulated during processing will be
rounded away in the final result.  Since built-in floating-point
precision is more than adequate for any reasonable dimension we might
choose, rounding is not detrimental.  For instance, the rounding
behavior used rounds to the nearest tenth decimal place which would
round a meter to the nearest angstrom.

In this spirit, rounding should only be used to modify values right
before they are passed to CSXCAD.  It can additionally be used to
compare (but not modify) values which denote CSXCAD dimensions.
"""

import numpy as np

# numeric decimal precision to avoid floating point errors due to
# rounding.
PREC = 10


def fp_nearest(val_or_arr):
    """
    Nearest floating-point values to the provided ones using the
    precision for the final CSXCAD structure.
    """
    return np.around(val_or_arr, PREC)


def fp_equalp(val1: float, val2: float) -> bool:
    """
    Test whether two floating-point values are equal with respect to
    the CSXCAD structure precision.

    :param val1: First floating-point value.
    :param val1: Second floating-point value.

    :returns: True if the values are the same with respect to the
              model precision and False if not.
    """
    return np.around(val1, PREC) == np.around(val2, PREC)


def fp_gtp(val1: float, val2: float) -> bool:
    """
    Test whether the floating-point value ``val1`` is strictly greater
    than the floating value ``val2`` with respect to the CSXCAD
    structure precision.  I.e., this performs the test:

    val1 > val2

    :param val1: The first floating-point value.
    :param val2: The second floating-point value.

    :returns: True if ``val1`` greater than ``val2``.
    """
    return np.around(val1, PREC) > np.around(val2, PREC)


def fp_gep(val1: float, val2: float) -> bool:
    """
    Test whether the floating-point value ``val1`` is greater than or
    equal to the floating value ``val2`` with respect to the CSXCAD
    structure precision.  I.e., this performs the test:

    val1 >= val2

    :param val1: The first floating-point value.
    :param val2: The second floating-point value.

    :returns: True if ``val1`` greater than or equal to ``val2``.
    """
    return np.around(val1, PREC) >= np.around(val2, PREC)


def fp_ltp(val1: float, val2: float) -> bool:
    """
    Test whether the floating-point value ``val1`` is strictly less
    than the floating value ``val2`` with respect to the CSXCAD
    structure precision.  I.e., this performs the test:

    val1 < val2

    :param val1: The first floating-point value.
    :param val2: The second floating-point value.

    :returns: True if ``val1`` less than ``val2``.
    """
    return np.around(val1, PREC) < np.around(val2, PREC)


def fp_lep(val1: float, val2: float) -> bool:
    """
    Test whether the floating-point value ``val1`` is less than or
    equal to the floating value ``val2`` with respect to the CSXCAD
    structure precision.  I.e., this performs the test:

    val1 <= val2

    :param val1: The first floating-point value.
    :param val2: The second floating-point value.

    :returns: True if ``val1`` less than or equal to ``val2``.
    """
    return np.around(val1, PREC) <= np.around(val2, PREC)
