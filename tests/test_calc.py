import numpy as np
from pyems.calc import (
    microstrip_effective_dielectric,
    phase_shift_length,
    skin_depth,
)


def test_microstrip_effective_dielectric():
    """
    """
    assert np.isclose(
        microstrip_effective_dielectric(3.66, 0.1702, 0.34), 2.8324394821
    )


def test_phase_shift_length():
    """
    """
    assert np.isclose(
        phase_shift_length(90, 2.8324394821, 5.6e9), 7.95229284805
    )


def test_skin_depth():
    """
    """
    assert np.isclose(skin_depth(5.6e9), 8.71727524699e-7)
