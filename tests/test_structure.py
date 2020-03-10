import pytest
import numpy as np


@pytest.fixture
def pcb_prop_oshpark4():
    from pyems.pcb import common_pcbs

    return common_pcbs["oshpark4"]


@pytest.fixture
def sim():
    from pyems.simulation_beta import Simulation

    return Simulation(freq=np.linspace(4e9, 8e9, 501), unit=1e-3)


@pytest.fixture
def pcb_oshpark4(sim, pcb_prop_oshpark4):
    from pyems.structure import PCB

    return PCB(sim=sim, pcb_prop=pcb_prop_oshpark4, length=30, width=10)


def test_copper_layer_elevation(pcb_oshpark4):
    assert np.isclose(pcb_oshpark4.copper_layer_elevation(0), 0)
    assert np.isclose(pcb_oshpark4.copper_layer_elevation(1), -0.1702)
    assert np.isclose(pcb_oshpark4.copper_layer_elevation(2), -1.364)
    assert np.isclose(pcb_oshpark4.copper_layer_elevation(3), -1.5342)
