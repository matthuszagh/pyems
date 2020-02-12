"""
A collection of ready-made simulations related to transmission lines.
"""

from typing import List
import numpy as np
import matplotlib.pyplot as plt
from pyems.port import MicrostripPort
from pyems.pcb import PCB
from pyems.network import Network
from pyems.simulation import Simulation
from pyems.field_dump import FieldDump
from pyems.utilities import pretty_print
from openEMS import openEMS
from CSXCAD.CSXCAD import ContinuousStructure


def microstrip_width_z0(
    center_freq: float,
    half_bandwidth: float,
    width: float,
    pcb: PCB,
    plot=False,
):
    """
    Find the characteristic impedance of a microstrip transmission
    line for a given width.
    """
    fdtd = openEMS(EndCriteria=1e-5)
    csx = ContinuousStructure()
    trace_len = 100
    micro_port = MicrostripPort(
        csx=csx,
        bounding_box=[
            [-trace_len / 2, -width / 2, -pcb.layer_separation()[0]],
            [trace_len / 2, width / 2, 0],
        ],
        thickness=pcb.layer_thickness()[0],
        conductivity=pcb.metal_conductivity(),
        excite=True,
    )
    substrate = csx.AddMaterial(
        "substrate",
        epsilon=pcb.epsr_at_freq(center_freq),
        kappa=pcb.substrate_conductivity(),
    )
    substrate.AddBox(
        priority=0,
        start=np.multiply(
            1e-3, [-trace_len / 2, -20, -pcb.layer_separation()[0]]
        ),
        stop=np.multiply(1e-3, [trace_len / 2, 20, 0]),
    )
    efield = FieldDump(
        csx=csx,
        box=[
            [-trace_len / 2, -20, -pcb.layer_separation()[0]],
            [trace_len / 2, 20, 0],
        ],
    )
    network = Network(csx=csx, ports=[micro_port])
    sim = Simulation(
        fdtd=fdtd,
        csx=csx,
        center_freq=center_freq,
        half_bandwidth=half_bandwidth,
        boundary_conditions=["PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"],
        network=network,
        field_dumps=[efield],
    )
    sim.finalize_structure(expand_bounds=[0, 0, 10, 10, 0, 10])
    sim.view_network()
    sim.simulate()
    sim.view_field()
    z0 = sim.network.ports[0].characteristic_impedance()
    pretty_print(data=[z0[0] / 1e9, z0[1]], col_names=["freq", "z0"])
    if plot:
        plt.figure()
        plt.plot(z0[0], z0[1])
        plt.show()
