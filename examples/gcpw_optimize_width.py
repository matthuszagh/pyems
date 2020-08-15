#!/usr/bin/env python3

import numpy as np
from scipy.optimize import minimize
from pyems.simulation import Simulation
from pyems.structure import PCB, Microstrip, ViaWall
from pyems.coordinate import Coordinate2, Box2, Axis
from pyems.pcb import common_pcbs
from pyems.mesh import Mesh
from pyems.utilities import mil_to_mm

freq = np.arange(1e9, 18e9, 1e7)
pcb_prop = common_pcbs["oshpark4"]
pcb_len = 20
pcb_width = 5
gap = mil_to_mm(6)
via_gap = 0.4


def gcpw(trace_width: float):
    """
    """
    sim = Simulation(freq=freq, unit=1e-3)
    pcb = PCB(
        sim=sim,
        pcb_prop=pcb_prop,
        length=pcb_len,
        width=pcb_width,
        layers=range(3),
    )
    box = Box2(
        Coordinate2(-pcb_len / 2, -trace_width / 2),
        Coordinate2(pcb_len / 2, trace_width / 2),
    )
    Microstrip(
        pcb=pcb,
        position=box.center(),
        length=box.length(),
        width=box.width(),
        propagation_axis=Axis("x"),
        trace_layer=0,
        gnd_layer=1,
        gnd_gap=(gap, gap),
        port_number=1,
        ref_impedance=50,
        excite=True,
    )

    ViaWall(
        pcb=pcb,
        position=Coordinate2(0, trace_width / 2 + gap + via_gap),
        length=pcb_len,
        width=via_gap / 2,
    )

    ViaWall(
        pcb=pcb,
        position=Coordinate2(0, -trace_width / 2 - gap - via_gap),
        length=pcb_len,
        width=via_gap / 2,
    )

    Mesh(
        sim=sim,
        metal_res=1 / 120,
        nonmetal_res=1 / 40,
        smooth=(1.1, 1.5, 1.5),
        min_lines=25,
        expand_bounds=((0, 0), (24, 24), (24, 24)),
    )

    sim.run(csx=False)

    return np.average(np.abs(np.abs(sim.ports[0].impedance()) - 50))


res = minimize(
    gcpw, 0.34, method="Nelder-Mead", tol=1e-2, options={"disp": True}
)
print(res.x)
