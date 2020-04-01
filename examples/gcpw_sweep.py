#!/usr/bin/env python

import numpy as np
from pyems.pcb import common_pcbs
from pyems.structure import PCB, Microstrip
from pyems.coordinate import Box2, Coordinate2, Axis
from pyems.mesh import Mesh
from pyems.simulation import Simulation, sweep
from pyems.utilities import mil_to_mm, array_index, print_table


freq = np.linspace(4e9, 8e9, 501)
sim = Simulation(freq=freq, unit=1e-3)
pcb_prop = common_pcbs["oshpark4"]
pcb_len = 30
pcb_width = 10
z0_target = 50
width_dev_factor = 0.1
center_width = 0.34
gap = mil_to_mm(6)
via_gap = 0.4
# num_points = 11
num_points = 31


def gen_sim(width: float) -> Simulation:
    """
    Create simulation objects to sweep over.

    :param width: Top layer trace width.  This is the parameter we
        sweep over.
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
        Coordinate2(-pcb_len / 2, -width / 2),
        Coordinate2(pcb_len / 2, width / 2),
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
        via_gap=(via_gap, via_gap),
        via=None,
        via_spacing=1.27,
        port_number=1,
        excite=True,
    )
    Mesh(
        sim=sim,
        metal_res=1 / 80,
        nonmetal_res=1 / 40,
        smooth=(1.1, 1.5, 1.5),
        min_lines=25,
        expand_bounds=((0, 0), (8, 8), (8, 8)),
    )
    return sim


def sim_impedance(sim: Simulation):
    """
    Get the characteristic impedance of a simulation.
    """
    sim.run(csx=False)
    ports = sim.ports
    z0 = ports[0].impedance()
    idx = array_index(sim.center_frequency(), sim.freq)
    return np.abs(z0[idx])


widths = np.linspace(
    center_width * (1 - width_dev_factor),
    center_width * (1 + width_dev_factor),
    num_points,
)
sims = [gen_sim(width=width) for width in widths]
sim_vals = sweep(sims=sims, func=sim_impedance, processes=11)
print_table(data=[widths, sim_vals], col_names=["width", "z0"], prec=[4, 4])
