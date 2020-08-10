#!/usr/bin/env python

import numpy as np
from pyems.structure import PCB, Microstrip
from pyems.simulation import Simulation
from pyems.mesh import Mesh
from pyems.pcb import common_pcbs
from pyems.coordinate import Coordinate2, Axis
from pyems.calc import optimize_parameter, sweep
from pyems.utilities import print_table

freq = np.arange(1e9, 18e9, 1e7)
ref_freq = 5.6e9
unit = 1e-3
pcb_len = 10
pcb_width = 10
trace_width = 0.38
pcb_prop = common_pcbs["oshpark4"]


def func(gnd_gap: float):
    sim = Simulation(
        freq=freq, unit=unit, reference_frequency=ref_freq, sim_dir=None
    )
    pcb = PCB(
        sim=sim,
        pcb_prop=pcb_prop,
        length=pcb_len,
        width=pcb_width,
        layers=range(3),
    )

    Microstrip(
        pcb=pcb,
        position=Coordinate2(0, 0),
        length=pcb_len,
        width=trace_width,
        propagation_axis=Axis("x"),
        gnd_gap=(gnd_gap, gnd_gap),
        port_number=1,
        excite=True,
        ref_impedance=50,
    )

    Mesh(
        sim=sim,
        metal_res=1 / 80,
        nonmetal_res=1 / 10,
        min_lines=9,
        expand_bounds=((0, 0), (0, 0), (10, 40)),
    )

    sim.run(csx=False)
    return np.abs(sim.ports[0].impedance(freq=ref_freq))


# res = optimize_parameter(
#     func,
#     start=0.25 * trace_width,
#     step=0.25 * trace_width,
#     tol=0.2,
#     max_steps=20,
#     display_progress=True,
# )
# print("Minimum ground gap: {}".format(res))

gaps = np.arange(0.5 * trace_width, 3 * trace_width, 0.1 * trace_width)
sim_vals = sweep(func, gaps, processes=11)

norm_gaps = gaps/trace_width

print_table(data=[norm_gaps, sim_vals], col_names=["gap", "z0"], prec=[4, 4])
