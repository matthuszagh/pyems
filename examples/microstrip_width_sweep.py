#!/usr/bin/env python3

import numpy as np
from pyems.pcb import common_pcbs
from pyems.structure import PCB, Microstrip
from pyems.coordinate import Coordinate2, Axis
from pyems.mesh import Mesh
from pyems.simulation import Simulation
from pyems.calc import sweep
from pyems.utilities import mil_to_mm, array_index, print_table


freq = np.linspace(0, 18e9, 501)
ref_freq = 5.6e9
pcb_prop = common_pcbs["oshpark4"]
pcb_len = 10
pcb_width = 5
width_dev_factor = 0.1
center_width = 0.38
num_points = 31

def z0_for_width(width: float) -> float:
    """
    """
    sim = Simulation(freq=freq, unit=1e-3, reference_frequency=ref_freq, sim_dir=None)
    pcb = PCB(
        sim=sim,
        pcb_prop=pcb_prop,
        length=pcb_len,
        width=pcb_width,
        layers=range(3),
        omit_copper=[0],
    )
    Microstrip(
        pcb=pcb,
        position=Coordinate2(0, 0),
        length=pcb_len,
        width=width,
        propagation_axis=Axis("x"),
        port_number=1,
        excite=True,
        ref_impedance=50,
    )
    Mesh(
        sim=sim,
        metal_res=1 / 80,
        nonmetal_res=1 / 10,
        min_lines=5,
        expand_bounds=((0, 0), (0, 0), (10, 40)),
    )
    sim.run(csx=False)
    return np.abs(sim.ports[0].impedance(freq=ref_freq))


widths = np.linspace(
    center_width * (1 - width_dev_factor),
    center_width * (1 + width_dev_factor),
    num_points,
)
sim_vals = sweep(func=z0_for_width, params=widths, processes=11)
print_table(data=[widths, sim_vals], col_names=["width", "z0"], prec=[4, 4])
