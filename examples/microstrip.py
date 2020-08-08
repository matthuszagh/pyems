#!/usr/bin/env python

import numpy as np
from pyems.structure import PCB, Microstrip
from pyems.simulation import Simulation
from pyems.mesh import Mesh
from pyems.pcb import common_pcbs
from pyems.coordinate import Coordinate2, Axis, Box3, Coordinate3
from pyems.field_dump import FieldDump, DumpType
from pyems.utilities import print_table

freq = np.arange(0, 18e9, 1e7)
ref_freq = 5.6e9
unit = 1e-3
sim = Simulation(freq=freq, unit=unit, reference_frequency=ref_freq)
pcb_len = 10
pcb_width = 5
trace_width = 0.38

pcb_prop = common_pcbs["oshpark4"]
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
    width=trace_width,
    propagation_axis=Axis("x"),
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

FieldDump(
    sim=sim,
    box=Box3(
        Coordinate3(-pcb_len / 2, -pcb_width / 2, 0),
        Coordinate3(pcb_len / 2, pcb_width / 2, 0),
    ),
    dump_type=DumpType.current_density_time,
)

sim.run()
sim.view_field()

print_table(
    data=[sim.freq / 1e9, np.abs(sim.ports[0].impedance()), sim.s_param(1, 1)],
    col_names=["freq", "z0", "s11"],
    prec=[2, 4, 4],
)
