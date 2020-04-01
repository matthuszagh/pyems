#!/usr/bin/env python

import numpy as np
from pyems.simulation import Simulation
from pyems.pcb import common_pcbs
from pyems.structure import DifferentialMicrostrip, PCB
from pyems.coordinate import Coordinate2, Axis, Box3, Coordinate3
from pyems.utilities import pretty_print, mil_to_mm
from pyems.field_dump import FieldDump, DumpType
from pyems.mesh import Mesh

freq = np.arange(0, 18e9, 10e6)
unit = 1e-3
ref_freq = 5.6e9
sim = Simulation(freq=freq, unit=unit, reference_frequency=ref_freq)

pcb_len = 30
pcb_width = 10

trace_width = 0.85
gnd_gap = mil_to_mm(6)
trace_gap = mil_to_mm(6)
via_gap = 0.4

pcb_prop = common_pcbs["oshpark4"]
pcb = PCB(
    sim=sim,
    pcb_prop=pcb_prop,
    length=pcb_len,
    width=pcb_width,
    layers=range(3),
)

DifferentialMicrostrip(
    pcb=pcb,
    position=Coordinate2(0, 0),
    length=pcb_len,
    width=trace_width,
    gap=trace_gap,
    propagation_axis=Axis("x"),
    gnd_gap=(gnd_gap, gnd_gap),
    via_gap=(via_gap, via_gap),
    port_number=1,
    excite=True,
    ref_impedance=50,
)

Mesh(
    sim=sim,
    metal_res=1 / 80,
    nonmetal_res=1 / 40,
    smooth=(1.1, 1.5, 1.5),
    min_lines=5,
    expand_bounds=((0, 0), (8, 8), (8, 8)),
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

pretty_print(
    data=[sim.freq / 1e9, np.abs(sim.ports[0].impedance()), sim.s_param(1, 1)],
    col_names=["freq", "z0", "s11"],
    prec=[2, 4, 4],
)
