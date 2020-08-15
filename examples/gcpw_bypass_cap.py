#!/usr/bin/env python3

import numpy as np
from CSXCAD.CSTransform import CSTransform
from pyems.pcb import common_pcbs
from pyems.structure import PCB, Microstrip, common_smd_passives, SMDPassive
from pyems.simulation import Simulation
from pyems.coordinate import Coordinate2, Axis, Box3, Coordinate3
from pyems.utilities import mil_to_mm, print_table
from pyems.mesh import Mesh
from pyems.field_dump import FieldDump, DumpType


freq = np.linspace(1e9, 18e9, 501)
unit = 1e-3
sim = Simulation(freq=freq, unit=unit, reference_frequency=5.6e9)

pcb_len = 30
pcb_width = 10
trace_width = 0.34
gap = mil_to_mm(6)
via_gap = 0.4

pcb_prop = common_pcbs["oshpark4"]
pcb = PCB(
    sim=sim,
    pcb_prop=pcb_prop,
    length=pcb_len,
    width=pcb_width,
    layers=range(3),
)

Microstrip(
    pcb=pcb,
    position=Coordinate2(-pcb_len / 4, 0),
    length=pcb_len / 2,
    width=trace_width,
    propagation_axis=Axis("x"),
    gnd_gap=(gap, gap),
    via_gap=(via_gap, via_gap),
    port_number=1,
    excite=True,
    ref_impedance=50,
)

cap_0402 = common_smd_passives["0402C"]
cap_0402.set_unit(unit)

# values based on Murata GJM1555C1HR50BB01 (ESR at 12GHz)
SMDPassive(
    pcb=pcb,
    position=Coordinate2(0, -cap_0402.length / 2),
    axis=Axis("y"),
    dimensions=cap_0402,
    pad_width=0.5,
    pad_length=trace_width,
    gap=None,
    c=0.5e-12,
    # r=0.4,
    # l=3.5e-10,
)

Microstrip(
    pcb=pcb,
    position=Coordinate2(pcb_len / 4, 0),
    length=pcb_len / 2,
    width=trace_width,
    propagation_axis=Axis("x", direction=-1),
    gnd_gap=(gap, gap),
    via_gap=(via_gap, via_gap),
    port_number=2,
    excite=False,
    ref_impedance=50,
)

Mesh(
    sim=sim,
    metal_res=1 / 80,
    nonmetal_res=1 / 40,
    smooth=[1.1, 1.5, 1.5],
    min_lines=9,
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

print_table(
    data=[sim.freq / 1e9, sim.s_param(1, 1), sim.s_param(2, 1)],
    col_names=["freq", "s11", "s21"],
    prec=[4, 4, 4],
)
