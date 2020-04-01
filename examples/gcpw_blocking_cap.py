#!/usr/bin/env python

import numpy as np
from pyems.pcb import common_pcbs
from pyems.simulation import Simulation
from pyems.field_dump import FieldDump
from pyems.utilities import pretty_print, mil_to_mm
from pyems.structure import (
    PCB,
    Microstrip,
    common_smd_passives,
    SMDPassive,
    Taper,
    ViaWall,
)
from pyems.coordinate import Box2, Box3, Coordinate2, Coordinate3, Axis
from pyems.mesh import Mesh

unit = 1e-3
freq = np.arange(0, 18e9, 1e7)
# freq = np.linspace(4e9, 8e9, 501)
sim = Simulation(freq=freq, unit=unit)
pcb_prop = common_pcbs["oshpark4"]
pcb_len = 10
pcb_width = 5
trace_width = 0.34
gap = mil_to_mm(6)
via_gap = 0.4
z0_ref = 50

cap_dim = common_smd_passives["0402C"]
cap_dim.set_unit(unit)
pad_length = 0.6
pad_width = cap_dim.width

pcb = PCB(
    sim=sim,
    pcb_prop=pcb_prop,
    length=pcb_len,
    width=pcb_width,
    layers=range(3),
    omit_copper=[0],
)

box = Box2(
    Coordinate2(-pcb_len / 2, -trace_width / 2),
    Coordinate2(-(cap_dim.length / 2) - (pad_length / 2), trace_width / 2),
)
microstrip1 = Microstrip(
    pcb=pcb,
    position=box.center(),
    length=box.length(),
    width=box.width(),
    propagation_axis=Axis("x"),
    trace_layer=0,
    gnd_layer=1,
    # gnd_gap=(gap, gap),
    # via_gap=(via_gap, via_gap),
    # via=None,
    port_number=1,
    excite=True,
    feed_shift=0.3,
)

# ViaWall(
#     pcb=pcb,
#     position=Coordinate2(0, trace_width / 2 + gap + via_gap),
#     length=pcb_len,
#     width=via_gap / 2,
# )

# ViaWall(
#     pcb=pcb,
#     position=Coordinate2(0, -trace_width / 2 - gap - via_gap),
#     length=pcb_len,
#     width=via_gap / 2,
# )

taper = Taper(
    pcb=pcb,
    position=None,
    pcb_layer=0,
    width1=trace_width,
    width2=pad_width,
    length=pad_width,
    gap=gap,
)

# values based on Murata GJM1555C1H100FB01 (ESR at 6GHz)
cap = SMDPassive(
    pcb=pcb,
    position=Coordinate2(0, 0),
    axis=Axis("x"),
    dimensions=cap_dim,
    pad_width=pad_width,
    # pad_width=trace_width,
    pad_length=pad_length,
    # gap=gap,
    c=10e-12,
    # r=0.7,
    # l=4.4e-10,
    pcb_layer=0,
    # gnd_cutout_width=1.2,
    # gnd_cutout_length=1,
    # taper=taper,
)
box = Box2(
    Coordinate2(pcb_len / 2, trace_width / 2),
    Coordinate2((cap_dim.length / 2) + (pad_length / 2), -trace_width / 2),
)
microstrip2 = Microstrip(
    pcb=pcb,
    position=box.center(),
    length=box.length(),
    width=box.width(),
    propagation_axis=Axis("x", direction=-1),
    trace_layer=0,
    gnd_layer=1,
    # gnd_gap=(gap, gap),
    # via_gap=(via_gap, via_gap),
    # via=None,
    port_number=2,
    excite=False,
)

dump = FieldDump(
    sim=sim,
    box=Box3(
        Coordinate3(-pcb_len / 2, -pcb_width / 2, 0),
        Coordinate3(pcb_len / 2, pcb_width / 2, 0),
    ),
)

mesh = Mesh(
    sim=sim,
    metal_res=1 / 120,
    nonmetal_res=1 / 40,
    smooth=(1.5, 1.5, 1.5),
    min_lines=5,
    expand_bounds=((0, 0), (8, 8), (8, 8)),
)

sim.run()
sim.view_field()

pretty_print(
    data=[sim.freq / 1e9, sim.s_param(1, 1), sim.s_param(2, 1)],
    col_names=["freq", "s11", "s21"],
    prec=[4, 4, 4],
)
