#!/usr/bin/env python

import numpy as np
from pyems.pcb import common_pcbs
from pyems.simulation import Simulation
from pyems.field_dump import FieldDump
from pyems.utilities import print_table, mil_to_mm
from pyems.structure import PCB, Microstrip, Via
from pyems.coordinate import Box2, Coordinate2, Axis
from pyems.mesh import Mesh


unit = 1e-3
pcb_prop = common_pcbs["oshpark4"]
z0_ref = 50
freq = np.linspace(4e9, 8e9, 501)
pcb_len = 40
pcb_width = 10
gcpw_width = 0.34
gcpw_gap = mil_to_mm(6)
via_gap = 0.4
# drill_radius = gcpw_width / 2 - pcb_prop.via_plating_thickness(unit)
drill_radius = 0.13
drill_diam = 2 * drill_radius
annular_width = 0.145
antipad = gcpw_gap
keepout_radius = drill_radius + annular_width + antipad
via_sep = 0.76
via_fence_sep = via_sep * 2

sim = Simulation(freq=freq, unit=unit)
pcb = PCB(sim=sim, pcb_prop=pcb_prop, length=pcb_len, width=pcb_width)
box = Box2(
    Coordinate2(-pcb_len / 2, -gcpw_width / 2), Coordinate2(0, gcpw_width / 2),
)
Microstrip(
    pcb=pcb,
    position=box.center(),
    length=box.length(),
    width=box.width(),
    propagation_axis=Axis("x"),
    trace_layer=0,
    gnd_layer=1,
    gnd_gap=(gcpw_gap, gcpw_gap),
    via_gap=(via_gap, via_gap),
    via=None,
    port_number=1,
    feed_shift=0.2,
    excite=True,
)

Via(
    pcb=pcb,
    position=Coordinate2(0, 0),
    drill=drill_diam,
    annular_ring=annular_width,
    antipad=antipad,
    noconnect_layers=[0, 1, 2, 3],
)

Via(
    pcb=pcb,
    position=Coordinate2(0, -via_sep),
    drill=drill_diam,
    annular_ring=annular_width,
    antipad=antipad,
)

Via(
    pcb=pcb,
    position=Coordinate2(0, via_sep),
    drill=drill_diam,
    annular_ring=annular_width,
    antipad=antipad,
)
box = Box2(
    Coordinate2(pcb_len / 2, gcpw_width / 2), Coordinate2(0, -gcpw_width / 2),
)
Microstrip(
    pcb=pcb,
    position=box.center(),
    length=box.length(),
    width=box.width(),
    propagation_axis=Axis("x", direction=-1),
    trace_layer=3,
    gnd_layer=2,
    gnd_gap=(gcpw_gap, gcpw_gap),
    via_gap=(via_gap, via_gap),
    via=None,
    port_number=2,
    excite=False,
)

mesh = Mesh(
    sim=sim,
    metal_res=1 / 80,
    nonmetal_res=1 / 40,
    smooth=(1.3, 1.5, 1.5),
    min_lines=3,
    expand_bounds=((0, 0), (8, 8), (8, 8)),
)

dump = FieldDump(sim=sim, box=mesh.sim_box(include_pml=False))

sim.run()
sim.view_field()

print_table(
    data=[sim.freq / 1e9, sim.s_param(1, 1), sim.s_param(2, 1)],
    col_names=["freq", "s11", "s21"],
    prec=[4, 4, 4],
)
