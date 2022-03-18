#!/usr/bin/env python3

import os
import sys
import numpy as np
from pyems.pcb import common_pcbs
from pyems.simulation import Simulation
from pyems.field_dump import FieldDump
from pyems.utilities import print_table, mil_to_mm
from pyems.structure import PCB, Microstrip, Via
from pyems.coordinate import Box2, Axis
from pyems.mesh import Mesh


unit = 1e-3
pcb_prop = common_pcbs["oshpark4"]
z0_ref = 50
freq = np.linspace(0, 18e9, 501)
pcb_len = 20
pcb_width = 10
microstrip_width = 0.38
drill_radius = 0.13
drill_diam = 2 * drill_radius
annular_width = 0.145
antipad = mil_to_mm(6)
keepout_radius = drill_radius + annular_width + antipad
via_sep = 0.76

sim = Simulation(freq=freq, unit=unit)
pcb = PCB(
    sim=sim,
    pcb_prop=pcb_prop,
    length=pcb_len,
    width=pcb_width,
    omit_copper=[0, 3],
)
box = Box2((-pcb_len / 2, -microstrip_width / 2), (0, microstrip_width / 2))
Microstrip(
    pcb=pcb,
    position=box.center(),
    length=box.length(),
    width=box.width(),
    propagation_axis=Axis("x"),
    trace_layer=0,
    gnd_layer=1,
    port_number=1,
    feed_shift=0.3,
    excite=True,
    ref_impedance=z0_ref,
)

Via(
    pcb=pcb,
    position=(0, 0),
    drill=drill_diam,
    annular_ring=annular_width,
    antipad=antipad,
    noconnect_layers=[0, 1, 2, 3],
)

Via(
    pcb=pcb,
    position=(0, -via_sep),
    drill=drill_diam,
    annular_ring=annular_width,
    antipad=antipad,
)

Via(
    pcb=pcb,
    position=(0, via_sep),
    drill=drill_diam,
    annular_ring=annular_width,
    antipad=antipad,
)
box = Box2((pcb_len / 2, microstrip_width / 2), (0, -microstrip_width / 2))
Microstrip(
    pcb=pcb,
    position=box.center(),
    length=box.length(),
    width=box.width(),
    propagation_axis=Axis("x", direction=-1),
    trace_layer=3,
    gnd_layer=2,
    port_number=2,
    excite=False,
    ref_impedance=z0_ref,
)

mesh = Mesh(
    sim=sim,
    metal_res=1 / 80,
    nonmetal_res=1 / 40,
    min_lines=3,
    expand_bounds=((0, 0), (0, 0), (20, 20)),
)

dump = FieldDump(sim=sim, box=mesh.sim_box(include_pml=False))

if os.getenv("_PYEMS_PYTEST"):
    sys.exit(0)

sim.run()
sim.view_field()

print_table(
    data=[sim.freq / 1e9, sim.s_param(1, 1), sim.s_param(2, 1)],
    col_names=["freq", "s11", "s21"],
    prec=[2, 4, 4],
)
