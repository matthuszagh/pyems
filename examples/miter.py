#!/usr/bin/env python3

import numpy as np
from pyems.simulation import Simulation
from pyems.structure import PCB, Microstrip, Miter
from pyems.coordinate import Coordinate2, Box2, Coordinate3, Box3, Axis
from pyems.pcb import common_pcbs
from pyems.mesh import Mesh
from pyems.utilities import print_table, mil_to_mm
from pyems.field_dump import FieldDump, DumpType

freq = np.linspace(4e9, 8e9, 501)
sim = Simulation(freq=freq, unit=1e-3)
pcb_prop = common_pcbs["oshpark4"]
pcb_len = 10
trace_width = 0.38
gap = mil_to_mm(6)
via_gap = 0.4

pcb = PCB(
    sim=sim, pcb_prop=pcb_prop, length=pcb_len, width=pcb_len, layers=range(3)
)
box = Box2(
    Coordinate2(-pcb_len / 2, pcb_len / 4 - (trace_width / 2)),
    Coordinate2(pcb_len / 4, pcb_len / 4 + (trace_width / 2)),
)

miter = Miter(
    pcb=pcb,
    position=Coordinate2(pcb_len / 4, pcb_len / 4),
    pcb_layer=0,
    gnd_layer=1,
    trace_width=trace_width,
    gap=gap,
)
print(
    "miter: {:.4f}mm ({:.2f}%)".format(
        miter.miter, 100 * miter.miter / (trace_width * np.sqrt(2))
    )
)

Microstrip(
    pcb=pcb,
    position=box.center(),
    length=box.max_corner.x - box.min_corner.x,
    width=trace_width,
    propagation_axis=Axis("x"),
    trace_layer=0,
    gnd_layer=1,
    gnd_gap=(gap, gap),
    via_gap=(via_gap, via_gap),
    via=None,
    shorten_via_wall=(0, gap + via_gap - miter.inset_length()),
    port_number=1,
    feed_shift=0.4,
    excite=True,
)

end_pos = miter.end_point()

box = Box2(
    Coordinate2(end_pos.x - (trace_width / 2), end_pos.y),
    Coordinate2(end_pos.x + (trace_width / 2), -pcb_len / 2),
)
Microstrip(
    pcb=pcb,
    position=box.center(),
    length=box.min_corner.y - box.max_corner.y,
    width=trace_width,
    propagation_axis=Axis("y"),
    trace_layer=0,
    gnd_layer=1,
    gnd_gap=(gap, gap),
    via_gap=(via_gap, via_gap),
    via=None,
    shorten_via_wall=(0, gap + via_gap - miter.inset_length()),
    port_number=2,
    excite=False,
)

dump = FieldDump(
    sim=sim,
    box=Box3(
        Coordinate3(-pcb_len / 2, -pcb_len / 2, 0),
        Coordinate3(pcb_len / 2, pcb_len / 2, 0),
    ),
    dump_type=DumpType.current_density_time,
)

mesh = Mesh(
    sim=sim,
    metal_res=1 / 120,
    nonmetal_res=1 / 40,
    smooth=(1.5, 1.5, 1.5),
    min_lines=5,
    expand_bounds=((0, 8), (0, 8), (8, 8)),
)

sim.run()
sim.view_field()

print_table(
    data=[sim.freq / 1e9, sim.s_param(1, 1), sim.s_param(2, 1)],
    col_names=["freq", "s11", "s21"],
    prec=[4, 4, 4],
)
