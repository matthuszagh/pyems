#!/usr/bin/env python

import numpy as np
from CSXCAD.CSTransform import CSTransform
from pyems.structure import Microstrip, PCB, Miter
from pyems.simulation import Simulation
from pyems.pcb import common_pcbs
from pyems.calc import (
    phase_shift_length,
    microstrip_effective_dielectric,
    pozar_z0_width,
)
from pyems.utilities import print_table
from pyems.coordinate import Coordinate2, Axis, Box3, Coordinate3
from pyems.mesh import Mesh
from pyems.field_dump import FieldDump, DumpType

freq = np.linspace(0e9, 18e9, 501)
ref_freq = 5.6e9
unit = 1e-3
sim = Simulation(freq=freq, unit=unit, reference_frequency=ref_freq)

pcb_prop = common_pcbs["oshpark4"]
# trace_width = pozar_z0_width(
#     50,
#     pcb_prop.substrate_thickness(0, unit=unit),
#     substrate_dielectric=pcb_prop.substrate.epsr_at_freq(ref_freq),
# )
# print("trace width: {:.4f}".format(trace_width))
trace_width = 0.38
trace_spacing = 0.2
eeff = microstrip_effective_dielectric(
    pcb_prop.substrate.epsr_at_freq(ref_freq),
    substrate_height=pcb_prop.substrate_thickness(0),
    trace_width=trace_width,
)
coupler_length = phase_shift_length(90, eeff, ref_freq)
print("coupler_length: {:.4f}".format(coupler_length))
pcb_len = 1.5 * coupler_length
pcb_width = 0.5 * pcb_len
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
    position=Coordinate2(-pcb_len / 4, trace_spacing / 2 + trace_width / 2),
    length=pcb_len / 2,
    width=trace_width,
    propagation_axis=Axis("x"),
    port_number=1,
    excite=True,
    ref_impedance=50,
)

Microstrip(
    pcb=pcb,
    position=Coordinate2(pcb_len / 4, trace_spacing / 2 + trace_width / 2),
    length=pcb_len / 2,
    width=trace_width,
    propagation_axis=Axis("x", direction=-1),
    port_number=2,
    excite=False,
    ref_impedance=50,
)

Microstrip(
    pcb=pcb,
    position=Coordinate2(0, -trace_spacing / 2 - trace_width / 2),
    length=coupler_length,
    width=trace_width,
    propagation_axis=Axis("x"),
)

end_miter = Miter(
    pcb=pcb,
    position=Coordinate2(
        coupler_length / 2, -trace_spacing / 2 - trace_width / 2
    ),
    pcb_layer=0,
    gnd_layer=1,
    trace_width=trace_width,
    gap=None,
    miter=None,
)

end_miter_pos = end_miter.end_point()
miter_offset = end_miter_pos.x - coupler_length / 2

tr = CSTransform()
tr.AddTransform("RotateAxis", "z", 90)

start_miter = Miter(
    pcb=pcb,
    position=Coordinate2(
        -coupler_length / 2 - miter_offset,
        -trace_spacing / 2 - trace_width - end_miter.inset_length(),
    ),
    pcb_layer=0,
    gnd_layer=1,
    trace_width=trace_width,
    gap=None,
    miter=None,
    transform=tr,
)

micro_y_dist = end_miter_pos.y + pcb_width / 2
micro_y_mid = np.average([end_miter_pos.y, -pcb_width / 2])

Microstrip(
    pcb=pcb,
    position=Coordinate2(-coupler_length / 2 - miter_offset, micro_y_mid),
    length=micro_y_dist,
    width=trace_width,
    propagation_axis=Axis("y"),
    ref_impedance=50,
    port_number=3,
)

Microstrip(
    pcb=pcb,
    position=Coordinate2(coupler_length / 2 + miter_offset, micro_y_mid),
    length=micro_y_dist,
    width=trace_width,
    propagation_axis=Axis("y"),
    ref_impedance=50,
    port_number=4,
)

Mesh(
    sim=sim,
    metal_res=1 / 160,
    nonmetal_res=1 / 40,
    smooth=(1.1, 1.5, 1.5),
    min_lines=5,
    expand_bounds=((0, 0), (0, 8), (8, 8)),
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
    data=[
        sim.freq / 1e9,
        np.abs(sim.ports[0].impedance()),
        sim.s_param(1, 1),
        sim.s_param(2, 1),
        sim.s_param(3, 1),
        sim.s_param(4, 1),
    ],
    col_names=["freq", "z0", "s11", "s21", "s31", "s41"],
    prec=[4, 4, 4, 4, 4, 4],
)
