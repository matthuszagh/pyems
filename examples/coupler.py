#!/usr/bin/env python

import numpy as np
from pyems.structure import Microstrip, PCB, MicrostripCoupler
from pyems.simulation import Simulation
from pyems.pcb import common_pcbs
from pyems.calc import phase_shift_length, microstrip_effective_dielectric
from pyems.utilities import print_table
from pyems.coordinate import Coordinate2, Axis, Box3, Coordinate3
from pyems.mesh import Mesh
from pyems.field_dump import FieldDump, DumpType
from pyems.kicad import write_footprint

freq = np.linspace(0e9, 18e9, 501)
ref_freq = 5.6e9
unit = 1e-3
sim = Simulation(freq=freq, unit=unit, reference_frequency=ref_freq)

pcb_prop = common_pcbs["oshpark4"]
trace_width = 0.38
trace_spacing = 0.2
eeff = microstrip_effective_dielectric(
    pcb_prop.substrate.epsr_at_freq(ref_freq),
    substrate_height=pcb_prop.substrate_thickness(0, unit=unit),
    trace_width=trace_width,
)
coupler_length = phase_shift_length(90, eeff, ref_freq)
print("coupler_length: {:.4f}".format(coupler_length))
pcb_len = 2 * coupler_length
pcb_width = 0.5 * pcb_len
pcb = PCB(
    sim=sim,
    pcb_prop=pcb_prop,
    length=pcb_len,
    width=pcb_width,
    layers=range(3),
    omit_copper=[0],
)

coupler = MicrostripCoupler(
    pcb=pcb,
    position=Coordinate2(0, 0),
    trace_layer=0,
    gnd_layer=1,
    trace_width=trace_width,
    trace_gap=trace_spacing,
    length=coupler_length,
    miter=None,
)

coupler_port_positions = coupler.port_positions()
port0_x = coupler_port_positions[0].x
port0_y = coupler_port_positions[0].y

Microstrip(
    pcb=pcb,
    position=Coordinate2(np.average([port0_x, -pcb_len / 2]), port0_y),
    length=port0_x + pcb_len / 2,
    width=trace_width,
    propagation_axis=Axis("x"),
    port_number=1,
    excite=True,
    ref_impedance=50,
    feed_shift=0.3,
)

port1_x = coupler_port_positions[1].x
Microstrip(
    pcb=pcb,
    position=Coordinate2(np.average([port1_x, pcb_len / 2]), port0_y),
    length=pcb_len / 2 - port1_x,
    width=trace_width,
    propagation_axis=Axis("x", direction=-1),
    port_number=2,
    excite=False,
    ref_impedance=50,
)

port2_x = coupler_port_positions[2].x
port2_y = coupler_port_positions[2].y
Microstrip(
    pcb=pcb,
    position=Coordinate2(port2_x, np.average([port2_y, -pcb_width / 2])),
    length=port2_y + pcb_width / 2,
    width=trace_width,
    propagation_axis=Axis("y"),
    ref_impedance=50,
    port_number=3,
)

port3_x = coupler_port_positions[3].x
Microstrip(
    pcb=pcb,
    position=Coordinate2(port3_x, np.average([port2_y, -pcb_width / 2])),
    length=port2_y + pcb_width / 2,
    width=trace_width,
    propagation_axis=Axis("y"),
    ref_impedance=50,
    port_number=4,
)

Mesh(
    sim=sim,
    metal_res=1 / 120,
    nonmetal_res=1 / 40,
    smooth=(1.1, 1.5, 1.5),
    min_lines=3,
    expand_bounds=((0, 0), (0, 0), (10, 20)),
)

FieldDump(
    sim=sim,
    box=Box3(
        Coordinate3(-pcb_len / 2, -pcb_width / 2, 0),
        Coordinate3(pcb_len / 2, pcb_width / 2, 0),
    ),
    dump_type=DumpType.current_density_time,
)

write_footprint(coupler, "coupler_20db", "coupler_20db.kicad_mod")

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
