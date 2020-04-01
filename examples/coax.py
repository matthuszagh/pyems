#!/usr/bin/env python

import numpy as np
from pyems.simulation import Simulation
from pyems.structure import Coax
from pyems.coordinate import Coordinate3, Axis
from pyems.utilities import mil_to_mm, print_table
from pyems.calc import coax_core_diameter
from pyems.material import common_dielectrics
from pyems.field_dump import FieldDump, DumpType
from pyems.mesh import Mesh
from pyems.boundary import BoundaryConditions

freq = np.linspace(4e9, 8e9, 501)
unit = 1e-3
sim = Simulation(
    freq=freq,
    unit=unit,
    boundary_conditions=BoundaryConditions(
        (("PML_8", "PML_8"), ("PML_8", "PML_8"), ("PML_8", "PML_8")),
    ),
)

dielectric = common_dielectrics["PTFE"]
length = 50
coax_rad = mil_to_mm(190 / 2)  # RG-141
core_rad = (
    coax_core_diameter(
        2 * coax_rad, dielectric.epsr_at_freq(sim.center_frequency())
    )
    / 2
)

Coax(
    sim=sim,
    position=Coordinate3(0, 0, 0),
    length=length,
    radius=coax_rad,
    core_radius=core_rad,
    shield_thickness=mil_to_mm(5),
    dielectric=dielectric,
    propagation_axis=Axis("x"),
    port_number=1,
    excite=True,
    feed_shift=0.3,
    ref_impedance=50,
)

mesh = Mesh(
    sim=sim,
    metal_res=1 / 40,
    nonmetal_res=1 / 10,
    smooth=(1.1, 1.5, 1.5),
    min_lines=9,
    expand_bounds=((0, 0), (8, 8), (8, 8)),
)

box = mesh.sim_box(include_pml=False)
field = FieldDump(sim=sim, box=box, dump_type=DumpType.efield_time)

sim.run()
sim.view_field()

z0 = sim.ports[0].impedance()
s11 = sim.s_param(1, 1)
print_table(
    data=[sim.freq / 1e9, np.abs(z0), s11],
    col_names=["freq", "z0", "s11"],
    prec=[4, 4, 4],
)
