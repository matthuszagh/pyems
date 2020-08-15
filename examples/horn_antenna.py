#!/usr/bin/env python3

import os
import numpy as np
from pyems.utilities import print_table
from pyems.port import RectWaveguidePort
from pyems.simulation import Simulation
from pyems.field_dump import FieldDump
from pyems.structure import standard_waveguides
from pyems.coordinate import Coordinate3, Box3, Axis
from pyems.mesh import Mesh
from pyems.nf2ff import NF2FF

unit = 1e-3
freq = np.linspace(5.3e9, 5.9e9, 501)
sim = Simulation(freq=freq, unit=unit)

metal = sim.csx.AddMetal("metal")
stl = metal.AddPolyhedronReader(
    filename=os.path.abspath("horn-antenna.stl")
)
stl.ReadFile()

wg = standard_waveguides["WR159"]
wg.set_unit(unit)
wg_len = 40
port = RectWaveguidePort(
    sim=sim,
    box=Box3(
        Coordinate3(-wg.a / 2, -wg.b / 2, -wg_len),
        Coordinate3(wg.a / 2, wg.b / 2, 0),
    ),
    propagation_axis=Axis("z"),
    excite=True,
)
port.add_metal_shell(thickness=5)

mesh = Mesh(
    sim=sim,
    metal_res=1 / 20,
    nonmetal_res=1 / 10,
    smooth=(1.5, 1.5, 1.5),
    min_lines=5,
    expand_bounds=((16, 16), (16, 16), (8, 24)),
)
field_dump = FieldDump(sim=sim, box=mesh.sim_box(include_pml=False))
nf2ff = NF2FF(sim=sim)

sim.run()
sim.view_field()

s11 = sim.s_param(1, 1)
print_table(
    np.concatenate(([sim.freq / 1e9], [s11])),
    col_names=["freq", "s11"],
    prec=[4, 4],
)

theta = np.arange(-90, 90, 1)
phi = np.arange(0, 360, 1)

nf2ff.calc(theta=theta, phi=phi)

horn_width = 109.9e-3
horn_height = 80e-3
effective_aperture = horn_height * horn_width
print(nf2ff.directivity(effective_aperture))
print("gain: {:.2f} dB".format(nf2ff.gain()))

rad_phi0 = nf2ff.radiation_pattern(phi=0)
rad_phi90 = nf2ff.radiation_pattern(phi=90)

print("phi0")
print_table(
    np.concatenate(([theta], [rad_phi0])),
    col_names=["theta", "gain"],
    prec=[4, 4],
)

print("phi90")
print_table(
    np.concatenate(([theta], [rad_phi90])),
    col_names=["theta", "gain"],
    prec=[4, 4],
)
