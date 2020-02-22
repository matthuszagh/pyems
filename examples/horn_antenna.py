#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
import CSXCAD as csxcad
import openEMS as openems
from pyems.network import Network
from pyems.utilities import wavelength, array_index
from pyems.port import RectWaveguidePort, standard_waveguides
from pyems.simulation import Simulation
from pyems.field_dump import FieldDump

fdtd = openems.openEMS(EndCriteria=1e-5)
csx = csxcad.ContinuousStructure()
unit = 1e-3
csx.GetGrid().SetDeltaUnit(unit)

wg = standard_waveguides["WR159"]
wg_len = 40 * unit
waveguide_box = np.multiply(
    1 / unit,
    [[-wg["a"] / 2, -wg["b"] / 2, -wg_len], [wg["a"] / 2, wg["b"] / 2, 0]],
)
port = RectWaveguidePort(
    csx=csx, box=waveguide_box, propagation_axis=2, excite=True
)
port.add_metal_shell(thickness=5)

metal = csx.AddMetal("metal")
stl = metal.AddPolyhedronReader(filename=os.path.abspath("./horn-antenna.stl"))
stl.ReadFile()

center_freq = 5.6e9
delta_freq = 5e9
network = Network(csx=csx, ports=[port])

sim_box = network.get_sim_box()
field_dump = FieldDump(csx=csx, box=sim_box)

network.generate_mesh(
    min_wavelength=wavelength(center_freq + delta_freq, unit),
    metal_res=1 / 40,
    min_lines=2,
    expand_bounds=[20, 20, 20, 20, 10, 40],
)
network._write_csx()
network.view()

sim = Simulation(
    fdtd=fdtd,
    csx=csx,
    center_freq=center_freq,
    half_bandwidth=delta_freq,
    boundary_conditions=["PML_8", "PML_8", "PML_8", "PML_8", "PML_8", "PML_8"],
    network=network,
    field_dumps=[field_dump],
)

sim.simulate(nf2ff=True)
sim.view_field()

s11 = network.s_param(1, 1)
ports = network.get_ports()
freq = sim.get_freq()

plt.figure()
plt.plot(freq, s11)
plt.show()

theta = np.arange(-90, 90, 1)
phi = np.arange(0, 360, 1)

nf2ff = sim.calc_nf2ff(theta=theta, phi=phi)

horn_width = 109.9 * 1e-3
horn_height = 80 * 1e-3
effective_aperture = horn_height * horn_width
directivity = (
    effective_aperture * 4 * np.pi / np.power(wavelength(center_freq, 1), 2)
)
gain = nf2ff.Dmax[0]
gain_db = 10 * np.log10(gain)
efficiency = gain / directivity

freq_array = sim.get_freq()
f0_idx = array_index(center_freq, freq_array)
incident_power = ports[0].incident_power()[f0_idx]
print("gain:\t\t{:.2f} dB".format(gain_db))
print("efficiency:\t{:.2f} %".format(100 * efficiency))
print("efficiency2:\t{:.2f} %".format(100 * nf2ff.Prad[0] / incident_power))

enorm = nf2ff.E_norm[0]
phi90_idx = array_index(90, phi)

plt.plot(theta, 20 * np.log10(enorm[:, 0] / np.amax(enorm[:, 0])) + gain_db)
plt.show()
plt.plot(
    theta,
    20 * np.log10(enorm[:, phi90_idx] / np.amax(enorm[:, phi90_idx]))
    + gain_db,
)
plt.show()
