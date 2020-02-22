#!/usr/bin/env python

from pyems.sims.transmission_line import MicrostripSimulation
from pyems.pcb import common_pcbs

pcb = common_pcbs["oshpark4"]
sim = MicrostripSimulation()
sim.width_z0(
    center_freq=5.6e9,
    half_bandwidth=1e9,
    width=0.34,
    pcb=pcb,
    unit=1e-3,
    plot=True,
)
