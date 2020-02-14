#!/usr/bin/env python

from pyems.port import standard_waveguides
from pyems.sims.transmission_line import RectWGSimulation

wg = standard_waveguides["WR159"]
sim = RectWGSimulation()
sim.rectwg_sim(5.6e9, 1e9, wg, True)
