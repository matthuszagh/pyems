"""
A collection of ready-made simulations related to transmission lines.
"""

from typing import List
import numpy as np
import matplotlib.pyplot as plt
from pyems.port import MicrostripPort
from pyems.pcb import PCB
from pyems.network import Network
from pyems.simulation import Simulation
from pyems.field_dump import FieldDump
from pyems.utilities import pretty_print
from pyems import calc
from openEMS import openEMS
from CSXCAD.CSXCAD import ContinuousStructure


class MicrostripSimulation:
    """
    """

    def microstrip_width_z0(
        self,
        center_freq: float,
        half_bandwidth: float,
        width: float,
        pcb: PCB,
        plot=False,
    ):
        """
        Find the characteristic impedance of a microstrip transmission
        line for a given width.
        """
        sim = self._gen_microstrip_sim(
            center_freq, half_bandwidth, width, pcb, view_field=True
        )
        sim.view_network()
        sim.simulate()
        sim.view_field()
        net_ports = sim.get_network().get_ports()
        z0 = net_ports[0].characteristic_impedance()
        pretty_print(data=[z0[0] / 1e9, z0[1]], col_names=["freq", "z0"])
        if plot:
            plt.figure()
            plt.plot(z0[0], z0[1])
            plt.show()

    def microstrip_width_sweep(
        self,
        pcb: PCB,
        center_freq: float,
        half_bandwidth: float,
        z0_target: float,
        center_width: float = None,
        width_dev_factor: float = 0.1,
        num_points: int = 11,
        plot: bool = False,
    ):
        """
        """
        if center_width is None:
            center_width = calc.wheeler_z0_width(
                z0=z0_target,
                t=pcb.layer_thickness[0],
                er=pcb.epsr_at_freq(center_freq),
                h=pcb.layer_sep[0],
            )
        widths = np.linspace(
            center_width * (1 - width_dev_factor),
            center_width * (1 + width_dev_factor),
            num_points,
        )
        sims = [
            self._gen_microstrip_sim(center_freq, half_bandwidth, width, pcb)
            for width in widths
        ]

    def imped_at_freq(sim: Simulation, freq: float):
        """
        """
        net_ports = sim.get_network().get_ports()
        z0 = net_ports[0].characteristic_impedance()

    def _gen_microstrip_sim(
        self,
        center_freq: float,
        half_bandwidth: float,
        width: float,
        pcb: PCB,
        view_field: bool = False,
    ) -> Simulation:
        """
        """
        width *= 1e-3
        fdtd = openEMS(EndCriteria=1e-5)
        csx = ContinuousStructure()
        trace_len = 100e-3
        sub_width = 40e-3
        micro_port = MicrostripPort(
            csx=csx,
            bounding_box=[
                [-trace_len / 2, -width / 2, -pcb.layer_separation()[0]],
                [trace_len / 2, width / 2, 0],
            ],
            thickness=pcb.layer_thickness()[0],
            conductivity=pcb.metal_conductivity(),
            excite=True,
        )
        substrate = csx.AddMaterial(
            "substrate",
            epsilon=pcb.epsr_at_freq(center_freq),
            kappa=pcb.substrate_conductivity(),
        )
        substrate.AddBox(
            priority=0,
            start=[-trace_len / 2, -sub_width / 2, -pcb.layer_separation()[0]],
            stop=[trace_len / 2, sub_width / 2, 0],
        )
        if view_field:
            efield = FieldDump(
                csx=csx,
                box=[
                    [
                        -trace_len / 2,
                        -sub_width / 2,
                        -pcb.layer_separation()[0],
                    ],
                    [trace_len / 2, sub_width / 2, 0],
                ],
            )
        network = Network(csx=csx, ports=[micro_port])
        sim = Simulation(
            fdtd=fdtd,
            csx=csx,
            center_freq=center_freq,
            half_bandwidth=half_bandwidth,
            boundary_conditions=["PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"],
            network=network,
            field_dumps=[efield],
        )
        sim.finalize_structure(expand_bounds=[0, 0, 10, 10, 0, 10])
        return sim
