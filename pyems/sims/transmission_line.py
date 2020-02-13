"""
A collection of ready-made simulations related to transmission lines.
"""

import sys
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from pyems.port import MicrostripPort, CPWPort
from pyems.pcb import PCB
from pyems.network import Network
from pyems.simulation import Simulation, sweep
from pyems.field_dump import FieldDump
from pyems.utilities import pretty_print, table_interp_val
from pyems import calc
from openEMS import openEMS
from CSXCAD.CSXCAD import ContinuousStructure


def _imped_at_freq(sim: Simulation, freq: float):
    """
    """
    sim.simulate()
    net_ports = sim.get_network().get_ports()
    z0 = net_ports[0].characteristic_impedance()
    return float(table_interp_val(z0, target_col=1, sel_val=freq, sel_col=0))


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
        pretty_print(data=[z0[:, 0] / 1e9, z0[:, 1]], col_names=["freq", "z0"])
        if plot:
            plt.figure()
            plt.plot(z0[:, 0], z0[:, 1])
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
        processes: int = 11,
        plot: bool = False,
    ):
        """
        """
        if center_width is None:
            center_width = 1e3 * calc.wheeler_z0_width(
                z0=z0_target,
                t=pcb.layer_thickness()[0],
                er=pcb.epsr_at_freq(center_freq),
                h=pcb.layer_separation()[0],
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
        func = partial(_imped_at_freq, freq=center_freq)
        sim_vals = sweep(sims=sims, func=func, processes=processes)
        analytic_vals = [
            calc.wheeler_z0(
                w=1e-3 * width,
                t=pcb.layer_thickness()[0],
                er=pcb.epsr_at_freq(center_freq),
                h=pcb.layer_separation()[0],
            )
            for width in widths
        ]
        pretty_print(
            data=[widths, sim_vals, analytic_vals],
            col_names=["width", "sim", "wheeler"],
        )
        if plot:
            plt.figure()
            plt.plot(widths, sim_vals)
            plt.plot(widths, analytic_vals)
            plt.show()

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
        efields = None
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
            efields = [efield]
        network = Network(csx=csx, ports=[micro_port])
        sim = Simulation(
            fdtd=fdtd,
            csx=csx,
            center_freq=center_freq,
            half_bandwidth=half_bandwidth,
            boundary_conditions=["PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"],
            network=network,
            field_dumps=efields,
        )
        sim.finalize_structure(expand_bounds=[0, 0, 10, 10, 0, 10])
        return sim


class GCPWSimulation:
    """
    """

    def gcpw_width_z0(
        self,
        center_freq: float,
        half_bandwidth: float,
        width: float,
        gap: float,
        pcb: PCB,
        plot=False,
    ):
        """
        Find the characteristic impedance of a grounded coplanar
        waveguide transmission line for a given width.
        """
        sim = self._gen_gcpw_sim(
            center_freq, half_bandwidth, width, gap, pcb, view_field=True
        )
        sim.view_network()
        sim.simulate()
        sim.view_field()
        sim.save_field("tmp")
        net_ports = sim.get_network().get_ports()
        z0 = net_ports[0].characteristic_impedance()
        pretty_print(data=[z0[:, 0] / 1e9, z0[:, 1]], col_names=["freq", "z0"])
        if plot:
            plt.figure()
            plt.plot(z0[:, 0], z0[:, 1])
            plt.show()

    def gcpw_width_sweep(
        self,
        pcb: PCB,
        center_freq: float,
        half_bandwidth: float,
        z0_target: float,
        center_width: float = None,
        gap: float = None,
        width_dev_factor: float = 0.1,
        num_points: int = 11,
        processes: int = 11,
        plot: bool = False,
        out_file=sys.stdout,
    ):
        """
        """
        if center_width is None:
            center_width = 1e3 * calc.wheeler_z0_width(
                z0=z0_target,
                t=pcb.layer_thickness()[0],
                er=pcb.epsr_at_freq(center_freq),
                h=pcb.layer_separation()[0],
            )
        widths = np.linspace(
            center_width * (1 - width_dev_factor),
            center_width * (1 + width_dev_factor),
            num_points,
        )
        sims = [
            self._gen_gcpw_sim(center_freq, half_bandwidth, width, gap, pcb)
            for width in widths
        ]
        func = partial(_imped_at_freq, freq=center_freq)
        sim_vals = sweep(sims=sims, func=func, processes=processes)
        pretty_print(
            data=[widths, sim_vals],
            col_names=["width", "sim"],
            out_file=out_file,
        )
        if plot:
            plt.figure()
            plt.plot(widths, sim_vals)
            plt.show()

    def _gen_gcpw_sim(
        self,
        center_freq: float,
        half_bandwidth: float,
        width: float,
        gap: float,
        pcb: PCB,
        view_field: bool = False,
    ) -> Simulation:
        """
        """
        width *= 1e-3
        gap *= 1e-3
        fdtd = openEMS(EndCriteria=1e-5)
        csx = ContinuousStructure()
        trace_len = 100e-3
        sub_width = 40e-3
        cpw_port = CPWPort(
            csx=csx,
            bounding_box=[
                [-trace_len / 2, -width / 2, -pcb.layer_separation()[0]],
                [trace_len / 2, width / 2, 0],
            ],
            gap=gap,
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
        ground = csx.AddConductingSheet(
            "Ground",
            conductivity=pcb.metal_conductivity(),
            thickness=pcb.layer_thickness()[0],
        )
        ground.AddBox(
            priority=999,
            start=[-trace_len / 2, -sub_width / 2, 0],
            stop=[trace_len / 2, -width / 2 - gap, 0],
        )
        ground.AddBox(
            priority=999,
            start=[-trace_len / 2, width / 2 + gap, 0],
            stop=[trace_len / 2, sub_width / 2, 0],
        )
        efields = None
        if view_field:
            efield = FieldDump(
                csx=csx,
                box=[
                    [
                        -trace_len / 2,
                        -sub_width / 2,
                        -pcb.layer_separation()[0] / 2,
                    ],
                    [
                        trace_len / 2,
                        sub_width / 2,
                        -pcb.layer_separation()[0] / 2,
                    ],
                ],
            )
            efields = [efield]
        network = Network(csx=csx, ports=[cpw_port])
        sim = Simulation(
            fdtd=fdtd,
            csx=csx,
            center_freq=center_freq,
            half_bandwidth=half_bandwidth,
            boundary_conditions=["PML_8", "PML_8", "MUR", "MUR", "PEC", "MUR"],
            network=network,
            field_dumps=efields,
        )
        sim.finalize_structure(expand_bounds=[0, 0, 10, 10, 0, 10])
        # # Add vias. This is an abuse of the API...
        # mesh = network.get_mesh()
        # x_lines = mesh.mesh_lines[0]
        # _, ylow = mesh.nearest_mesh_line(1, -width / 2 - (2 * gap))
        # _, yhigh = mesh.nearest_mesh_line(1, width / 2 + (2 * gap))
        # metal = csx.AddMetal("PEC")
        # for xval in x_lines:
        #     metal.AddBox(
        #         priority=999,
        #         start=[xval, ylow, -pcb.layer_separation()[0]],
        #         stop=[xval, ylow, 0],
        #     )
        #     metal.AddBox(
        #         priority=999,
        #         start=[xval, yhigh, -pcb.layer_separation()[0]],
        #         stop=[xval, yhigh, 0],
        #     )
        return sim
