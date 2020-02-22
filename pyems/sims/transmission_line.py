"""
A collection of ready-made simulations related to transmission lines.
"""

import sys
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from pyems.port import MicrostripPort, CPWPort, RectWaveguidePort
from pyems.pcb import PCB
from pyems.network import Network
from pyems.simulation import Simulation, sweep
from pyems.field_dump import FieldDump
from pyems.utilities import pretty_print, table_interp_val, get_unit
from pyems import calc
from openEMS import openEMS
from CSXCAD.CSXCAD import ContinuousStructure


def _imped_at_freq(sim: Simulation, freq: float):
    """
    """
    sim.simulate()
    net_ports = sim.get_network().get_ports()
    z0 = net_ports[0].impedance()
    return float(table_interp_val(z0, target_col=1, sel_val=freq, sel_col=0))


class MicrostripSimulation:
    """
    """

    def width_z0(
        self,
        center_freq: float,
        half_bandwidth: float,
        width: float,
        pcb: PCB,
        unit: float = 1,
        plot=False,
    ):
        """
        Find the characteristic impedance of a microstrip transmission
        line for a given width.
        """
        sim = self._gen_sim(
            center_freq, half_bandwidth, width, pcb, unit=unit, view_field=True
        )
        sim.view_network()
        sim.simulate()
        sim.view_field()
        net_ports = sim.get_network().get_ports()
        freq = net_ports[0].frequency()
        z0 = net_ports[0].impedance()
        pretty_print(
            data=[freq / 1e9, z0], col_names=["freq", "z0"], prec=[4, 4]
        )
        if plot:
            plt.figure()
            plt.plot(freq, z0)
            plt.show()

    # def width_sweep(
    #     self,
    #     pcb: PCB,
    #     center_freq: float,
    #     half_bandwidth: float,
    #     z0_target: float,
    #     center_width: float = None,
    #     width_dev_factor: float = 0.1,
    #     num_points: int = 11,
    #     processes: int = 11,
    #     plot: bool = False,
    # ):
    #     """
    #     Find the impedance for a range of widths.
    #     """
    #     if center_width is None:
    #         center_width = 1e3 * calc.wheeler_z0_width(
    #             z0=z0_target,
    #             t=pcb.layer_thickness()[0],
    #             er=pcb.epsr_at_freq(center_freq),
    #             h=pcb.layer_separation()[0],
    #         )
    #     widths = np.linspace(
    #         center_width * (1 - width_dev_factor),
    #         center_width * (1 + width_dev_factor),
    #         num_points,
    #     )
    #     sims = [
    #         self._gen_sim(center_freq, half_bandwidth, width, pcb)
    #         for width in widths
    #     ]
    #     func = partial(_imped_at_freq, freq=center_freq)
    #     sim_vals = sweep(sims=sims, func=func, processes=processes)
    #     analytic_vals = [
    #         calc.wheeler_z0(
    #             w=1e-3 * width,
    #             t=pcb.layer_thickness()[0],
    #             er=pcb.epsr_at_freq(center_freq),
    #             h=pcb.layer_separation()[0],
    #         )
    #         for width in widths
    #     ]
    #     pretty_print(
    #         data=[widths, sim_vals, analytic_vals],
    #         col_names=["width", "sim", "wheeler"],
    #     )
    #     if plot:
    #         plt.figure()
    #         plt.plot(widths, sim_vals)
    #         plt.plot(widths, analytic_vals)
    #         plt.show()

    def _gen_sim(
        self,
        center_freq: float,
        half_bandwidth: float,
        width: float,
        pcb: PCB,
        unit: float = 1,
        view_field: bool = False,
    ) -> Simulation:
        """
        """
        fdtd = openEMS(EndCriteria=1e-5)
        csx = ContinuousStructure()
        csx.GetGrid().SetDeltaUnit(unit)
        trace_len = 100
        sub_width = 40
        micro_port = MicrostripPort(
            csx=csx,
            bounding_box=[
                [
                    -trace_len / 2,
                    -width / 2,
                    -pcb.layer_separation(get_unit(csx))[0],
                ],
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
            start=[
                -trace_len / 2,
                -sub_width / 2,
                -pcb.layer_separation(get_unit(csx))[0],
            ],
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
                        -pcb.layer_separation(get_unit(csx))[0],
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

    def width_z0(
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
        sim = self._gen_sim(
            center_freq, half_bandwidth, width, gap, pcb, view_field=True
        )
        sim.view_network()
        sim.simulate()
        sim.view_field()
        sim.save_field("tmp")
        net_ports = sim.get_network().get_ports()
        freq = net_ports[0].frequency()
        z0 = net_ports[0].impedance()
        pretty_print(data=[freq / 1e9, z0], col_names=["freq", "z0"])
        if plot:
            plt.figure()
            plt.plot(freq, z0)
            plt.show()

    # def width_sweep(
    #     self,
    #     pcb: PCB,
    #     center_freq: float,
    #     half_bandwidth: float,
    #     z0_target: float,
    #     center_width: float = None,
    #     gap: float = None,
    #     width_dev_factor: float = 0.1,
    #     num_points: int = 11,
    #     processes: int = 11,
    #     plot: bool = False,
    #     out_file=sys.stdout,
    # ):
    #     """
    #     """
    #     if center_width is None:
    #         center_width = 1e3 * calc.wheeler_z0_width(
    #             z0=z0_target,
    #             t=pcb.layer_thickness()[0],
    #             er=pcb.epsr_at_freq(center_freq),
    #             h=pcb.layer_separation()[0],
    #         )
    #     widths = np.linspace(
    #         center_width * (1 - width_dev_factor),
    #         center_width * (1 + width_dev_factor),
    #         num_points,
    #     )
    #     sims = [
    #         self._gen_sim(center_freq, half_bandwidth, width, gap, pcb)
    #         for width in widths
    #     ]
    #     func = partial(_imped_at_freq, freq=center_freq)
    #     sim_vals = sweep(sims=sims, func=func, processes=processes)
    #     pretty_print(
    #         data=[widths, sim_vals],
    #         col_names=["width", "sim"],
    #         out_file=out_file,
    #     )
    #     if plot:
    #         plt.figure()
    #         plt.plot(widths, sim_vals)
    #         plt.show()

    def _gen_sim(
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
                [
                    -trace_len / 2,
                    -width / 2,
                    -pcb.layer_separation(get_unit(csx))[0],
                ],
                [trace_len / 2, width / 2, 0],
            ],
            gap=gap,
            thickness=pcb.layer_thickness(get_unit(csx))[0],
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
            start=[
                -trace_len / 2,
                -sub_width / 2,
                -pcb.layer_separation(get_unit(csx))[0],
            ],
            stop=[trace_len / 2, sub_width / 2, 0],
        )
        ground = csx.AddConductingSheet(
            "Ground",
            conductivity=pcb.metal_conductivity(),
            thickness=pcb.layer_thickness(get_unit(csx))[0],
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


class RectWaveguideSimulation:
    """
    """

    def rectwg_sim(
        self, center_freq: float, half_bandwidth: float, wg, plot=False,
    ):
        """
        Find the characteristic impedance of a grounded coplanar
        waveguide transmission line for a given width.
        """
        sim = self._gen_rectwg_sim(
            wg, center_freq, half_bandwidth, view_field=True
        )
        sim.view_network()
        sim.simulate()
        sim.view_field()
        net_ports = sim.get_network().get_ports()
        freq = net_ports[0].frequency()
        z0 = net_ports[0].impedance()
        s11 = sim.get_network().s_param(1, 1)
        s12 = sim.get_network().s_param(1, 2)
        pretty_print(
            data=np.array([freq / 1e9, z0, s11, s12]),
            col_names=["freq", "z0", "s11", "s12"],
            prec=[4, 4, 4, 4],
        )
        if plot:
            plt.figure()
            plt.plot(freq, z0)
            plt.plot(freq, s11)
            plt.plot(freq, s12)
            plt.show()

    def _gen_rectwg_sim(
        self, wg, center_freq: float, half_bandwidth: float, view_field=False,
    ):
        """
        """
        fdtd = openEMS(EndCriteria=1e-5)
        csx = ContinuousStructure()

        wg_len = 1000e-3
        port_len = wg_len / 5
        port1_box = [
            [0, -wg["a"] / 2, -wg["b"] / 2],
            [port_len, wg["a"] / 2, wg["b"] / 2],
        ]
        port2_box = [
            [wg_len, wg["a"] / 2, wg["b"] / 2],
            [wg_len - port_len, -wg["a"] / 2, -wg["b"] / 2],
        ]
        port1 = RectWaveguidePort(
            csx=csx, box=port1_box, propagation_axis=0, excite=True,
        )
        port2 = RectWaveguidePort(
            csx=csx, box=port2_box, propagation_axis=0, excite=False,
        )
        efields = []
        if view_field:
            efield = FieldDump(
                csx=csx,
                box=[
                    [-port_len, -wg["a"] / 2, -wg["b"] / 2],
                    [wg_len + port_len, wg["a"] / 2, wg["b"] / 2],
                ],
            )
            efields.append(efield)
        network = Network(csx=csx, ports=[port1, port2])
        sim = Simulation(
            fdtd=fdtd,
            csx=csx,
            center_freq=center_freq,
            half_bandwidth=half_bandwidth,
            boundary_conditions=["PML_8", "PML_8", "PEC", "PEC", "PEC", "PEC"],
            network=network,
            field_dumps=efields,
        )
        sim.finalize_structure(
            simulation_bounds=[
                -port_len,
                wg_len + port_len,
                -wg["a"] / 2,
                wg["a"] / 2,
                -wg["b"] / 2,
                wg["b"] / 2,
            ],
        )
        return sim
