# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Matt Huszagh (huszaghmatt@gmail.com)
# Copyright (C) 2015,2016 Thorsten Liebig (Thorsten.Liebig@gmx.de)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

"""
A port is an abstraction of an electrical port.  It provides an entry
point to an electrical network and performs the calculations needed to
determine S-parameters and other useful quantities.

This module overlaps significantly with OpenEMS's own implementation
of ports.  It reimplements many parts and, in other cases, delegates
to it for functionality.  The reason for reimpleenting this
functionality here is to allow better integration with automatic mesh
generation and to allow more flexible port creation.  OpenEMS
generally requires that a mesh be defined before the constituent
ports, which prevents mesh generation from being port-aware.  Port
awareness is necessary for proper mesh generation because ports
contain physical structures in addition to their non-physical
structures.  There are good reasons for generating ports based on an
existing mesh (e.g. voltage probes must be placed on mesh lines and
current probes must be placed between mesh lines, etc.), but this
module takes the position that we can always modify probe positions
after mesh generation.
"""

from typing import Tuple
from copy import deepcopy
from abc import ABC, abstractmethod
import numpy as np
from pyems.simulation import Simulation
from pyems.coordinate import Box3, Coordinate3, Axis
from pyems.mesh import Mesh
from pyems.probe import Probe
from pyems.utilities import array_index
from pyems.calc import wavenumber
from pyems.feed import Feed
from pyems.priority import priorities
from pyems.physical_constant import Z0


class Port(ABC):
    """
    Port base class.
    """

    def __init__(self, sim: Simulation, number: int, excite: bool = False):
        """
        :param sim: The Simulation to which this port is added.
        :param excite: Set to True if this port should generate an
            excitation.  The actual excitation type is set by the
            `Simulation` object that contains this port.
        """
        self._sim = sim
        self._number = number
        self.excite = excite
        self.feeds = []
        self.vprobes = []
        self.iprobes = []
        self._data_read = False
        self.v_inc = None
        self.v_ref = None
        self.i_inc = None
        self.i_ref = None
        self.p_inc = None
        self.p_ref = None
        self.z0 = None
        self.beta = None

        self.sim.add_port(self)

    @property
    def sim(self) -> Simulation:
        """
        """
        return self._sim

    @property
    def number(self) -> int:
        """
        Port number.
        """
        return self._number

    def snap_to_mesh(self, mesh: Mesh) -> None:
        """
        Generate the probes and feed and ensure they're located correctly
        in relation to the mesh.  You must call this in order to get
        correct simulation behavior.
        """
        self._set_probes(mesh=mesh)
        self._set_feed(mesh=mesh)

    def pml_overlap(self) -> bool:
        """
        Indicate if a probe or feed overlaps a PML boundary.

        :returns: True if there is overlap.
        """
        for vprobe in self.vprobes:
            if vprobe.pml_overlap():
                return True
        for iprobe in self.iprobes:
            if iprobe.pml_overlap():
                return True
        for feed in self.feeds:
            if feed.pml_overlap():
                return True
        return False

    def incident_voltage(self) -> np.array:
        """
        Get the incident voltage.  This can be used to calculate
        S-parameters.

        :returns: Array of the incident voltage magnitude for each
            frequency bin.
        """
        if not self._data_readp():
            raise RuntimeError("Must call calc() before retreiving values.")
        return self.v_inc

    def reflected_voltage(self) -> np.array:
        """
        Get the reflected voltage.  This can be used to calculate
        S-parameters.

        :returns: Array of the reflected voltage magnitude for each
            frequency bin.
        """
        if not self._data_readp():
            raise RuntimeError("Must call calc() before retreiving values.")
        return self.v_ref

    def incident_current(self) -> np.array:
        """
        Get the incident current.

        :returns: Array of the incident current magnitude for each
            frequency bin.
        """
        if not self._data_readp():
            raise RuntimeError("Must call calc() before retreiving values.")
        return self.i_inc

    def reflected_current(self) -> np.array:
        """
        Get the reflected current.

        :returns: Array of the reflected current magnitude for each
            frequency bin.
        """
        if not self._data_readp():
            raise RuntimeError("Must call calc() before retreiving values.")
        return self.i_ref

    def impedance(self, freq: float = None) -> np.array:
        """
        Get the characteristic impedance.

        :param freq: The frequency at which the impedance should be
            calculated.  If the the provided frequency doesn't
            correspond to a frequency bin, the nearest frequency bin
            will be selected.  If left as the default None, all
            frequency values will be given.

        :returns: Array of the characteristic impedance magnitude for
                  each frequency bin.
        """
        if not self._data_readp():
            raise RuntimeError("Must call calc() before retreiving values.")
        if freq is None:
            return self.z0

        idx = array_index(freq, self.sim.freq)
        return self.z0[idx]

    def incident_power(self) -> np.array:
        """
        Get the incident power.  This is generally useful for
        calculating efficiency.

        :returns: Array of the incident power magnitude for each
            frequency bin.
        """
        if not self._data_readp():
            raise RuntimeError("Must call calc() before retreiving values.")
        return self.p_inc

    def reflected_power(self) -> np.array:
        """
        Get the reflected power.

        :returns: Array of the reflected power magnitude for each
            frequency bin.
        """
        if not self._data_readp():
            raise RuntimeError("Must call calc() before retreiving values.")
        return self.p_ref

    def _calc_v_inc(self, v, i) -> None:
        """
        Calculate the incident voltage.

        See Pozar 4e section 4.3 (p.185) for derivation.

        :param v: Total voltage.
        :param i: Total current.
        """
        self.v_inc = (v + (self._ref_impedance * i)) / 2

    def _calc_v_ref(self, v, i) -> None:
        """
        Calculate the reflected voltage.

        See Pozar 4e section 4.3 (p.185) for derivation.

        :param v: Total voltage.
        :param i: Total current.
        """
        self.v_ref = (v - (self._ref_impedance * i)) / 2

    def _calc_i_inc(self, v, i) -> None:
        """
        Calculate the incident current.

        :param v: Total voltage.
        :param i: Total current.
        """
        self.i_inc = (i + (v / self._ref_impedance)) / 2

    def _calc_i_ref(self, v, i) -> None:
        """
        Calculate the reflected current.

        :param v: Total voltage.
        :param i: Total current.
        """
        self.i_ref = ((v / self._ref_impedance) - i) / 2

    def _calc_p_inc(self) -> None:
        """
        Calculate the port's incident power wave.
        """
        if self.v_inc is None or self.i_inc is None:
            raise RuntimeError(
                "Must calculate incoming and reflected "
                "voltages / current before power."
            )
        else:
            self.p_inc = (1 / 2) * np.real(self.v_inc * np.conj(self.i_inc))

    def _calc_p_ref(self) -> None:
        """
        Calculate the port's reflected power wave.
        """
        if self.v_ref is None or self.i_ref is None:
            raise RuntimeError(
                "Must calculate incoming and reflected "
                "voltages / current before power."
            )
        else:
            self.p_ref = (1 / 2) * np.real(self.v_ref * np.conj(self.i_ref))

    def _calc_beta(self, v, i, dv, di) -> None:
        """
        Calculate the transmission line propagation constant.

        Use tx line equations (see Pozar ch.2 for derivation):

        ..  math:: dV/dz = -(R+jwL)I

        ..  math:: dI/dz = -(G+jwC)V

        ..  math:: \gamma = \sqrt{(R+jwL)(G+jwC)}
        """
        self.beta = np.sqrt(-dv * di / (i * v))
        self.beta[np.real(self.beta) < 0] *= -1

    def _calc_z0(self, v, i, dv, di) -> None:
        """
        Calculate the transmission line characteristic impedance.

        Use tx line equations (see Pozar ch.2 for derivation):

        ..  math:: dV/dz = -(R+jwL)I

        ..  math:: dI/dz = -(G+jwC)V

        ..  math:: Z0 = \sqrt{(R+jwL)/(G+jwC)}
        """
        self.z0 = np.sqrt(v * dv / (i * di))
        if self._ref_impedance is None:
            self._ref_impedance = self.z0

    def _probe_vi(self) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Read probe voltage and current values.

        :returns: (v, i, dv, di)
        """
        [vprobe.read() for vprobe in self.vprobes]
        [iprobe.read() for iprobe in self.iprobes]
        prop_axis = self.propagation_axis().axis
        self._data_read = True
        v = self.vprobes[1].get_freq_data()[1]
        i = 0.5 * (
            self.iprobes[0].get_freq_data()[1]
            + self.iprobes[1].get_freq_data()[1]
        )
        dv = (
            self.vprobes[2].get_freq_data()[1]
            - self.vprobes[0].get_freq_data()[1]
        ) / (
            self.sim.unit
            * np.abs(
                self.vprobes[2].box.min_corner[prop_axis]
                - self.vprobes[0].box.min_corner[prop_axis]
            )
        )
        di = (
            self.iprobes[1].get_freq_data()[1]
            - self.iprobes[0].get_freq_data()[1]
        ) / (
            self.sim.unit
            * np.abs(
                self.iprobes[1].box.min_corner[prop_axis]
                - self.iprobes[0].box.min_corner[prop_axis]
            )
        )

        return (v, i, dv, di)

    def calc(self) -> None:
        """
        Calculate the characteristic impedance, propagation constant,
        and incident and reflected power.
        """
        v, i, dv, di = self._probe_vi()

        self._calc_beta(v, i, dv, di)
        self._calc_z0(v, i, dv, di)
        self._calc_v_inc(v, i)
        self._calc_v_ref(v, i)
        self._calc_i_inc(v, i)
        self._calc_i_ref(v, i)
        self._calc_p_inc()
        self._calc_p_ref()

    @abstractmethod
    def propagation_axis(self) -> Axis:
        """
        """
        pass

    def _data_readp(self) -> bool:
        """
        Return True if data has been read from simulation result.
        """
        return self._data_read


class MicrostripPort(Port):
    """
    Microstrip transmission line port.
    """

    def __init__(
        self,
        sim: Simulation,
        box: Box3,
        propagation_axis: Axis,
        excitation_axis: Axis,
        number: int,
        thickness: float,
        conductivity: float = 5.8e7,
        excite: bool = False,
        feed_impedance: float = None,
        feed_shift: float = 0.2,
        ref_impedance: float = None,
        measurement_shift: float = 0.5,
    ):
        """
        :param sim: Simulation to which microstrip port is added.
        :param box: 3D box specifying the port dimensions.  The trace
            dimensions are given by the rectangle perpendicular to the
            `excitation_axis` and at the max value for that excitation
            axis.  The excitation will dimensions are from the minimum
            `exctitation_axis` value to the maximum value.  The order
            of coordinates does not matter.
        :param propagation_axis: Specifies the coordinate axis and
            direction in which the signal propagation occurs.
        :param excitation_axis: Axis and direction of signal
            excitation.
        :param thickness: Metal trace thickness.  Units are whatever
            you set the CSX unit to, defaults to m.
        :param conductivity: Metal conductivity (in S/m).  The default
            uses the conductivity of copper.
        :param feed_impedance: The feeding impedance value.  The
            default value of None creates an infinite impedance.  If
            you use the default value ensure that the port is
            terminated by a PML.  When performing a characteristic
            impedance measurement use the default value and PML, which
            gives better results than attempting to use a matching
            impedance.
        :param feed_shift: The amount by which to shift the feed as a
            fraction of the total port length.  The final position
            will be influenced by this value but adjusted for the mesh
            used.
        :param ref_impedance: The impedance used to calculate the port
            voltage and current values.  If left as the default value
            of None, the calculated characteristic impedance is used
            to calculate these values.
        :param measurement_shift: The amount by which to shift the
            measurement probes as a fraction of the total port length.
            By default, the measurement port is placed halfway between
            the start and stop.  Like `feed_shift`, the final position
            will be adjusted for the mesh used.  This is important
            since voltage probes need to lie on mesh lines and current
            probes need to be placed equidistant between them.
        """
        super().__init__(sim=sim, number=number, excite=excite)
        self._box = box
        self._box.set_increasing()
        self._propagation_axis = propagation_axis
        self._excitation_axis = excitation_axis
        self._check_axes_perpendicular()
        self.thickness = thickness
        self.conductivity = conductivity
        self.feed_impedance = feed_impedance
        self.feed_shift = feed_shift
        self._ref_impedance = ref_impedance
        self.measurement_shift = measurement_shift

        self._set_trace()

    @property
    def box(self) -> Box3:
        """
        """
        return self._box

    def propagation_axis(self) -> Axis:
        """
        """
        return self._propagation_axis

    def get_feed_shift(self) -> float:
        """
        """
        return self.feed_shift

    def _check_axes_perpendicular(self) -> None:
        """
        """
        if self._propagation_axis.axis == self._excitation_axis.axis:
            raise ValueError(
                "Excitation and propagation axes must be perpendicular."
            )

    def _set_trace(self) -> None:
        """
        Set trace.
        """
        trace_prop = self._sim.csx.AddConductingSheet(
            "Microstrip_Trace_" + str(self.number),
            conductivity=self.conductivity,
            thickness=self.thickness,
        )
        box = self._trace_box()
        trace_prop.AddBox(
            priority=priorities["trace"], start=box.start(), stop=box.stop(),
        )

    def _trace_box(self) -> Box3:
        """
        Get the trace box.
        """
        trace_box = deepcopy(self.box)
        excitation_axis = self._excitation_axis.axis
        if self._excitation_axis.is_positive_direction():
            trace_box.min_corner[excitation_axis] = trace_box.max_corner[
                excitation_axis
            ]
        else:
            trace_box.max_corner[excitation_axis] = trace_box.min_corner[
                excitation_axis
            ]

        return trace_box

    def _propagation_direction(self) -> int:
        """
        Get the direction of the signal propagation.
        """
        return self._propagation_axis.direction

    def _excitation_direction(self) -> int:
        """
        Get the direction of the signal excitation.
        """
        return self._excitation_axis.direction

    def _trace_perpendicular_axis(self) -> Axis:
        """
        """
        axes = [0, 1, 2]
        axes.remove(self._propagation_axis.axis)
        axes.remove(self._excitation_axis.axis)
        trace_perp_axis = axes[0]
        if self._propagation_axis.is_positive_direction():
            return Axis(trace_perp_axis)
        else:
            return Axis(trace_perp_axis, -1)

    def _set_feed(self, mesh: Mesh) -> None:
        """
        Set excitation feed.
        """
        excite_type = None
        if self.excite:
            excite_type = 0

        feed = Feed(
            sim=self._sim,
            box=self._feed_box(mesh),
            excite_direction=self._excitation_axis.as_list(),
            excite_type=excite_type,
            impedance=self.feed_impedance,
        )
        self.feeds.append(feed)

    def _set_probes(self, mesh: Mesh) -> None:
        """
        Set measurement probes.
        """
        trace_box = deepcopy(self._trace_box())
        trace_perp_axis = self._trace_perpendicular_axis().axis
        trace_perp_low = trace_box.min_corner[trace_perp_axis]
        trace_perp_high = trace_box.max_corner[trace_perp_axis]
        trace_perp_mid = np.average([trace_perp_low, trace_perp_high])

        excitation_axis = self._excitation_axis.axis
        if self._excitation_axis.is_positive_direction():
            gnd_pos = self.box.min_corner[excitation_axis]
            trace_pos = self.box.max_corner[excitation_axis]
        else:
            gnd_pos = self.box.max_corner[excitation_axis]
            trace_pos = self.box.min_corner[excitation_axis]

        prop_axis = self._propagation_axis.axis
        trace_prop_low = trace_box.min_corner[prop_axis]
        trace_prop_high = trace_box.max_corner[prop_axis]
        if self._propagation_axis.is_positive_direction():
            prop_index, vxmid = mesh.nearest_mesh_line(
                prop_axis,
                trace_box.min_corner[prop_axis]
                + (
                    self.measurement_shift * (trace_prop_high - trace_prop_low)
                ),
            )
        else:
            prop_index, vxmid = mesh.nearest_mesh_line(
                prop_axis,
                trace_box.max_corner[prop_axis]
                - (
                    self.measurement_shift * (trace_prop_high - trace_prop_low)
                ),
            )
        mesh.set_lines_equidistant(0, prop_index - 1, prop_index + 1)

        v_prop_pos = [
            mesh.get_mesh_line(
                prop_axis, prop_index - self._propagation_direction()
            ),
            mesh.get_mesh_line(prop_axis, prop_index),
            mesh.get_mesh_line(
                prop_axis, prop_index + self._propagation_direction()
            ),
        ]
        i_prop_pos = [
            (v_prop_pos[0] + v_prop_pos[1]) / 2,
            (v_prop_pos[1] + v_prop_pos[2]) / 2,
        ]

        for idx in range(3):
            box = Box3(
                Coordinate3(None, None, None), Coordinate3(None, None, None)
            )
            box.min_corner[prop_axis] = v_prop_pos[idx]
            box.max_corner[prop_axis] = v_prop_pos[idx]
            box.min_corner[trace_perp_axis] = trace_perp_mid
            box.max_corner[trace_perp_axis] = trace_perp_mid
            box.min_corner[excitation_axis] = gnd_pos
            box.max_corner[excitation_axis] = trace_pos
            self.vprobes.append(
                Probe(
                    sim=self._sim,
                    box=box,
                    p_type=0,
                    weight=self._excitation_direction(),
                )
            )

        for idx in range(2):
            box = Box3(
                Coordinate3(None, None, None), Coordinate3(None, None, None)
            )
            box.min_corner[prop_axis] = i_prop_pos[idx]
            box.max_corner[prop_axis] = i_prop_pos[idx]
            box.min_corner[trace_perp_axis] = trace_perp_low
            box.max_corner[trace_perp_axis] = trace_perp_high
            box.min_corner[excitation_axis] = trace_pos
            box.max_corner[excitation_axis] = trace_pos
            self.iprobes.append(
                Probe(
                    sim=self._sim,
                    box=box,
                    p_type=1,
                    normal_axis=self._propagation_axis,
                    weight=-self._propagation_direction(),  # TODO negative??
                )
            )

    def _feed_box(self, mesh: Mesh) -> Box3:
        """
        Get the excitation feed box.
        """
        box = deepcopy(self.box)
        feed_axis = self._excitation_axis.axis
        if not self._excitation_axis.is_positive_direction():
            old_max = box.max_corner[feed_axis]
            box.max_corner[feed_axis] = box.min_corner[feed_axis]
            box.min_corner[feed_axis] = old_max

        prop_axis = self._propagation_axis.axis
        prop_dist = (
            self.box.max_corner[prop_axis] - self.box.min_corner[prop_axis]
        )
        if self._propagation_axis.is_positive_direction():
            _, prop_pos = mesh.nearest_mesh_line(
                prop_axis,
                self.box.min_corner[prop_axis] + (self.feed_shift * prop_dist),
            )
        else:
            _, prop_pos = mesh.nearest_mesh_line(
                prop_axis,
                self.box.max_corner[prop_axis] - (self.feed_shift * prop_dist),
            )

        box.max_corner[prop_axis] = prop_pos
        box.min_corner[prop_axis] = prop_pos

        return box


class DifferentialMicrostripPort(Port):
    """
    """

    def __init__(
        self,
        sim: Simulation,
        box: Box3,
        propagation_axis: Axis,
        excitation_axis: Axis,
        number: int,
        gap: float,
        thickness: float,
        conductivity: float = 5.8e7,
        excite: bool = False,
        feed_impedance: float = None,
        feed_shift: float = 0.2,
        ref_impedance: float = None,
        measurement_shift: float = 0.5,
    ):
        """
        :param sim: Simulation to which microstrip port is added.
        :param box: 3D box specifying the port dimensions.  This box
            fully encompasses both microstrip traces.  The length of
            each trace is identical and is equal to the length of the
            box in the direction of `propagation_axis`.  The width of
            the box in the direction of `excitation_axis` gives the
            distance between the outer edges of each trace.  The size
            of the box in the direction perpendicular to these two
            gives the elevation of the differential pair and must have
            size 0.  The order of coordinates in the box does not
            matter, since the direction of the port is set by the
            direction of `propagation_axis`.
        :param propagation_axis: Specifies the coordinate axis and
            direction in which the signal propagation occurs.
        :param excitation_axis: Axis and direction of signal
            excitation.  This should point from one trace to the other
            trace.  In other words, `propagation_axis` and
            `excitation_axis` (along with the box elevation) together
            determine the plane in which the differential traces
            reside.
        :param gap: Separation between inner edges of microstrip
            traces.
        :param thickness: Metal trace thickness.  Units are whatever
            you set the CSX unit to, defaults to m.
        :param conductivity: Metal conductivity (in S/m).  The default
            uses the conductivity of copper.
        :param feed_impedance: The feeding impedance value.  The
            default value of None creates an infinite impedance.  If
            you use the default value ensure that the port is
            terminated by a PML.  When performing a characteristic
            impedance measurement use the default value and PML, which
            gives better results than attempting to use a matching
            impedance.
        :param feed_shift: The amount by which to shift the feed as a
            fraction of the total port length.  The final position
            will be influenced by this value but adjusted for the mesh
            used.
        :param ref_impedance: The impedance used to calculate the port
            voltage and current values.  If left as the default value
            of None, the calculated characteristic impedance is used
            to calculate these values.
        :param measurement_shift: The amount by which to shift the
            measurement probes as a fraction of the total port length.
            By default, the measurement port is placed halfway between
            the start and stop.  Like `feed_shift`, the final position
            will be adjusted for the mesh used.  This is important
            since voltage probes need to lie on mesh lines and current
            probes need to be placed equidistant between them.
        """
        super().__init__(sim=sim, number=number, excite=excite)
        self._box = box
        self._box.set_increasing()
        self._propagation_axis = propagation_axis
        self._excitation_axis = excitation_axis
        self._check_axes_perpendicular()
        self._check_normal_axis_size()
        self._gap = gap
        self._thickness = thickness
        self._conductivity = conductivity
        self._feed_impedance = feed_impedance
        self._feed_shift = feed_shift
        self._ref_impedance = ref_impedance
        self._measurement_shift = measurement_shift

        self._set_traces()

    def propagation_axis(self) -> Axis:
        """
        """
        return self._propagation_axis

    def _set_traces(self) -> None:
        """
        """
        trace_prop = self._sim.csx.AddConductingSheet(
            "Differential_Microstrip_Trace_" + str(self.number),
            conductivity=self._conductivity,
            thickness=self._thickness,
        )
        for box in self._trace_boxes():
            trace_prop.AddBox(
                priority=priorities["trace"],
                start=box.start(),
                stop=box.stop(),
            )

    def _trace_boxes(self) -> Tuple[Box3, Box3]:
        """
        """
        lower_box = deepcopy(self._box)
        upper_box = deepcopy(self._box)
        excite_axis = self._excitation_axis.axis
        trace_width = self._trace_width()

        lower_box.max_corner[excite_axis] = (
            lower_box.min_corner[excite_axis] + trace_width
        )
        upper_box.min_corner[excite_axis] = (
            upper_box.max_corner[excite_axis] - trace_width
        )

        return (lower_box, upper_box)

    def _trace_width(self) -> float:
        """
        """
        box_width = (
            self._box.max_corner[self._excitation_axis.axis]
            - self._box.min_corner[self._excitation_axis.axis]
        )
        return (box_width - self._gap) / 2

    def _normal_axis(self) -> Axis:
        """
        """
        axes = [0, 1, 2]
        axes.remove(self._propagation_axis.axis)
        axes.remove(self._excitation_axis.axis)
        return Axis(axes[0])

    def _check_axes_perpendicular(self) -> None:
        """
        """
        if self._propagation_axis.axis == self._excitation_axis.axis:
            raise ValueError(
                "Excitation and propagation axes must be perpendicular."
            )

    def _check_normal_axis_size(self) -> None:
        """
        """
        normal_axis = self._normal_axis().axis
        if (
            self._box.max_corner[normal_axis]
            != self._box.min_corner[normal_axis]
        ):
            raise ValueError(
                "Size of box in direction normal to microstrip plane must be 0."
            )

    def _propagation_direction(self) -> int:
        """
        Get the direction of the signal propagation.
        """
        return self._propagation_axis.direction

    def _excitation_direction(self) -> int:
        """
        Get the direction of the signal excitation.
        """
        return self._excitation_axis.direction

    def _set_feed(self, mesh: Mesh) -> None:
        """
        Set excitation feed.
        """
        excite_type = None
        if self.excite:
            excite_type = 0

        feed = Feed(
            sim=self._sim,
            box=self._feed_box(mesh),
            excite_direction=self._excitation_axis.as_list(),
            excite_type=excite_type,
            impedance=self._feed_impedance,
        )
        self.feeds.append(feed)

    def _feed_box(self, mesh: Mesh) -> Box3:
        """
        Get the excitation feed box.
        """
        box = deepcopy(self._box)
        feed_axis = self._excitation_axis.axis
        trace_width = self._trace_width()
        box.max_corner[feed_axis] -= trace_width
        box.min_corner[feed_axis] += trace_width
        if not self._excitation_axis.is_positive_direction():
            old_max = box.max_corner[feed_axis]
            box.max_corner[feed_axis] = box.min_corner[feed_axis]
            box.min_corner[feed_axis] = old_max

        prop_axis = self._propagation_axis.axis
        prop_dist = (
            self._box.max_corner[prop_axis] - self._box.min_corner[prop_axis]
        )
        if self._propagation_axis.is_positive_direction():
            _, prop_pos = mesh.nearest_mesh_line(
                prop_axis,
                self._box.min_corner[prop_axis]
                + (self._feed_shift * prop_dist),
            )
        else:
            _, prop_pos = mesh.nearest_mesh_line(
                prop_axis,
                self._box.max_corner[prop_axis]
                - (self._feed_shift * prop_dist),
            )

        box.max_corner[prop_axis] = prop_pos
        box.min_corner[prop_axis] = prop_pos

        return box

    def _set_probes(self, mesh: Mesh) -> None:
        """
        Set measurement probes.
        """
        prop_axis = self._propagation_axis.axis
        excite_axis = self._excitation_axis.axis
        normal_axis = self._normal_axis().axis
        normal_pos = self._box.min_corner[normal_axis]
        trace_width = self._trace_width()
        excite_pos_lower = self._box.min_corner[excite_axis] + trace_width
        excite_pos_upper = self._box.max_corner[excite_axis] - trace_width

        prop_dist = (
            self._box.max_corner[prop_axis] - self._box.min_corner[prop_axis]
        )

        if self._propagation_axis.is_positive_direction():
            prop_index, _ = mesh.nearest_mesh_line(
                prop_axis,
                self._box.min_corner[prop_axis]
                + (self._measurement_shift * prop_dist),
            )
        else:
            prop_index, _ = mesh.nearest_mesh_line(
                prop_axis,
                self._box.max_corner[prop_axis]
                - (self._measurement_shift * prop_dist),
            )
        mesh.set_lines_equidistant(0, prop_index - 1, prop_index + 1)

        v_prop_pos = [
            mesh.get_mesh_line(
                prop_axis, prop_index - self._propagation_direction()
            ),
            mesh.get_mesh_line(prop_axis, prop_index),
            mesh.get_mesh_line(
                prop_axis, prop_index + self._propagation_direction()
            ),
        ]
        i_prop_pos = [
            (v_prop_pos[0] + v_prop_pos[1]) / 2,
            (v_prop_pos[1] + v_prop_pos[2]) / 2,
        ]

        for idx in range(3):
            box = Box3(
                Coordinate3(None, None, None), Coordinate3(None, None, None)
            )
            box.min_corner[prop_axis] = v_prop_pos[idx]
            box.max_corner[prop_axis] = v_prop_pos[idx]
            box.min_corner[normal_axis] = normal_pos
            box.max_corner[normal_axis] = normal_pos
            box.min_corner[excite_axis] = excite_pos_lower
            box.max_corner[excite_axis] = excite_pos_upper
            self.vprobes.append(
                Probe(
                    sim=self._sim,
                    box=box,
                    p_type=0,
                    weight=self._excitation_direction(),
                )
            )

        for idx in range(2):
            boxes = [None, None]
            boxes[0] = Box3(
                Coordinate3(None, None, None), Coordinate3(None, None, None)
            )
            boxes[1] = deepcopy(boxes[0])

            for box in boxes:
                box.min_corner[prop_axis] = i_prop_pos[idx]
                box.max_corner[prop_axis] = i_prop_pos[idx]
                box.min_corner[normal_axis] = normal_pos
                box.max_corner[normal_axis] = normal_pos

            boxes[0].min_corner[excite_axis] = (
                self._box.max_corner[excite_axis] - trace_width
            )
            boxes[0].max_corner[excite_axis] = self._box.max_corner[
                excite_axis
            ]
            boxes[1].min_corner[excite_axis] = self._box.min_corner[
                excite_axis
            ]
            boxes[1].max_corner[excite_axis] = (
                self._box.min_corner[excite_axis] + trace_width
            )

            for i, box in enumerate(boxes):
                self.iprobes.append(
                    Probe(
                        sim=self._sim,
                        box=box,
                        p_type=1,
                        normal_axis=self._propagation_axis,
                        weight=-self._propagation_direction()
                        * ((i + 1) ** -1),  # TODO negative prop direction??
                    )
                )

    def _probe_vi(self) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        Read probe voltage and current values.

        :returns: (v, i, dv, di)
        """
        [vprobe.read() for vprobe in self.vprobes]
        [iprobe.read() for iprobe in self.iprobes]
        prop_axis = self.propagation_axis().axis
        self._data_read = True
        v = self.vprobes[1].get_freq_data()[1]
        # TODO using both current probes gives slightly incorrect
        # results for some unknown reason.
        # # the currents on each trace should be exactly opposite, but
        # # we use both just in case.
        # i1 = 0.5 * (
        #     self.iprobes[0].get_freq_data()[1]
        #     - self.iprobes[1].get_freq_data()[1]
        # )
        # i2 = 0.5 * (
        #     self.iprobes[2].get_freq_data()[1]
        #     - self.iprobes[3].get_freq_data()[1]
        # )
        # i = 0.5 * (i1 + i2)
        i = 0.5 * (
            self.iprobes[0].get_freq_data()[1]
            + self.iprobes[2].get_freq_data()[1]
        )
        dv = (
            self.vprobes[2].get_freq_data()[1]
            - self.vprobes[0].get_freq_data()[1]
        ) / (
            self.sim.unit
            * np.abs(
                self.vprobes[2].box.min_corner[prop_axis]
                - self.vprobes[0].box.min_corner[prop_axis]
            )
        )
        # di = (i2 - i1) / (
        #     self.sim.unit
        #     * np.abs(
        #         self.iprobes[2].box.min_corner[prop_axis]
        #         - self.iprobes[0].box.min_corner[prop_axis]
        #     )
        # )
        di = (
            self.iprobes[2].get_freq_data()[1]
            - self.iprobes[0].get_freq_data()[1]
        ) / (
            self.sim.unit
            * np.abs(
                self.iprobes[2].box.min_corner[prop_axis]
                - self.iprobes[0].box.min_corner[prop_axis]
            )
        )

        return (v, i, dv, di)


class CPWPort(Port):
    """
    Coplanar waveguide transmission line port.
    """

    def __init__(
        self,
        sim: Simulation,
        box: Box3,
        gap: float,
        thickness: float,
        conductivity: float = 5.8e7,
        excite: bool = False,
        feed_impedance: float = None,
        feed_shift: float = 0.2,
        ref_impedance: float = None,
        measurement_shift: float = 0.5,
    ):
        """
        :param gap: Gap between adjacent ground planes and trace (in m).
        """
        self.gap = gap
        raise RuntimeError(
            "TODO CPWPort is not correctly implemented. "
            "See MicrostripPort for guidance on how to implement correctly."
        )
        super().__init__(
            sim=sim,
            box=box,
            thickness=thickness,
            conductivity=conductivity,
            excite=excite,
            feed_impedance=feed_impedance,
            feed_shift=feed_shift,
            ref_impedance=ref_impedance,
            measurement_shift=measurement_shift,
        )

    def calc(self) -> None:
        """
        Calculate the characteristic impedance, propagation constant,
        and incident and reflected power.
        """
        [vprobe.read() for vprobe in self.vprobes]
        [iprobe.read() for iprobe in self.iprobes]
        self._data_read = True
        v0 = 0.5 * (
            self.vprobes[0].get_freq_data()[1]
            + self.vprobes[1].get_freq_data()[1]
        )
        v1 = 0.5 * (
            self.vprobes[2].get_freq_data()[1]
            + self.vprobes[3].get_freq_data()[1]
        )
        v2 = 0.5 * (
            self.vprobes[4].get_freq_data()[1]
            + self.vprobes[5].get_freq_data()[1]
        )
        v = v1

        i = 0.5 * (
            self.iprobes[0].get_freq_data()[1]
            + self.iprobes[1].get_freq_data()[1]
        )
        dv = (v2 - v0) / (
            self.sim.unit
            * (
                self.vprobes[2].box.min_corner.x
                - self.vprobes[0].box.min_corner.x
            )
        )
        di = (
            self.iprobes[1].get_freq_data()[1]
            - self.iprobes[0].get_freq_data()[1]
        ) / (
            self.sim.unit
            * (
                self.iprobes[1].box.min_corner.x
                - self.iprobes[0].box.min_corner.x
            )
        )

        self._calc_beta(v, i, dv, di)
        self._calc_z0(v, i, dv, di)
        self._calc_v_inc(v, i)
        self._calc_v_ref(v, i)
        self._calc_i_inc(v, i)
        self._calc_i_ref(v, i)
        self._calc_p_inc()
        self._calc_p_ref()

    def _set_feed(self, mesh: Mesh) -> None:
        """
        Set excitation feed.
        """
        excite_type = None
        if self.excite:
            excite_type = 0

        # when the user specifies a feed impedance, this means they
        # want the cpw port terminated with that impedance. To get
        # the best termination, use a single feed extending the trace
        # line. The user must provide a ground plane 1 gap width
        # behind the trace end.
        if self.feed_impedance:
            trace_box = self._trace_box()
            feed = Feed(
                sim=self._sim,
                box=Box3(
                    Coordinate3(
                        trace_box.min_corner.x
                        - (self._propagation_direction() * self.gap),
                        trace_box.min_corner.y,
                        trace_box.min_corner.z,
                    ),
                    Coordinate3(
                        trace_box.min_corner.x,
                        trace_box.max_corner.y,
                        trace_box.max_corner.z,
                    ),
                ),
                excite_direction=[self._propagation_direction(), 0, 0],
                excite_type=excite_type,
                impedance=self.feed_impedance,
            )
            self.feeds.append(feed)

        else:
            for box, excite_dir in zip(
                self._get_feed_boxes(mesh), [[0, 1, 0], [0, -1, 0]]
            ):
                feed = Feed(
                    sim=self._sim,
                    box=box,
                    excite_direction=excite_dir,
                    excite_type=excite_type,
                    impedance=self.feed_impedance,
                )
                self.feeds.append(feed)

    def _get_feed_boxes(self, mesh: Mesh) -> None:
        """
        """
        _, xpos = mesh.nearest_mesh_line(
            0,
            self.box.min_corner.x
            + (
                self.feed_shift
                * (self.box.max_corner.x - self.box.min_corner.x)
            ),
        )
        feed_boxes = [
            Box3(Coordinate3(xpos, ystart, 0), Coordinate3(xpos, yend, 0))
            for ystart, yend in zip(
                [
                    self.box.min_corner.y
                    - (self._propagation_direction() * self.gap),
                    self.box.max_corner.y
                    + (self._propagation_direction() * self.gap),
                ],
                [self.box.min_corner.y, self.box.max_corner.y],
            )
        ]
        return feed_boxes

    def _set_probes(self, mesh: Mesh) -> None:
        """
        Set measurement probes.
        """
        trace_box = self._trace_box()
        trace_ylow = trace_box.min_corner.y
        trace_yhigh = trace_box.max_corner.y
        trace_z = trace_box.max_corner.z

        trace_xlow = trace_box.min_corner.x
        trace_xhigh = trace_box.max_corner.x
        x_index, vxmid = mesh.nearest_mesh_line(
            0,
            trace_box.min_corner.x
            + (self.measurement_shift * (trace_xhigh - trace_xlow)),
        )
        mesh.set_lines_equidistant(0, x_index - 1, x_index + 1)

        vxpos = [
            mesh.get_mesh_line(0, x_index - self._propagation_direction()),
            mesh.get_mesh_line(0, x_index),
            mesh.get_mesh_line(0, x_index + self._propagation_direction()),
        ]
        ixpos = [
            (vxpos[0] + vxpos[1]) / 2,
            (vxpos[1] + vxpos[2]) / 2,
        ]

        self.vprobes = []
        for xpos in vxpos:
            # TODO ensure probe weights set correctly
            self.vprobes.append(
                Probe(
                    sim=self._sim,
                    box=Box3(
                        Coordinate3(xpos, trace_ylow - self.gap, trace_z),
                        Coordinate3(xpos, trace_ylow, trace_z),
                    ),
                    p_type=0,
                    weight=self._excitation_direction(),
                )
            )
            self.vprobes.append(
                Probe(
                    sim=self._sim,
                    box=Box3(
                        Coordinate3(xpos, trace_yhigh, trace_z),
                        Coordinate3(xpos, trace_yhigh + self.gap, trace_z),
                    ),
                    p_type=0,
                    weight=-self._excitation_direction(),
                )
            )

        self.iprobes = [
            Probe(
                sim=self._sim,
                box=Box3(
                    Coordinate3(xpos, trace_ylow, trace_z),
                    Coordinate3(xpos, trace_yhigh, trace_z),
                ),
                p_type=1,
                normal_axis=Axis("x"),
                weight=-self._propagation_direction(),
            )
            for xpos in ixpos
        ]


class RectWaveguidePort(Port):
    """
    Rectangular waveguide port base class.
    """

    def __init__(
        self,
        sim: Simulation,
        box: Box3,
        propagation_axis: Axis,
        mode_name: str = "TE10",
        excite: bool = False,
        delay: float = 0,
        ref_impedance: float = None,
    ):
        """
        :param box: Waveguide box.  The waveguide width and height are
            computed from this box.
        :param propagation_axis: Direction the waveguide is facing.
            Set to 0, 1, or 2, for x, y, or z.
        :param mode_name: Waveguide propagation mode.  If you don't
            know this you probably want the default.  Otherwise, see
            Pozar (4e) p.110 for more information.
        :param excite: If True, add an excitation to the port.
        """
        super().__init__(sim, excite)
        self._box = box
        self._propagation_axis = propagation_axis
        self.mode_name = mode_name
        self.excite = excite
        self.delay = delay
        self._ref_impedance = ref_impedance

        # set later
        self.te = None
        self.e_func = [0, 0, 0]
        self.h_func = [0, 0, 0]
        self.kc = None
        self.direction = None
        self.a = None
        self.b = None

        self._set_width_height()
        self._set_direction()
        self._parse_mode_name()
        self._set_func()

    def propagation_axis(self) -> Axis:
        """
        """
        return self._propagation_axis

    @property
    def box(self) -> Box3:
        """
        """
        return self._box

    def calc(self):
        """
        Calculate the characteristic impedance, propagation constant,
        and incident and reflected voltage.

        TODO this doesn't account for reference plane shifts.  See
        p.184 of Pozar, or original openems ports.py file for this.
        """
        k = wavenumber(self.sim.freq, 1)  # use meters
        self._calc_cutoff_wavenumber()
        self._calc_beta(k)
        self._calc_z0(k)
        [vprobe.read() for vprobe in self.vprobes]
        [iprobe.read() for iprobe in self.iprobes]
        self._data_read = True
        v = self.vprobes[0].get_freq_data()[1]
        i = self.iprobes[0].get_freq_data()[1]
        self._calc_v_inc(v, i)
        self._calc_v_ref(v, i)
        self._calc_i_inc(v, i)
        self._calc_i_ref(v, i)
        self._calc_p_inc()
        self._calc_p_ref()

    def add_metal_shell(self, thickness: float) -> None:
        """
        """
        shell_prop = self._sim.csx.AddMetal("rect_wg_metal")
        back_face = self._shell_face_box(
            const_dim=self.propagation_axis().intval(),
            const_dim_idx=0,
            thickness=thickness,
        )
        shell_prop.AddBox(start=back_face.start(), stop=back_face.stop())
        dims = list(range(3))
        del dims[self.propagation_axis().intval()]
        for dim in dims:
            for i in range(2):
                face = self._shell_face_box(
                    const_dim=dim, const_dim_idx=i, thickness=thickness
                )
                shell_prop.AddBox(start=face.start(), stop=face.stop())

    def _shell_face_box(
        self, const_dim: int, const_dim_idx: int, thickness: float,
    ) -> Box3:
        """
        """
        box = Box3(
            Coordinate3(None, None, None), Coordinate3(None, None, None)
        )
        for dim in range(3):
            if dim == const_dim:
                box.min_corner[dim] = self.box[const_dim_idx][dim] + (
                    self._dim_outer_dir(dim, const_dim_idx) * thickness
                )
                box.max_corner[dim] = self.box[const_dim_idx][dim]
            else:
                box.min_corner[dim] = self.box.min_corner[dim] - thickness
                box.max_corner[dim] = self.box.max_corner[dim] + thickness

        return box

    def _dim_outer_dir(self, dim: int, dim_idx: int) -> int:
        """
        """
        other_dim_idx = (dim_idx + 1) % 2
        return np.sign(self.box[dim_idx][dim] - self.box[other_dim_idx][dim])

    def _set_width_height(self) -> None:
        """
        Set the waveguide width (a) and height (b) based on the box
        dimensions and propagation axis.  The width is always taken to
        be the larger of the two.
        """
        dimensions = [
            np.abs(self.box.max_corner[dim] - self.box.min_corner[dim])
            for dim in range(3)
        ]
        del dimensions[self.propagation_axis().intval()]
        self.a = np.amax(dimensions)
        self.b = np.amin(dimensions)

    def _parse_mode_name(self) -> None:
        """
        Parse mode_name argument to extract mode information.
        """
        assert len(self.mode_name) == 4, "Invalid mode definition"
        if self.mode_name.startswith("TE"):
            self.te = True
        else:
            self.te = False
        self.m = float(self.mode_name[2])
        self.n = float(self.mode_name[3])

        assert (
            self.te
        ), "Currently only TE-modes are supported! Mode found: {}".format(
            self.mode_name
        )

    def _set_direction(self) -> None:
        """
        Compute whether the port faces in the positive or negative
        direction of propagation_axis.
        """
        self.direction = int(
            np.sign(
                self.box.max_corner[self.propagation_axis().intval()]
                - self.box.min_corner[self.propagation_axis().intval()]
            )
        )

    def _set_func(self) -> None:
        """
        """
        ny_p = (self.propagation_axis().intval() + 1) % 3
        ny_pp = (self.propagation_axis().intval() + 2) % 3

        xyz = "xyz"
        if self.box.min_corner[ny_p] != 0:
            name_p = "({}-{})".format(xyz[ny_p], self.box.min_corner[ny_p])
        else:
            name_p = xyz[ny_p]
        if self.box.min_corner[ny_pp] != 0:
            name_pp = "({}-{})".format(xyz[ny_p], self.box.min_corner[ny_p])
        else:
            name_pp = xyz[ny_p]

        if self.n > 0:
            self.e_func[ny_p] = "{}*cos({}*{})*sin({}*{})".format(
                self.n / self.b,
                self.m * np.pi / self.a,
                name_p,
                self.n * np.pi / self.b,
                name_pp,
            )
        if self.m > 0:
            self.e_func[ny_pp] = "{}*sin({}*{})*cos({}*{})".format(
                -self.m / self.a,
                self.m * np.pi / self.a,
                name_p,
                self.n * np.pi / self.b,
                name_pp,
            )

        if self.m > 0:
            self.h_func[ny_p] = "{}*sin({}*{})*cos({}*{})".format(
                self.m / self.a,
                self.m * np.pi / self.a,
                name_p,
                self.n * np.pi / self.b,
                name_pp,
            )
        if self.n > 0:
            self.h_func[ny_pp] = "{}*cos({}*{})*sin({}*{})".format(
                self.n / self.b,
                self.m * np.pi / self.a,
                name_p,
                self.n * np.pi / self.b,
                name_pp,
            )

    def _calc_beta(self, k) -> None:
        """
        Calculate the propagation constant.
        """
        self.beta = np.sqrt(np.power(k, 2) - np.power(self.kc, 2))

    def _calc_z0(self, k) -> None:
        """
        Calculate the characteristic impedance.
        """
        self.z0 = k * Z0 / self.beta
        if self._ref_impedance is None:
            self._ref_impedance = self.z0

    def _set_probes(self, mesh: Mesh) -> None:
        """
        Set measurement probes.
        """
        _, prop_pos = mesh.nearest_mesh_line(
            self.propagation_axis().intval(),
            self.box[1][self.propagation_axis().intval()],
        )

        probe_box = deepcopy(self.box)
        probe_box.min_corner[self.propagation_axis().intval()] = prop_pos
        probe_box.max_corner[self.propagation_axis().intval()] = prop_pos

        self.vprobes = [
            Probe(
                sim=self._sim,
                box=probe_box,
                p_type=10,
                mode_function=self.e_func,
            )
        ]

        self.iprobes = [
            Probe(
                sim=self._sim,
                box=probe_box,
                p_type=11,
                weight=self.direction,
                mode_function=self.h_func,
            )
        ]

    def _set_feed(self, mesh: Mesh) -> None:
        """
        Set excitation feed.
        """
        if self.excite:
            _, prop_pos = mesh.nearest_mesh_line(
                self.propagation_axis().intval(),
                self.box[0][self.propagation_axis().intval()],
            )

            feed_box = deepcopy(self.box)
            feed_box.min_corner[self.propagation_axis().intval()] = prop_pos
            feed_box.max_corner[self.propagation_axis().intval()] = prop_pos

            feed_vec = np.ones(3)
            feed_vec[self.propagation_axis().intval()] = 0
            weight_func = [str(x) for x in self.e_func]
            feed = Feed(
                sim=self._sim,
                box=feed_box,
                excite_direction=feed_vec,
                excite_type=0,
                weight_func=weight_func,
                delay=self.delay,
            )
            self.feeds.append(feed)

    def _calc_cutoff_wavenumber(self) -> None:
        """
        Calculate the minimum wavenumber for wave to propagate.

        See Pozar (4e) p.112 for derivation.
        """
        a = self.a * self.sim.unit
        b = self.b * self.sim.unit
        self.kc = np.sqrt(
            np.power((self.m * np.pi / a), 2)
            + np.power((self.n * np.pi / b), 2)
        )


class CoaxPort(Port):
    """
    Coaxial port.  This does not construct a coaxial cable.  For that,
    use Coax in structure.py.
    """

    def __init__(
        self,
        sim: Simulation,
        number: int,
        start: Coordinate3,
        stop: Coordinate3,
        radius: float,
        core_radius: float,
        excite: bool = False,
        feed_shift: float = 0.2,
        feed_impedance: float = None,
        measurement_shift: float = 0.5,
        delay: float = 0,
        ref_impedance: float = None,
    ):
        """
        :param start: Starting position of the coaxial port at its
            radial center.
        :param stop: Ending position of the coaxial port at its radial
            center.  `start` and `stop` can only differ in one
            dimension.  In other words, the coaxial port must be
            parallel to a coordinate axis.
        :param radius: Distance between center and outer conductor for
            a cross section of the coaxial cable.
        :param core_radius: Radius of the inner copper core.
        :param excite: If True, add an excitation to the port.
        :param feed_shift: Offsets feed from starting position.  Given
            as a proportion of the total length.
        :param feed_impedance: The feeding impedance value.  The
            default value of None creates an infinite impedance.  If
            you use the default value ensure that the port is
            terminated by a PML.  When performing a characteristic
            impedance measurement use the default value and PML, which
            gives better results than attempting to use a matching
            impedance.
        :param measurement_shift: Measurement probes offset.  Given as
            a proportion of the total length.
        :param delay:
        :param ref_impedance: Impedance used to calculate port
            parameters.  This does not affect the calculated
            characteristic impedance.
        """
        super().__init__(sim=sim, number=number, excite=excite)
        self._start = start
        self._stop = stop
        self._check_start_stop()
        self._radius = radius
        self._core_radius = core_radius
        self._excite = excite
        self._feed_shift = feed_shift
        self._feed_impedance = feed_impedance
        self._measurement_shift = measurement_shift
        self._delay = delay
        self._ref_impedance = ref_impedance

        self._set_core()

    def _check_start_stop(self) -> None:
        """
        """
        diff = np.diff(
            [self._stop.coordinate_list(), self._start.coordinate_list()],
            axis=0,
        )
        if np.count_nonzero(diff) != 1:
            raise ValueError(
                "Invalid start and stop coordinates. Port must be parallel "
                "to a coordinate axis."
            )

    def _propagation_axis(self) -> Axis:
        """
        """
        for i in range(3):
            if np.isclose(self._start[i], self._stop[i]):
                return Axis(i)
        raise RuntimeError("Unable to determine propagation axis.")

    def _direction(self) -> int:
        """
        Compute whether the port faces in the positive or negative
        direction of propagation_axis.
        """
        prop_axis = self._propagation_axis().axis
        direction = int(
            np.sign(self._stop[prop_axis] - self._start[prop_axis])
        )
        return direction

    def _set_core(self) -> None:
        """
        """
        # TODO bug?? cylinder ignored if start > stop
        if self._direction() == 1:
            start = self._start.coordinate_list()
            stop = self._stop.coordinate_list()
        else:
            start = self._stop.coordinate_list()
            stop = self._start.coordinate_list()

        core_prop = self.sim.csx.AddMetal(self._core_name())
        core_prop.AddCylinder(
            start=start,
            stop=stop,
            radius=self._core_radius,
            priority=priorities["trace"],
        )

    def propagation_axis(self) -> Axis:
        """
        """
        diff = np.diff(
            [self._stop.coordinate_list(), self._start.coordinate_list()],
            axis=0,
        )[0]
        for i in range(3):
            if diff[i] != 0:
                return Axis(i)

    def _direction(self) -> int:
        """
        +1 for positive direction in propagation axis direction and -1
        for negative direction.
        """
        axis_int = self.propagation_axis().intval()
        return int(np.sign(self._stop[axis_int] - self._start[axis_int]))

    def _core_name(self) -> str:
        """
        """
        return "Coax_core_" + str(self.number)

    def _set_probes(self, mesh: Mesh) -> None:
        """
        Set measurement probes.
        """
        prop_axis = self.propagation_axis().intval()
        pos = (
            self._measurement_shift
            * (self._stop[prop_axis] - self._start[prop_axis])
        ) + self._start[prop_axis]
        mid_idx, mid_pos = mesh.nearest_mesh_line(prop_axis, pos)
        mesh.set_lines_equidistant(prop_axis, mid_idx - 1, mid_idx + 1)
        low_idx = mid_idx - self._direction()
        high_idx = mid_idx + self._direction()

        vlow = mesh.get_mesh_line(prop_axis, low_idx)
        vmid = mesh.get_mesh_line(prop_axis, mid_idx)
        vhigh = mesh.get_mesh_line(prop_axis, high_idx)
        vpos = [vlow, vmid, vhigh]

        ilow = np.average([vlow, vmid])
        ihigh = np.average([vmid, vhigh])
        ipos = [ilow, ihigh]

        for pos in vpos:
            box = Box3(
                Coordinate3(None, None, None), Coordinate3(None, None, None)
            )
            box.min_corner[prop_axis] = pos
            box.max_corner[prop_axis] = pos

            other_axis1 = (prop_axis + 1) % 3
            box.min_corner[other_axis1] = (
                self._start[other_axis1] - self._radius
            )
            box.max_corner[other_axis1] = self._start[other_axis1]

            other_axis2 = (prop_axis + 2) % 3
            box.min_corner[other_axis2] = self._start[other_axis2]
            box.max_corner[other_axis2] = self._start[other_axis2]
            self.vprobes.append(
                Probe(sim=self.sim, box=box, p_type=0, weight=1)
            )

        for pos in ipos:
            box = Box3(
                Coordinate3(None, None, None), Coordinate3(None, None, None)
            )
            box.min_corner[prop_axis] = pos
            box.max_corner[prop_axis] = pos

            other_axis1 = (prop_axis + 1) % 3
            box.min_corner[other_axis1] = (
                self._start[other_axis1] - self._core_radius
            )
            box.max_corner[other_axis1] = (
                self._start[other_axis1] + self._core_radius
            )

            other_axis2 = (prop_axis + 2) % 3
            box.min_corner[other_axis2] = (
                self._start[other_axis2] - self._core_radius
            )
            box.max_corner[other_axis2] = (
                self._start[other_axis2] + self._core_radius
            )
            self.iprobes.append(
                Probe(
                    sim=self.sim,
                    box=box,
                    p_type=1,
                    normal_axis=self.propagation_axis(),
                    weight=-self._direction(),  # TODO negative??
                )
            )

    def _set_feed(self, mesh: Mesh) -> None:
        """
        Set excitation feed.
        """
        if self._excite:
            excite_type = 0
        else:
            excite_type = None

        prop_axis = self.propagation_axis().intval()
        pos = (
            self._direction()
            * self._feed_shift
            * (self._stop[prop_axis] - self._start[prop_axis])
        ) + self._start[prop_axis]
        _, feed_pos = mesh.nearest_mesh_line(prop_axis, pos)

        box = Box3(
            Coordinate3(None, None, None), Coordinate3(None, None, None)
        )
        box.min_corner[prop_axis] = feed_pos
        box.max_corner[prop_axis] = feed_pos

        other_axis1 = (prop_axis + 1) % 3
        box.min_corner[other_axis1] = self._start[other_axis1] - self._radius
        box.max_corner[other_axis1] = self._start[other_axis1]

        other_axis2 = (prop_axis + 2) % 3
        box.min_corner[other_axis2] = self._start[other_axis2]
        box.max_corner[other_axis2] = self._start[other_axis2]

        excite_direction = [0, 0, 0]
        excite_direction[other_axis1] = 1
        feed = Feed(
            sim=self.sim,
            box=box,
            excite_direction=excite_direction,
            excite_type=excite_type,
            impedance=self._feed_impedance,
            delay=self._delay,
        )
        self.feeds.append(feed)
