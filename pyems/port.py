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

from abc import ABC, abstractmethod
from typing import List
import numpy as np
from CSXCAD.CSXCAD import ContinuousStructure
from CSXCAD.CSTransform import CSTransform
from pyems.automesh import Mesh
from pyems.probe import Probe
from pyems.utilities import max_priority, wavenumber, get_unit
from pyems.feed import Feed


C0 = 299792458  # m/s
MUE0 = 4e-7 * np.pi  # N/A^2
EPS0 = 1 / (MUE0 * C0 ** 2)  # F/m
# free space wave impedance
Z0 = np.sqrt(MUE0 / EPS0)  # Ohm


class Port(ABC):
    """
    Port base class.
    """

    # Used to provide unique filenames, since openems requires that we
    # specify these.
    unique_ctr = 0

    def __init__(
        self, csx: ContinuousStructure, excite: bool = False,
    ):
        """
        :param csx: The CSXCAD ContinuousStructure to which this port
            is added.
        :param excite: Set to True if this port should generate an
            excitation.  The actual excitation type is set by the
            `Simulation` object that contains this port.
        """
        self.csx = csx
        self.excite = excite
        self.unit = get_unit(self.csx)
        self.feeds = []
        self.vprobes = []
        self.iprobes = []
        self.freq = None
        self.v_inc = None
        self.v_ref = None
        self.i_inc = None
        self.i_ref = None
        self.p_inc = None
        self.p_ref = None
        self.z0 = None
        self.beta = None

    def snap_to_mesh(self, mesh: Mesh) -> None:
        """
        Generate the probes and feed and ensure they're located correctly
        in relation to the mesh.  You must call this in order to get
        correct simulation behavior.
        """
        self._set_probes(mesh=mesh)
        self._set_feed(mesh=mesh)

    def frequency(self) -> np.array:
        """
        Get the frequency bins used for this port.
        """
        if not self._data_readp():
            raise RuntimeError("Must call calc() before retreiving values.")
        return self.freq

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

    def impedance(self) -> np.array:
        """
        Get the characteristic impedance.

        :returns: Array of the characteristic impedance magnitude for
            each frequency bin.
        """
        if not self._data_readp():
            raise RuntimeError("Must call calc() before retreiving values.")
        return self.z0

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
        self.v_inc = (v + (self.z0 * i)) / 2

    def _calc_v_ref(self, v, i) -> None:
        """
        Calculate the reflected voltage.

        See Pozar 4e section 4.3 (p.185) for derivation.

        :param v: Total voltage.
        :param i: Total current.
        """
        self.v_ref = (v - (self.z0 * i)) / 2

    def _calc_i_inc(self, v, i) -> None:
        """
        Calculate the incident current.

        :param v: Total voltage.
        :param i: Total current.
        """
        self.i_inc = (i + (v / self.z0)) / 2

    def _calc_i_ref(self, v, i) -> None:
        """
        Calculate the reflected current.

        :param v: Total voltage.
        :param i: Total current.
        """
        self.i_ref = ((v / self.z0) - i) / 2

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

    def _data_readp(self) -> bool:
        """
        Return True if data has been read from simulation result.
        """
        return self.freq is not None

    @classmethod
    def _get_ctr(cls):
        """
        Retrieve unique counter.
        """
        return cls.unique_ctr

    @classmethod
    def _inc_ctr(cls):
        """
        Increment unique counter.
        """
        cls.unique_ctr += 1

    @classmethod
    def _get_inc_ctr(cls):
        """
        Retrieve and increment unique counter.
        """
        ctr = cls._get_ctr()
        cls._inc_ctr()
        return ctr


class PlanarPort(Port):
    """
    Base class for planar ports (e.g. microstrip, coplanar waveguide,
    stripline, etc.).  Planar ports differ from one another in terms
    of the number, shape and position of their feeding and measurement
    probes.
    """

    def __init__(
        self,
        csx: ContinuousStructure,
        bounding_box: List[List[float]],
        thickness: float,
        conductivity: float = 5.8e7,
        excite: bool = False,
        feed_resistance: float = None,
        feed_shift: float = 0.2,
        ref_impedance: float = None,
        measurement_shift: float = 0.5,
        rotate: float = 0.0,
    ):
        """
        Planar port.

        The shape of the planar trace is rectangular in the xy plane.
        The first corner is determined by the x,y coordinates of
        `start_corner` and the opposite corner is determined by the
        x,y coordinates of `stop_corner`.  The z-position of the trace
        is determined by the z coordinate of `stop_corner`.  The z
        coordinate of `start_corner` gives the z position of the PCB
        ground plane beneath the top layer.  Specifically, it
        determines the height of the feed and measurement probes.

        By default, the trace extends in length from xmin to xmax.
        This behavior can be changed with the `rotate` parameter,
        which will rotate the structure at an angle about the z-axis.
        It is not currently possible to create a microstrip port that
        is not in the xy-plane.

        Excitation feeds are placed relative to `start_corner`'s x
        position.  See `feed_shift` for the relative positioning.

        :param bounding_box: A 2D list of 2 elements, where each
            element is an inner list of 3 elements.  The 1st list is
            the [x,y,z] components of the starting corner and the 2nd
            list is the opposite corner.  The actual trace height is 0
            and its shape is given by the x and y coordinates only.
            It lies in the xy-plane with the z-value given by the
            z-component in the 2nd inner list.  The z-value of the 1st
            list corresponds to the position of the ground plane.
            This is used for determining the position/length of feed
            and measurement probes.  Units are whatever you set the
            CSX unit to, defaults to m.
        :param thickness: Metal trace thickness.  Units are whatever
            you set the CSX unit to, defaults to m.
        :param conductivity: Metal conductivity (in S/m).  The default
            uses the conductivity of copper.
        :param feed_resistance: The feeding resistance value.  The
            default value of None creates an infinite resistance.  If
            you use the default value ensure that the port is
            terminated by a PML.  When performing a characteristic
            impedance measurement use the default value and PML, which
            gives better results than attempting to use a matching
            resistance.
        :param feed_shift: The amount by which to shift the feed as a
            fraction of the total port length.  The final position
            will be influenced by this value but adjusted for the mesh
            used.
        :param ref_impedance: The resistance used to calculate the
            port voltage and current values.  If left as the default
            value of None, the calculated characteristic impedance is
            used to calculate these values.
        :param measurement_shift: The amount by which to shift the
            measurement probes as a fraction of the total port length.
            By default, the measurement port is placed halfway between
            the start and stop.  Like `feed_shift`, the final position
            will be adjusted for the mesh used.  This is important
            since voltage probes need to lie on mesh lines and current
            probes need to be placed equidistant between them.
        :param rotate: The amount to rotate the port in degrees.  This
            uses `AddTransform('RotateAxis', 'z', rotate)`.
        """
        super().__init__(csx, excite)
        self.box = np.array(bounding_box)
        self.thickness = thickness
        self.conductivity = conductivity
        self.feed_resistance = feed_resistance
        self.feed_shift = feed_shift
        self.ref_impedance = ref_impedance
        self.measurement_shift = measurement_shift
        self.transform_args = ["RotateAxis", "z", rotate]
        self.rotate_transform = CSTransform()
        self.rotate_transform.AddTransform(*self.transform_args)

        self._set_trace()

    @abstractmethod
    def calc(self, sim_dir, freq) -> None:
        pass

    def get_feed_shift(self) -> float:
        """
        """
        return self.feed_shift

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
        if self.ref_impedance is not None:
            self.z0 = self.ref_impedance
        else:
            self.z0 = np.sqrt(v * dv / (i * di))

    def _set_trace(self) -> None:
        """
        Set trace.
        """
        trace = self.csx.AddConductingSheet(
            "ConductingSheet",
            conductivity=self.conductivity,
            thickness=self.thickness,
        )
        # trace = self.csx.AddMetal("trace")
        trace_box_coords = self._get_trace_box()
        trace_box = trace.AddBox(
            priority=max_priority(),
            start=trace_box_coords[0],
            stop=trace_box_coords[1],
        )
        trace_box.AddTransform(*self.transform_args)

    def _get_trace_box(self) -> List[List[float]]:
        """
        Get the pre-transformed trace box.
        """
        return [
            [self.box[0][0], self.box[0][1], self.box[1][2]],
            [self.box[1][0], self.box[1][1], self.box[1][2]],
        ]

    def _get_direction(self) -> int:
        """
        """
        return int(np.sign(self.box[1][0] - self.box[0][0]))

    def get_box(self) -> List[List[float]]:
        """
        """
        return self.box

    @abstractmethod
    def _set_probes(self, mesh: Mesh) -> None:
        pass

    @abstractmethod
    def _set_feed(self, mesh: Mesh) -> None:
        pass


class MicrostripPort(PlanarPort):
    """
    Microstrip transmission line port.
    """

    def calc(self, sim_dir, freq) -> None:
        """
        Calculate the characteristic impedance, propagation constant,
        and incident and reflected power.

        :param sim_dir: Simulation directory path.
        :param freq: Frequency bins.  Should be the same frequency
            bins as the ones used in the excitation.
        """
        self.freq = np.array(freq)
        [vprobe.read(sim_dir=sim_dir, freq=freq) for vprobe in self.vprobes]
        [iprobe.read(sim_dir=sim_dir, freq=freq) for iprobe in self.iprobes]
        v = self.vprobes[1].get_freq_data()[1]
        i = (
            0.5
            * -self._get_direction()
            * (
                self.iprobes[0].get_freq_data()[1]
                + self.iprobes[1].get_freq_data()[1]
            )
        )
        dv = (
            self.vprobes[2].get_freq_data()[1]
            - self.vprobes[0].get_freq_data()[1]
        ) / (
            self.unit
            * np.abs(self.vprobes[2].box[0][0] - self.vprobes[0].box[0][0])
        )
        di = (
            -self._get_direction()
            * (
                self.iprobes[1].get_freq_data()[1]
                - self.iprobes[0].get_freq_data()[1]
            )
            / (
                self.unit
                * np.abs(self.iprobes[1].box[0][0] - self.iprobes[0].box[0][0])
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

        feed = Feed(
            csx=self.csx,
            box=self._get_feed_box(mesh),
            excite_direction=[0, 0, 1],
            excite_type=excite_type,
            resistance=self.feed_resistance,
            transform_args=self.transform_args,
        )
        self.feeds.append(feed)

    def _set_probes(self, mesh: Mesh) -> None:
        """
        Set measurement probes.
        """
        trace_box = self._get_trace_box()
        trace_ylow = trace_box[0][1]
        trace_yhigh = trace_box[1][1]
        trace_ymid = (trace_ylow + trace_yhigh) / 2
        gnd_z = self.box[0][2]
        trace_z = trace_box[1][2]

        trace_xlow = trace_box[0][0]
        trace_xhigh = trace_box[1][0]
        x_index, vxmid = mesh.nearest_mesh_line(
            0,
            trace_box[0][0]
            + (self.measurement_shift * (trace_xhigh - trace_xlow)),
        )
        mesh.set_lines_equidistant(0, x_index - 1, x_index + 1)

        vxpos = [
            mesh.get_mesh_line(0, x_index - self._get_direction()),
            mesh.get_mesh_line(0, x_index),
            mesh.get_mesh_line(0, x_index + self._get_direction()),
        ]
        ixpos = [
            (vxpos[0] + vxpos[1]) / 2,
            (vxpos[1] + vxpos[2]) / 2,
        ]
        self.vprobes = [
            Probe(
                csx=self.csx,
                box=[[xpos, trace_ymid, gnd_z], [xpos, trace_ymid, trace_z]],
                p_type=0,
                transform_args=self.transform_args,
            )
            for xpos in vxpos
        ]
        self.iprobes = [
            Probe(
                csx=self.csx,
                box=[
                    [xpos, trace_ylow, trace_z],
                    [xpos, trace_yhigh, trace_z],
                ],
                p_type=1,
                norm_dir=0,
                transform_args=self.transform_args,
            )
            for xpos in ixpos
        ]

    def _get_excite_dir(self) -> List[int]:
        """
        """
        if self.box[0][2] < self.box[1][2]:
            return [0, 0, 1]
        else:
            return [0, 0, -1]

    def _get_feed_box(self, mesh: Mesh) -> List[List[float]]:
        """
        Get the pre-transformed excitation feed box.
        """
        _, xpos = mesh.nearest_mesh_line(
            0,
            self.box[0][0]
            + (self.feed_shift * (self.box[1][0] - self.box[0][0])),
        )
        return [
            [xpos, self.box[0][1], self.box[0][2]],
            [xpos, self.box[1][1], self.box[1][2]],
        ]


class CPWPort(PlanarPort):
    """
    Coplanar waveguide transmission line port.
    """

    def __init__(
        self,
        csx: ContinuousStructure,
        bounding_box: List[List[float]],
        gap: float,
        thickness: float,
        conductivity: float = 5.8e7,
        excite: bool = False,
        feed_resistance: float = None,
        feed_shift: float = 0.2,
        ref_impedance: float = None,
        measurement_shift: float = 0.5,
        rotate: float = 0.0,
    ):
        """
        :param gap: Gap between adjacent ground planes and trace (in m).
        """
        self.gap = gap
        super().__init__(
            csx,
            bounding_box,
            thickness,
            conductivity,
            excite,
            feed_resistance,
            feed_shift,
            ref_impedance,
            measurement_shift,
            rotate,
        )

    def calc(self, sim_dir, freq) -> None:
        """
        Calculate the characteristic impedance, propagation constant,
        and incident and reflected power.

        :param sim_dir: Simulation directory path.
        :param freq: Frequency bins.  Should be the same frequency
            bins as the ones used in the excitation.
        """
        self.freq = np.array(freq)
        [vprobe.read(sim_dir=sim_dir, freq=freq) for vprobe in self.vprobes]
        [iprobe.read(sim_dir=sim_dir, freq=freq) for iprobe in self.iprobes]
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

        i = (
            0.5
            * -self._get_direction()
            * (
                self.iprobes[0].get_freq_data()[1]
                + self.iprobes[1].get_freq_data()[1]
            )
        )
        dv = (v2 - v0) / (
            self.unit * (self.vprobes[2].box[0][0] - self.vprobes[0].box[0][0])
        )
        di = (
            -self._get_direction()
            * (
                self.iprobes[1].get_freq_data()[1]
                - self.iprobes[0].get_freq_data()[1]
            )
            / (
                self.unit
                * (self.iprobes[1].box[0][0] - self.iprobes[0].box[0][0])
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

        # when the user specifies a feed resistance, this means they
        # want the cpw port terminated with that resistance. To get
        # the best termination, use a single feed extending the trace
        # line. The user must provide a ground plane 1 gap width
        # behind the trace end.
        if self.feed_resistance:
            trace_box = self._get_trace_box()
            feed = Feed(
                csx=self.csx,
                box=[
                    [
                        trace_box[0][0] - (self._get_direction() * self.gap),
                        trace_box[0][1],
                        trace_box[0][2],
                    ],
                    [trace_box[0][0], trace_box[1][1], trace_box[1][2]],
                ],
                excite_direction=[self._get_direction(), 0, 0],
                excite_type=excite_type,
                resistance=self.feed_resistance,
                transform_args=self.transform_args,
            )
            self.feeds.append(feed)

        else:
            for box, excite_dir in zip(
                self._get_feed_boxes(mesh), [[0, 1, 0], [0, -1, 0]]
            ):
                feed = Feed(
                    csx=self.csx,
                    box=box,
                    excite_direction=excite_dir,
                    excite_type=excite_type,
                    resistance=self.feed_resistance,
                    transform_args=self.transform_args,
                )
                self.feeds.append(feed)

    def _get_feed_boxes(self, mesh: Mesh) -> None:
        """
        """
        _, xpos = mesh.nearest_mesh_line(
            0,
            self.box[0][0]
            + (self.feed_shift * (self.box[1][0] - self.box[0][0])),
        )
        feed_boxes = [
            [[xpos, ystart, 0], [xpos, yend, 0]]
            for ystart, yend in zip(
                [
                    self.box[0][1] - (self._get_direction() * self.gap),
                    self.box[1][1] + (self._get_direction() * self.gap),
                ],
                [self.box[0][1], self.box[1][1]],
            )
        ]
        return feed_boxes

    def _set_probes(self, mesh: Mesh) -> None:
        """
        Set measurement probes.
        """
        trace_box = self._get_trace_box()
        trace_ylow = trace_box[0][1]
        trace_yhigh = trace_box[1][1]
        trace_z = trace_box[1][2]

        trace_xlow = trace_box[0][0]
        trace_xhigh = trace_box[1][0]
        x_index, vxmid = mesh.nearest_mesh_line(
            0,
            trace_box[0][0]
            + (self.measurement_shift * (trace_xhigh - trace_xlow)),
        )
        mesh.set_lines_equidistant(0, x_index - 1, x_index + 1)

        vxpos = [
            mesh.get_mesh_line(0, x_index - self._get_direction()),
            mesh.get_mesh_line(0, x_index),
            mesh.get_mesh_line(0, x_index + self._get_direction()),
        ]
        ixpos = [
            (vxpos[0] + vxpos[1]) / 2,
            (vxpos[1] + vxpos[2]) / 2,
        ]

        self.vprobes = []
        for xpos in vxpos:
            self.vprobes.append(
                Probe(
                    csx=self.csx,
                    box=[
                        [xpos, trace_ylow - self.gap, trace_z],
                        [xpos, trace_ylow, trace_z],
                    ],
                    p_type=0,
                    transform_args=self.transform_args,
                )
            )
            self.vprobes.append(
                Probe(
                    csx=self.csx,
                    box=[
                        [xpos, trace_yhigh, trace_z],
                        [xpos, trace_yhigh + self.gap, trace_z],
                    ],
                    p_type=0,
                    transform_args=self.transform_args,
                )
            )

        self.iprobes = [
            Probe(
                csx=self.csx,
                box=[
                    [xpos, trace_ylow, trace_z],
                    [xpos, trace_yhigh, trace_z],
                ],
                p_type=1,
                norm_dir=0,
                transform_args=self.transform_args,
            )
            for xpos in ixpos
        ]


class RectWaveguidePort(Port):
    """
    Rectangular waveguide port base class.
    """

    def __init__(
        self,
        csx: ContinuousStructure,
        box: List[List[float]],
        propagation_axis: int,
        mode_name: str = "TE10",
        excite: bool = False,
        delay: float = 0,
        transform_args=None,
    ):
        """
        :param box: 2D list where first list is x,y,z of the first
            corner and second list is x,y,z of second corner.  The
            waveguide width and height are computed from this box.
        :param propagation_axis: Direction the waveguide is facing.
            Set to 0, 1, or 2, for x, y, or z.
        :param mode_name: Waveguide propagation mode.  If you don't
            know this you probably want the default.  Otherwise, see
            Pozar (4e) p.110 for more information.
        :param excite: If True, add an excitation to the port.
        """
        super().__init__(csx, excite)
        self.box = np.array(box, np.float)
        self.propagation_axis = propagation_axis
        self.mode_name = mode_name
        self.excite = excite
        self.delay = delay
        self.transform_args = transform_args

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

    def calc(self, sim_dir, freq):
        """
        Calculate the characteristic impedance, propagation constant,
        and incident and reflected voltage.

        TODO this doesn't account for reference plane shifts.  See
        p.184 of Pozar, or original openems ports.py file for this.

        :param sim_dir: Simulation directory path.
        :param freq: Frequency bins.  Should be the same frequency
            bins as the ones used in the excitation.
        """
        self.freq = np.array(freq)
        k = wavenumber(self.freq, 1)  # use m to be consistent with a,b
        self._calc_cutoff_wavenumber()
        self._calc_beta(k)
        self._calc_z0(k)
        [vprobe.read(sim_dir=sim_dir, freq=freq) for vprobe in self.vprobes]
        [iprobe.read(sim_dir=sim_dir, freq=freq) for iprobe in self.iprobes]
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
        shell_prop = self.csx.AddMetal("rect_wg_metal")
        back_face = self._shell_face_box(
            const_dim=self.propagation_axis,
            const_dim_idx=0,
            thickness=thickness,
        )
        shell_prop.AddBox(start=back_face[0], stop=back_face[1])
        dims = list(range(3))
        del dims[self.propagation_axis]
        for dim in dims:
            for i in range(2):
                face = self._shell_face_box(
                    const_dim=dim, const_dim_idx=i, thickness=thickness
                )
                shell_prop.AddBox(start=face[0], stop=face[1])

    def _shell_face_box(
        self, const_dim: int, const_dim_idx: int, thickness: float,
    ) -> List[List[float]]:
        """
        """
        box = [[None, None, None], [None, None, None]]
        for dim in range(3):
            if dim == const_dim:
                box[0][dim] = self.box[const_dim_idx][dim] + (
                    self._dim_outer_dir(dim, const_dim_idx) * thickness
                )
                box[1][dim] = self.box[const_dim_idx][dim]
            else:
                box[0][dim] = self.box[0][dim] - thickness
                box[1][dim] = self.box[1][dim] + thickness

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
            np.abs(self.box[1][dim] - self.box[0][dim]) for dim in range(3)
        ]
        del dimensions[self.propagation_axis]
        self.a = self.unit * np.amax(dimensions)
        self.b = self.unit * np.amin(dimensions)

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
                self.box[1][self.propagation_axis]
                - self.box[0][self.propagation_axis]
            )
        )

    def _set_func(self) -> None:
        """
        """
        ny_p = (self.propagation_axis + 1) % 3
        ny_pp = (self.propagation_axis + 2) % 3

        xyz = "xyz"
        if self.box[0][ny_p] != 0:
            name_p = "({}-{})".format(xyz[ny_p], self.box[0][ny_p])
        else:
            name_p = xyz[ny_p]
        if self.box[0][ny_pp] != 0:
            name_pp = "({}-{})".format(xyz[ny_p], self.box[0][ny_p])
        else:
            name_pp = xyz[ny_p]

        # TODO ???
        unit = get_unit(self.csx)
        a = self.a / unit
        b = self.b / unit
        if self.n > 0:
            self.e_func[ny_p] = "{}*cos({}*{})*sin({}*{})".format(
                self.n / b,
                self.m * np.pi / a,
                name_p,
                self.n * np.pi / b,
                name_pp,
            )
        if self.m > 0:
            self.e_func[ny_pp] = "{}*sin({}*{})*cos({}*{})".format(
                -self.m / a,
                self.m * np.pi / a,
                name_p,
                self.n * np.pi / b,
                name_pp,
            )

        if self.m > 0:
            self.h_func[ny_p] = "{}*sin({}*{})*cos({}*{})".format(
                self.m / a,
                self.m * np.pi / a,
                name_p,
                self.n * np.pi / b,
                name_pp,
            )
        if self.n > 0:
            self.h_func[ny_pp] = "{}*cos({}*{})*sin({}*{})".format(
                self.n / b,
                self.m * np.pi / a,
                name_p,
                self.n * np.pi / b,
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

    def _set_probes(self, mesh: Mesh) -> None:
        """
        Set measurement probes.
        """
        _, prop_pos = mesh.nearest_mesh_line(
            self.propagation_axis, self.box[1][self.propagation_axis]
        )

        probe_start = np.array(self.box[0])
        probe_stop = np.array(self.box[1])
        probe_start[self.propagation_axis] = prop_pos
        probe_stop[self.propagation_axis] = prop_pos

        self.vprobes = [
            Probe(
                csx=self.csx,
                box=[probe_start, probe_stop],
                p_type=10,
                transform_args=self.transform_args,
                mode_function=self.e_func,
            )
        ]

        self.iprobes = [
            Probe(
                csx=self.csx,
                box=[probe_start, probe_stop],
                p_type=11,
                transform_args=self.transform_args,
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
                self.propagation_axis, self.box[0][self.propagation_axis]
            )

            feed_start = np.array(self.box[0])
            feed_stop = np.array(self.box[1])
            feed_stop[self.propagation_axis] = prop_pos
            feed_start[self.propagation_axis] = prop_pos

            feed_vec = np.ones(3)
            feed_vec[self.propagation_axis] = 0
            weight_func = [str(x) for x in self.e_func]
            feed = Feed(
                csx=self.csx,
                box=[feed_start, feed_stop],
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
        self.kc = np.sqrt(
            np.power((self.m * np.pi / self.a), 2)
            + np.power((self.n * np.pi / self.b), 2)
        )


# See https://www.everythingrf.com/tech-resources/waveguides-sizes
standard_waveguides = {
    "WR159": {"a": np.multiply(1e-3, 40.386), "b": np.multiply(1e-3, 20.193)}
}
