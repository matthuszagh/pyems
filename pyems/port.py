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


def max_priority() -> int:
    """
    Priority that won't be overriden.

    :returns: highest priority.
    """
    return 999


class Port(ABC):
    """
    """

    def incident_power(self) -> np.array:
        """
        Get the incident power.  This is generally useful for
        calculating S-parameters.

        :returns: A 2D numpy array where the first column contains
                  frequency values and the second contains the
                  corresponding port incident power values.  It is
                  sorted by ascending frequency.
        """
        if not self._data_readp():
            raise RuntimeError("Must call calc() before retreiving values.")
        return np.concatenate(
            ([self.get_freq()], [np.absolute(self._get_p_inc())])
        ).T

    def reflected_power(self) -> np.array:
        """
        Get the reflected power.  This is generally useful for
        calculating S-parameters.

        :returns: A 2D numpy array where the first column contains
                  frequency values and the second contains the
                  corresponding port reflected power values.  It is
                  sorted by ascending frequency.
        """
        if not self._data_readp():
            raise RuntimeError("Must call calc() before retreiving values.")
        return np.array(
            ([self.get_freq()], [np.absolute(self._get_p_ref())])
        ).T

    def _data_readp(self) -> bool:
        """
        """
        return self.get_freq() is not None

    @abstractmethod
    def get_freq(self) -> np.array:
        pass

    @abstractmethod
    def _get_p_inc(self) -> np.array:
        pass

    @abstractmethod
    def _get_p_ref(self) -> np.array:
        pass


class PlanarPort(Port):
    """
    Base class for planar ports (e.g. microstrip, coplanar waveguide,
    stripline, etc.).  Planar ports differ from one another in terms
    of the number, shape and position of their feeding and measurement
    probes.
    """

    unique_ctr = 0

    def __init__(
        self,
        csx: ContinuousStructure,
        bounding_box: List[List[float]],
        thickness: float,
        conductivity: float = 5.8e7,
        excite: bool = False,
        feed_resistance: float = None,
        feed_shift: float = 0.2,
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

        :param csx: The CSXCAD ContinuousStructure to which this port
            is added.
        :param bounding_box: A 2D list of 2 elements, where each
            element is an inner list of 3 elements.  The 1st list is
            the [x,y,z] components of the starting corner and the 2nd
            list is the opposite corner.  The actual trace height is 0
            and its shape is given by the x and y coordinates only.
            It lies in the xy-plane with the z-value given by the
            z-component in the 2nd inner list.  The z-value of the 1st
            list corresponds to the position of the ground plane.
            This is used for determining the position/length of feed
            and measurement probes.  All dimensions are in mm.
        :param thickness: Metal trace thickness (in mm).
        :param conductivity: Metal conductivity (in S/m).  The default
            uses the conductivity of copper.
        :param excite: Set to True if this port should generate an
            excitation.  The actual excitation type is set by the
            `Simulation` object that contains this port.
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
        self.unit = 1
        self.csx = csx
        self.box = np.multiply(self.unit, bounding_box)
        self.thickness = self.unit * thickness
        self.conductivity = conductivity
        self.excite = excite
        self.feed_resistance = feed_resistance
        self.feed_shift = feed_shift
        self.measurement_shift = measurement_shift
        self.transform_args = ["RotateAxis", "z", rotate]
        self.rotate_transform = CSTransform()
        self.rotate_transform.AddTransform(*self.transform_args)

        # set later
        self.vprobes = None
        self.iprobes = None
        self.freq = None
        self.z0 = None
        self.beta = None
        self.P_inc = None
        self.P_ref = None
        self.feed_res_box = None
        self.excitation_box = None

        self._set_trace()
        self._set_feed()
        self._set_measurement_probes()

    def snap_probes_to_mesh(self, mesh: Mesh) -> None:
        """
        Position the probes so that they're located correctly in
        relation to the mesh.  You must call this in order to get
        correct simulation behavior.
        """
        # TODO this probably doesn't work when rotations are used.
        mid_idx, mid_xpos = mesh.nearest_mesh_line(
            0, self.vprobes[1].box[0][0]
        )
        new_vxpos = [
            mesh.mesh_lines[0][mid_idx - 1],
            mid_xpos,
            mesh.mesh_lines[0][mid_idx + 1],
        ]
        new_ixpos = [
            (new_vxpos[0] + new_vxpos[1]) / 2,
            (new_vxpos[1] + new_vxpos[2]) / 2,
        ]
        [vprobe.shift_x(xpos) for vprobe, xpos in zip(self.vprobes, new_vxpos)]
        [iprobe.shift_x(xpos) for iprobe, xpos in zip(self.iprobes, new_ixpos)]
        self._snap_feed_to_mesh(mesh)

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
        v = self.vprobes[1].get_freq_data()[:, 1]
        i = 0.5 * (
            self.iprobes[0].get_freq_data()[:, 1]
            + self.iprobes[1].get_freq_data()[:, 1]
        )
        dv = (
            self.vprobes[2].get_freq_data()[:, 1]
            - self.vprobes[0].get_freq_data()[:, 1]
        ) / (self.vprobes[2].box[0][0] - self.vprobes[0].box[0][0])
        di = (
            self.iprobes[1].get_freq_data()[:, 1]
            - self.iprobes[0].get_freq_data()[:, 1]
        ) / (self.iprobes[1].box[0][0] - self.iprobes[0].box[0][0])

        self._calc_beta(v, i, dv, di)
        self._calc_z0(v, i, dv, di)
        k = 1 / np.sqrt(np.absolute(self.z0))
        self._calc_power_inc(k, v, i)
        self._calc_power_ref(k, v, i)

    def characteristic_impedance(self) -> np.array:
        """
        Get the characteristic impedance.

        :returns: A 2D numpy array where the first column contains
                  frequency values and the second contains the
                  corresponding port characteristic impedance values.
                  It is sorted by ascending frequency.
        """
        if not self._data_readp():
            raise RuntimeError("Must call calc() before retreiving values.")
        return np.concatenate(([self.get_freq()], [np.absolute(self.z0)])).T

    def get_freq(self) -> np.array:
        """
        """
        return self.freq

    def _get_p_inc(self) -> np.array:
        """
        """
        return self.P_inc

    def _get_p_ref(self) -> np.array:
        """
        """
        return self.P_ref

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

    def _calc_power_inc(self, k, v, i) -> None:
        """
        Calculate the port's incident power wave.

        ..  math:: a_i = (1/2) k_i(V_i + Z_iI_i)

        ..  math:: k_i = sqrt{|Re(Z_i)|}^{-1}

        :param k: see equation
        :param v: voltage
        :param i: current
        """
        self.P_inc = (1 / 2) * k * (v + (self.z0 * i))

    def _calc_power_ref(self, k, v, i) -> None:
        """
        Calculate the port's reflected power wave.

        ..  math:: b_i = (1/2) k_i(V_i - Z_iI_i)

        ..  math:: k_i = sqrt{|Re(Z_i)|}^{-1}

        :param k: see equation
        :param v: voltage
        :param i: current
        """
        self.P_inc = (1 / 2) * k * (v - (np.conjugate(self.z0) * i))

    def _set_trace(self) -> None:
        """
        Set trace.
        """
        trace = self.csx.AddConductingSheet(
            "ConductingSheet",
            conductivity=self.conductivity,
            thickness=self.thickness,
        )
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

    def _set_measurement_probes(self):
        """
        Add measurement probes.
        """
        trace_box = self._get_trace_box()
        trace_ylow = trace_box[0][1]
        trace_yhigh = trace_box[1][1]
        trace_ymid = (trace_ylow + trace_yhigh) / 2
        gnd_z = self.box[0][2]
        trace_z = trace_box[1][2]
        vxpos = [
            trace_box[0][0] + (shift * (trace_box[1][0] - trace_box[0][0]))
            for shift in [
                self.measurement_shift - 0.1,
                self.measurement_shift,
                self.measurement_shift + 0.1,
            ]
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

    @abstractmethod
    def _set_feed(self) -> None:
        pass

    @abstractmethod
    def _snap_feed_to_mesh(self) -> None:
        pass

    @classmethod
    def _get_ctr(cls):
        """
        """
        return cls.unique_ctr

    @classmethod
    def _inc_ctr(cls):
        """
        """
        cls.unique_ctr += 1

    @classmethod
    def _get_inc_ctr(cls):
        """
        """
        ctr = cls._get_ctr()
        cls._inc_ctr()
        return ctr


class MicrostripPort(PlanarPort):
    """
    Microstrip transmission line port.
    """

    def _set_feed(self) -> None:
        """
        Set excitation feed.
        """
        if self.excite:
            excitation = self.csx.AddExcitation(
                name="excite_" + str(self._get_inc_ctr()),
                exc_type=0,
                exc_val=self._get_excite_dir(),
            )
            feed_box = self._get_feed_box()
            self.excitation_box = excitation.AddBox(
                start=feed_box[0], stop=feed_box[1], priority=max_priority()
            )
            self.excitation_box.AddTransform(*self.transform_args)

            if self.feed_resistance:
                feed_res = self.csx.AddLumpedElement(
                    name="resist_" + str(self._get_ctr()),
                    ny=2,
                    caps=True,
                    R=self.feed_resistance,
                )
                self.feed_res_box = feed_res.AddBox(
                    start=feed_box[0], stop=feed_box[1]
                )
                self.feed_res_box.AddTransform(*self.transform_args)

    def _snap_feed_to_mesh(self, mesh) -> None:
        """
        """
        if self.excite:
            old_start = self.excitation_box.GetStart()
            old_stop = self.excitation_box.GetStop()
            _, xpos = mesh.nearest_mesh_line(0, old_start[0])
            self.excitation_box.SetStart([xpos, old_start[1], old_start[2]])
            self.excitation_box.SetStop([xpos, old_stop[1], old_stop[2]])
            if self.feed_resistance:
                old_start = self.feed_res_box.GetStart()
                old_stop = self.feed_res_box.GetStop()
                _, xpos = mesh.nearest_mesh_line(0, old_start[0])
                self.feed_res_box.SetStart([xpos, old_start[1], old_start[2]])
                self.feed_res_box.SetStop([xpos, old_stop[1], old_stop[2]])

    def _get_excite_dir(self) -> List[int]:
        """
        """
        if self.box[0][2] < self.box[1][2]:
            return [0, 0, 1]
        else:
            return [0, 0, -1]

    def _get_feed_box(self) -> List[List[float]]:
        """
        Get the pre-transformed excitation feed box.
        """
        xpos = self.box[0][0] + (
            self.feed_shift * (self.box[1][0] - self.box[0][0])
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
        measurement_shift: float = 0.5,
        rotate: float = 0.0,
    ):
        """
        :param gap: Gap between adjacent ground planes and trace (in mm).
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
            measurement_shift,
            rotate,
        )

    def _set_feed(self) -> None:
        """
        Set excitation feed.
        """
        if self.excite:
            excitations = [
                self.csx.AddExcitation(
                    name="excite_" + str(self._get_inc_ctr()),
                    exc_type=0,
                    exc_val=direction,
                )
                for direction in [[0, 1, 0], [0, -1, 0]]
            ]
            feed_xpos = self.box[0][0] + (
                self.feed_shift * (self.box[1][0] - self.box[0][0])
            )
            feed_boxes = [
                [[feed_xpos, ystart, 0], [feed_xpos, yend, 0]]
                for ystart, yend in zip(
                    [self.box[0][1] - self.gap, self.box[1][1] + self.gap],
                    [self.box[0][1], self.box[1][1]],
                )
            ]
            self.excitation_boxes = [
                excitation.AddBox(
                    start=box[0], stop=box[1], priority=max_priority()
                )
                for excitation, box in zip(excitations, feed_boxes)
            ]
            [
                exc_box.AddTransform(*self.transform_args)
                for exc_box in self.excitation_boxes
            ]

            if self.feed_resistance:
                feed_res = self.csx.AddLumpedElement(
                    name="resist_" + str(self._get_ctr()),
                    ny=2,
                    caps=True,
                    R=2 * self.feed_resistance,  # parallel equiv resistance
                )
                self.feed_res_boxes = [
                    feed_res.AddBox(start=box[0], stop=box[1])
                    for box in feed_boxes
                ]
                [
                    res_box.AddTransform(*self.transform_args)
                    for res_box in self.feed_res_boxes
                ]

    def _snap_feed_to_mesh(self, mesh) -> None:
        """
        """
        if self.excite:
            for excitation_box in self.excitation_boxes:
                old_start = excitation_box.GetStart()
                old_stop = excitation_box.GetStop()
                _, xpos = mesh.nearest_mesh_line(0, old_start[0])
                excitation_box.SetStart([xpos, old_start[1], old_start[2]])
                excitation_box.SetStop([xpos, old_stop[1], old_stop[2]])
            if self.feed_resistance:
                for feed_res_box in self.feed_res_boxes:
                    old_start = feed_res_box.GetStart()
                    old_stop = feed_res_box.GetStop()
                    _, xpos = mesh.nearest_mesh_line(0, old_start[0])
                    feed_res_box.SetStart([xpos, old_start[1], old_start[2]])
                    feed_res_box.SetStop([xpos, old_stop[1], old_stop[2]])
