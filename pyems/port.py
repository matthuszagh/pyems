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
from openEMS.ports import UI_data
from CSXCAD.CSXCAD import ContinuousStructure
from CSXCAD.CSTransform import CSTransform
from pyems.automesh import Mesh


def max_priority() -> int:
    """
    Priority that won't be overriden.

    :returns: highest priority.
    """
    return 999


class Port(ABC):
    """
    """

    def __init__(
        self,
        csx: ContinuousStructure,
        port_number: int,
        start: List[float],
        stop: List[float],
        excite: bool = False,
    ):
        """
        """

    # def read(self, sim_path: str, ):


class Probe:
    """
    """

    unique_ctr = 0

    def __init__(
        self,
        csx: ContinuousStructure,
        box: List[List[float]],
        p_type: int = 0,
        norm_dir: int = None,
        transform_args: List[str] = None,
    ):
        """
        """
        self.csx = csx
        self.box = box
        self.p_type = p_type
        self.norm_dir = norm_dir
        self.name = self._probe_name_prefix() + "t_" + str(self._get_ctr())
        self._inc_ctr()
        self.freq = None
        self.time = None
        self.t_data = None
        self.f_data = None

        if self.norm_dir is not None:
            self.csx_probe = self.csx.AddProbe(
                name=self.name, p_type=self.p_type, norm_dir=self.norm_dir
            )
        else:
            self.csx_probe = self.csx.AddProbe(
                name=self.name, p_type=self.p_type
            )

        self.csx_box = self.csx_probe.AddBox(
            start=self.box[0], stop=self.box[1]
        )
        if transform_args:
            self.csx_box.AddTransform(*transform_args)

    def shift_x(self, new_xpos: float) -> None:
        """
        Shift the probe to a new x position.

        :param new_xpos: New x-position.
        """
        self.csx_box.SetStart([new_xpos, self.box[0][1], self.box[0][2]])
        self.csx_box.SetStop([new_xpos, self.box[1][1], self.box[1][2]])

    def read(self, sim_dir, freq, signal_type="pulse"):
        """
        Read data recorded from the simulation and generate the time-
        and frequency-series data.
        """
        self.freq = freq
        data = UI_data([self.name], sim_dir, freq, signal_type)
        self.time = data.ui_time[0]
        self.t_data = data.ui_val[0]
        self.f_data = data.ui_f_val[0]

    def get_freq_data(self) -> np.array:
        """
        Get probe frequency data.

        :returns: 2D numpy array where 1st array contains the
                  frequency bins and 2nd array contains the
                  corresponding frequency bin values.
        """
        if not self._data_readp():
            raise ValueError("Must call read() before retreiving data.")
        else:
            return np.array([self.freq, self.f_data])

    def get_time_data(self):
        """
        Get probe time data.

        :returns: 2D numpy array where 1st array contains the
                  time values and 2nd array contains the
                  corresponding time data values.
        """
        if not self._data_readp():
            raise ValueError("Must call read() before retreiving data.")
        else:
            return np.array([self.time, self.t_data])

    def _data_readp(self) -> bool:
        if self.freq is not None:
            return True
        else:
            return False

    @classmethod
    def _inc_ctr(cls):
        cls.unique_ctr += 1

    @classmethod
    def _get_ctr(cls):
        return cls.unique_ctr

    def _probe_name_prefix(self):
        if self.p_type == 0:
            return "v"
        elif self.p_type == 1:
            return "i"
        elif self.p_type == 2:
            return "e"
        elif self.p_type == 3:
            return "h"
        elif self.p_type == 10:
            return "wv"
        elif self.p_type == 11:
            return "wi"
        else:
            raise ValueError("invalid p_type")


class PlanarPort(Port):
    """
    Base class for planar ports (e.g. microstrip, coplanar waveguide,
    stripline, etc.).  Planar ports differ from one another in terms
    of the number, shape and position of their feeding and measurement
    probes.
    """


class MicrostripPort:
    """
    Microstrip transmission line port.
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
        Microstrip port.

        The shape of the microstrip trace is rectangular in the xy
        plane.  The first corner is determined by the x,y coordinates
        of `start_corner` and the opposite corner is determined by the
        x,y coordinates of `stop_corner`.  The z-position of the trace
        is determined by the z coordinate of `stop_corner`.  The z
        coordinate of `start_corner` gives the z position of the PCB
        ground plane beneath the top layer.  Specifically, it
        determines the height of the feed and measurement probes.

        By default, the microstrip line extends in length from xmin to
        xmax.  This behavior can be changed with the `rotate`
        parameter, which will rotate the structure at an angle about
        the z-axis.  It is not currently possible to create a
        microstrip port that is not in the xy-plane.

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
            and measurement probes. All dimensions are in mm.
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
        [vprobe.read(sim_dir=sim_dir, freq=freq) for vprobe in self.vprobes]
        [iprobe.read(sim_dir=sim_dir, freq=freq) for iprobe in self.iprobes]
        v = self.vprobes[1].get_freq_data()[1]
        i = 0.5 * (
            self.iprobes[0].get_freq_data()[1]
            + self.iprobes[1].get_freq_data()[1]
        )
        dv = (
            self.vprobes[2].get_freq_data()[1]
            - self.vprobes[0].get_freq_data()[1]
        ) / (self.vprobes[2].box[0][0] - self.vprobes[0].box[0][0])
        di = (
            self.iprobes[1].get_freq_data()[1]
            - self.iprobes[0].get_freq_data()[1]
        ) / (self.iprobes[1].box[0][0] - self.iprobes[0].box[0][0])

        self.freq = freq
        self._calc_beta(v, i, dv, di)
        self._calc_z0(v, i, dv, di)
        k = 1 / np.sqrt(np.absolute(self.z0))
        self._calc_power_inc(k, v, i)
        self._calc_power_ref(k, v, i)

    def characteristic_impedance(self) -> List[List[float]]:
        """
        Get the characteristic impedance.

        :returns: A 2D list where the first element contains the
                  frequency bins and the second contains the
                  characteristic impedance values corresponding to
                  those frequency values.
        """
        if not self._data_readp():
            raise RuntimeError("Must call calc() before retreiving values.")
        return [self.freq, np.absolute(self.z0)]

    def incident_power(self) -> List[List[float]]:
        """
        Get the incident power.  This is generally useful for
        calculating S-parameters.

        :returns: A 2D list where the first element contains the
                  frequency bins and the second contains the incident
                  power values corresponding to those frequency
                  values.
        """
        if not self._data_readp():
            raise RuntimeError("Must call calc() before retreiving values.")
        return [self.freq, np.absolute(self.P_inc)]

    def reflected_power(self) -> List[List[float]]:
        """
        Get the reflected power.  This is generally useful for
        calculating S-parameters.

        :returns: A 2D list where the first element contains the
                  frequency bins and the second contains the reflected
                  power values corresponding to those frequency
                  values.
        """
        if not self._data_readp():
            raise RuntimeError("Must call calc() before retreiving values.")
        return [self.freq, np.absolute(self.P_ref)]

    def _data_readp(self) -> bool:
        """
        """
        return self.freq is not None

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
        Set microstrip trace.
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

    def _get_trace_box(self) -> List[List[float]]:
        """
        Get the pre-transformed trace box.
        """
        return [
            [self.box[0][0], self.box[0][1], self.box[1][2]],
            [self.box[1][0], self.box[1][1], self.box[1][2]],
        ]

    @classmethod
    def _get_inc_ctr(cls):
        """
        """
        ctr = cls._get_ctr()
        cls._inc_ctr()
        return ctr

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
