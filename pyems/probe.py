from typing import List
import numpy as np
from openEMS.ports import UI_data
from pyems.simulation import Simulation
from pyems.mesh import Mesh
from pyems.coordinate import Box3, box_overlap, Axis


# TODO self.csx_box is messy. Should instead wrap CSPrimitives and
# member primitive should have a box.
class Probe:
    """
    """

    unique_ctr = 0

    def __init__(
        self,
        sim: Simulation,
        box: Box3,
        p_type: int = 0,
        normal_axis: Axis = None,
        weight: float = 1,
        mode_function: List = None,
    ):
        """
        """
        self._sim = sim
        self._box = box
        self.p_type = p_type
        self._normal_axis = normal_axis
        self.weight = weight
        self.mode_function = mode_function
        self.name = self._probe_name_prefix() + "t_" + str(self._get_ctr())
        self._inc_ctr()
        self.freq = None
        self.time = None
        self.t_data = None
        self.f_data = None
        self.csx_box = None

        self._set_probe()

    @property
    def sim(self) -> Simulation:
        """
        """
        return self._sim

    @property
    def box(self) -> Box3:
        """
        """
        return self._box

    def pml_overlap(self) -> bool:
        """
        """
        pml_boxes = self.sim.mesh.pml_boxes()
        for pml_box in pml_boxes:
            if box_overlap(self.box, pml_box):
                return True
        return False

    def _set_probe(self) -> None:
        """
        """
        self.csx_probe = self.sim.csx.AddProbe(
            name=self.name, p_type=self.p_type
        )
        self.csx_probe.SetWeighting(self.weight)

        if self._normal_axis is not None:
            self.csx_probe.SetNormalDir(self._normal_axis.axis)

        if self.mode_function is not None:
            self.csx_probe.SetModeFunction(self.mode_function)

        self.csx_box = self.csx_probe.AddBox(
            start=self.box.start(), stop=self.box.stop()
        )

    def snap_to_mesh(self, mesh) -> None:
        """
        Align probe with the provided mesh.  It is necessary to call
        this function in order to get correct simulation results.

        :param mesh: Mesh object.
        """
        for dim in [0, 1, 2]:
            self._snap_dim(mesh, dim)

    def _snap_dim(self, mesh: Mesh, dim: int) -> None:
        """
        Align probe to mesh for a given dimension.  This function will
        only have an effect when the provided dimension has zero size.

        :param mesh: Mesh object.
        :param dim: Dimension.  0, 1, 2 for x, y, z.
        """
        if self.box.min_corner[dim] == self.box.max_corner[dim]:
            start = self.csx_box.GetStart()
            stop = self.csx_box.GetStop()
            _, pos = mesh.nearest_mesh_line(dim, start[dim])
            start[dim] = pos
            stop[dim] = pos
            self.csx_box.SetStart(start)
            self.csx_box.SetStop(stop)

    def read(self, signal_type="pulse"):
        """
        Read data recorded from the simulation and generate the time-
        and frequency-series data.
        """
        freq = self.sim.freq
        sim_dir = self.sim.sim_dir
        self.freq = freq
        data = UI_data([self.name], sim_dir, freq, signal_type)
        self.time = data.ui_time[0]
        self.t_data = data.ui_val[0]
        self.f_data = data.ui_f_val[0]

    def get_freq_data(self) -> np.array:
        """
        Get probe frequency data.

        :returns: 2D numpy array where each inner array is a
                  frequency, value pair.  The result is sorted by
                  ascending frequency.
        """
        if not self._data_readp():
            raise ValueError("Must call read() before retreiving data.")
        else:
            return np.array([self.freq, self.f_data])

    def get_time_data(self):
        """
        Get probe time data.

        :returns: 2D numpy array where each inner array is a
                  time, value pair.  The result is sorted by
                  ascending time.
        """
        if not self._data_readp():
            raise ValueError("Must call read() before retreiving data.")
        else:
            return np.array([self.time, self.t_data]).T

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
