from typing import List
from CSXCAD.CSXCAD import ContinuousStructure
import numpy as np
from openEMS.ports import UI_data


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

        :returns: 2D numpy array where each inner array is a
                  frequency, value pair.  The result is sorted by
                  ascending frequency.
        """
        if not self._data_readp():
            raise ValueError("Must call read() before retreiving data.")
        else:
            return np.array([self.freq, self.f_data]).T

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
