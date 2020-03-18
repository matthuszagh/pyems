from tempfile import mkdtemp
from enum import Enum
import os
import subprocess
from pyems.simulation import Simulation
from pyems.coordinate import Box3


class DumpType(Enum):
    """
    Field (and other) dump types.
    """

    efield_time = 0
    hfield_time = 1
    current_time = 2
    current_density_time = 3
    efield_frequency = 10
    hfield_frequency = 11
    current_frequency = 12
    current_density_frequency = 13
    local_sar_frequency = 20
    average_sar_frequency_1g = 21
    average_sar_frequency_10g = 22
    raw_data = 29


class FieldDump:
    """
    """

    unique_index = 0

    def __init__(
        self,
        sim: Simulation,
        box: Box3,
        dump_type: DumpType = DumpType.efield_time,
        dir_path: str = "dump",
    ):
        """
        :param dir_path: Directory where field dump data is stored.
            The directory is interpreted as being relative to the
            simulation directory.  Therefore, the default will place
            the field dumps within a 'fields' subdirectory of the
            simulation directory.  If left as None, a system temporary
            directory will be used.
        """
        self._sim = sim
        self._box = box
        self._dump_type = dump_type
        self._index = self._get_inc_ctr()
        if dir_path is None:
            dir_path = mkdtemp()
        else:
            dir_path = os.path.abspath(
                os.path.join(
                    self._sim.sim_dir, dir_path + "_" + str(self._index)
                )
            )
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
        self._dir_path = dir_path

        dump = self._sim.csx.AddDump(
            os.path.join(self._dir_path, "Et_"),
            dump_type=self._dump_type.value,
            file_type=0,
        )
        dump.AddBox(start=self.box.start(), stop=self.box.stop())
        self._sim.add_field_dump(self)

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

    @property
    def dump_type(self) -> int:
        """
        """
        return self._dump_type

    def view(self):
        """
        """
        subprocess.run(
            [
                "paraview",
                "--data={}".format(os.path.join(self._dir_path, "Et__..vtr")),
            ]
        )

    @classmethod
    def _get_ctr(cls):
        """
        Retrieve unique counter.
        """
        return cls.unique_index

    @classmethod
    def _inc_ctr(cls):
        """
        Increment unique counter.
        """
        cls.unique_index += 1

    @classmethod
    def _get_inc_ctr(cls):
        """
        Retrieve and increment unique counter.
        """
        ctr = cls._get_ctr()
        cls._inc_ctr()
        return ctr
