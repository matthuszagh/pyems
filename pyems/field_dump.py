from tempfile import mkdtemp
from enum import Enum
import os
import subprocess
from pyems.simulation import Simulation
from pyems.coordinate import Box3
from pyems.csxcad import construct_box
from pyems.priority import priorities


class DumpType(Enum):
    """
    Field (and other) dump types.
    """

    efield_time = (0, "Et")
    hfield_time = (1, "Ht")
    current_time = (2, "It")
    current_density_time = (3, "Jt")
    efield_frequency = (10, "Ef")
    hfield_frequency = (11, "Hf")
    current_frequency = (12, "If")
    current_density_frequency = (13, "Jf")
    local_sar_frequency = (20, "SAR_f")
    average_sar_frequency_1g = (21, "SAR_1g_f")
    average_sar_frequency_10g = (22, "SAR_10g_f")
    raw_data = (29, "raw")


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
            os.path.join(self._dir_path, self._dump_type.value[1]),
            dump_type=self._dump_type.value[0],
            file_type=0,
        )
        construct_box(
            prop=dump, box=self.box, priority=priorities["x"],
        )
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
        return self._dump_type.value[0]

    def view(self):
        """
        """
        subprocess.run(
            [
                "paraview",
                "--data={}".format(
                    os.path.join(
                        self._dir_path, self._dump_type.value[1] + "_..vtr"
                    )
                ),
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
