from tempfile import mkdtemp
import os
import subprocess
from pyems.simulation_beta import Simulation
from pyems.coordinate import Box3


class FieldDump:
    """
    """

    unique_index = 0

    def __init__(
        self,
        sim: Simulation,
        box: Box3,
        field_type: int = 0,
        dir_path: str = "fields",
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
        self._field_type = field_type
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
            dump_type=self._field_type,
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
    def field_type(self) -> int:
        """
        """
        return self._field_type

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
