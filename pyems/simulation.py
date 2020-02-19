import tempfile
from typing import List
import numpy as np
from multiprocessing import Pool
from openEMS import openEMS
from CSXCAD.CSXCAD import ContinuousStructure
from pyems.network import Network
from pyems.field_dump import FieldDump
from pyems.utilities import wavelength, get_unit


class Simulation:
    """
    OpenEMS simulation.  This is the main entry-point for
    creating/running a simulation.
    """

    def __init__(
        self,
        fdtd: openEMS,
        csx: ContinuousStructure,
        center_freq: float,
        half_bandwidth: float,
        boundary_conditions: List[str],
        network: Network = None,
        field_dumps: List[FieldDump] = None,
    ):
        """
        """
        self.fdtd = fdtd
        self.csx = csx
        self.fdtd.SetCSX(self.csx)
        self.center_freq = center_freq
        self.half_bandwidth = half_bandwidth
        self.boundary_conditions = boundary_conditions
        self.fdtd.SetGaussExcite(center_freq, half_bandwidth)
        self.fdtd.SetBoundaryCond(boundary_conditions)
        self.network = network
        self.field_dumps = field_dumps

        # set later
        self.freq = None
        self.sim_dir = None
        self.nf2ff = None

    def set_network(self, network: Network) -> None:
        """
        """
        self.network = network

    def get_network(self) -> Network:
        return self.network

    def add_field_dump(self, field_dump: FieldDump) -> None:
        """
        """
        self.field_dumps.append(field_dump)

    def set_field_dumps(self, field_dumps: List[FieldDump]) -> None:
        """
        """
        self.field_dumps = field_dumps

    def get_field_dumps(self) -> List[FieldDump]:
        """
        """
        return self.field_dumps

    def finalize_structure(
        self, expand_bounds: List[float], simulation_bounds: List[float] = None
    ) -> None:
        """
        """
        self.network.generate_mesh(
            min_wavelength=wavelength(
                self.center_freq + self.half_bandwidth, get_unit(self.csx)
            ),
            expand_bounds=expand_bounds,
            simulation_bounds=simulation_bounds,
        )

    def simulate(self, num_freq_bins: int = 501, nf2ff: bool = False) -> None:
        """
        """
        if nf2ff:
            non_pml_box = self._get_sim_box_exc_pml()
            self.nf2ff = self.fdtd.CreateNF2FFBox(
                start=non_pml_box[0], stop=non_pml_box[1]
            )

        self.freq = np.linspace(
            self.center_freq - self.half_bandwidth,
            self.center_freq + self.half_bandwidth,
            num_freq_bins,
        )
        self.sim_dir = tempfile.mkdtemp()
        self.fdtd.Run(self.sim_dir, cleanup=True)
        self.network.calc(sim_dir=self.sim_dir, freq=self.freq)

    def _get_sim_box_exc_pml(self) -> np.array:
        """
        Return the simulation box volume excluding the PML boundaries.
        """
        exc_cells = np.zeros(6, dtype=int)
        for i, bound in enumerate(self.boundary_conditions):
            split = bound.split("_")
            if split[0] == "PML":
                exc_cells[i] = int(split[1]) + 1

        mesh = self.network.get_mesh().mesh_lines
        start = np.array(
            [
                mesh[0][exc_cells[0]],
                mesh[1][exc_cells[2]],
                mesh[2][exc_cells[4]],
            ]
        )
        stop = np.array(
            [
                mesh[0][-1 - exc_cells[1]],
                mesh[1][-1 - exc_cells[3]],
                mesh[2][-1 - exc_cells[5]],
            ]
        )
        return np.concatenate(([start], [stop]))

    def calc_nf2ff(
        self,
        theta: np.array = np.arange(0, 180, 1),
        phi: np.array = np.arange(0, 360, 1),
        radius: float = 1,
        center: List[float] = [0, 0, 0],
    ):
        """
        Perform a near-field to far-field transformation and return
        the results.
        """
        if self.nf2ff is None:
            raise RuntimeError(
                "You must set nf2ff to True in simulate() in order to perform "
                "this calculation."
            )
        print(
            "Running near-field to far-field transformation. "
            "This may take a while."
        )
        return self.nf2ff.CalcNF2FF(
            sim_path=self.sim_dir,
            freq=self.center_freq,
            theta=theta,
            phi=phi,
            radius=radius,
            center=center,
        )

    def get_freq(self) -> np.array:
        """
        """
        return self.freq

    def view_network(self) -> None:
        """
        View the simulation network.
        """
        self.network.view()

    def save_network(self, file_path: str) -> None:
        """
        Save the network to a file.

        :param file_path: The path where the network should be saved.
        """
        self.network.save(file_path=file_path)

    def view_field(self, index: int = 0) -> None:
        """
        View a field dump.

        :param index: The index of the field dump to view from the
            list of field dumps retreived by `get_field_dumps`.  This
            defaults to the first field dump if an index is not
            provided.
        """
        self.field_dumps[index].view()

    def save_field(self, dir_path: str, index: int = 0) -> None:
        """
        Save a field dump to a file.

        :param file_path: Directory path in which field dumps will be
            saved.
        :param index: The index of the field dump in `get_field_dumps`
            to save.  If this argument is ommitted, it defaults to the
            first field dump.
        """
        self.field_dumps[index].save(dir_path=dir_path)


def sweep(sims: List[Simulation], func, processes: int = 11):
    """
    Dispatch a number of simulations and then apply a function to each
    simulation after completion.  The function must take a single
    `Simulation` as an argument and return the result of interest.

    :param sims: A list of simulation objects to simulate and then
        process with `function_object`.  It is up to the caller to
        ensure that these simulation objects differ from one another
        in the desired way.
    :param func: The function object to apply to each simulation after
        that simulation has completed.  This must take a single
        Simulation object as an argument.  Use partials when the
        function takes more arguments.  Additionally, it must return
        the sweep data point.
    :param nodes: The number of simulations to run in parallel.
        OpenEMS already performs multithreading and so it's best to
        leave this significantly below the number of physical cores.

    :returns: A list where each entry is the return value of
              `function_object` applied to a single simulation from
              `simulations`.  The order between returned values and
              input simulations is preserved.  That is, the first item
              in the return value list corresponds to
              `simulations[0]`, etc.
    """
    # TODO requires pickling support for openems cython types
    # pool = Pool(processes=processes)
    # ret_vals = list(pool.map(func, sims))
    ret_vals = []
    for sim in sims:
        ret_vals.append(func(sim))
    return ret_vals
