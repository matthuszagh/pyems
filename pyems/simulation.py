import os
import subprocess
import sys
import shutil
from typing import List
from tempfile import mkdtemp
import numpy as np
from openEMS import openEMS
from CSXCAD.CSXCAD import ContinuousStructure
from pyems.boundary import BoundaryConditions


class Simulation:
    """
    OpenEMS simulation.  This is the main entry-point for
    creating/running a simulation.
    """

    def __init__(
        self,
        freq: np.array,
        unit: float = 1,
        boundary_conditions: BoundaryConditions = BoundaryConditions(
            (("PML_8", "PML_8"), ("PML_8", "PML_8"), ("PML_8", "PML_8")),
        ),
        end_criteria: float = 1e-5,
        sim_dir: str = "sim",
    ):
        """
        :param freq: An ordered (ascending) numpy array or list of
            frequency values to simulate.  All signal excitations will
            be automatically set based on this value, where the middle
            frequency value determines the center frequency of the
            gaussian excitation and the end frequency values are the
            -20dB values.  Additionally, the simulation timestep and
            post-processing frequency results are both dependent on
            this argument.  A larger number of values will increase
            the simulation time but will also increase the frequency
            resolution of output data.
        :param unit: Length dimension unit to use for all simulation
            distances.  Defaults to 1, which is meters.  For instance,
            1e-3 would be mm.
        :param boundary_conditions: The OpenEMS simulation boundary
            conditions.  Corresponds to ((xmin, xmax), (ymin, ymax),
            (zmin, zmax)).  See the OpenEMS documentation for details.
        :param end_criteria: FDTD termination energy.
        :param sim_dir: Directory where simulation results are stored.
            If you pass None, a temporary directory will be used.
            It's generally not recommended to use a temporary
            directory since it makes it more difficult to abort the
            simulation and to avoid rerunning unnecessary parts of the
            simulation.
        """
        self._freq = np.array(freq)
        self._unit = unit
        self._csx = ContinuousStructure()
        self._csx.GetGrid().SetDeltaUnit(self._unit)
        self._boundary_conditions = boundary_conditions
        self._end_criteria = end_criteria
        if sim_dir is None:
            self._sim_dir = mkdtemp()
        else:
            sim_dir = os.path.abspath(sim_dir)
            if os.path.exists(sim_dir):
                shutil.rmtree(sim_dir)
            os.mkdir(sim_dir)
            self._sim_dir = sim_dir
        self._fdtd = openEMS(EndCriteria=self._end_criteria)
        self._fdtd.SetGaussExcite(
            self.center_frequency(), self.half_bandwidth()
        )
        self._fdtd.SetBoundaryCond(self.boundary_conditions.as_list())
        self._fdtd.SetCSX(self._csx)
        self._ports = []
        self._field_dumps = []
        self._mesh = None
        self._csx_path = None
        self._nf2ff = None

    @property
    def freq(self) -> np.array:
        """
        """
        return self._freq

    @property
    def unit(self) -> float:
        """
        """
        return self._unit

    @property
    def csx(self) -> ContinuousStructure:
        """
        """
        return self._csx

    @property
    def fdtd(self) -> openEMS:
        """
        """
        return self._fdtd

    @property
    def boundary_conditions(self) -> BoundaryConditions:
        """
        """
        return self._boundary_conditions

    @property
    def ports(self):
        """
        """
        return self._ports

    @property
    def mesh(self):
        """
        """
        return self._mesh

    @property
    def sim_dir(self):
        """
        """
        return self._sim_dir

    @property
    def nf2ff(self):
        """
        """
        return self._nf2ff

    def run(self, csx: bool = True) -> None:
        """
        """
        if csx:
            self.view_csx(prompt=True)
        self.fdtd.Run(self.sim_dir, cleanup=False)
        self._calc_ports()

    def view_csx(self, prompt: bool = False) -> None:
        """
        View the CSX network.

        :param prompt: Prompt user whether to continue simulation.
        """
        subprocess.run(["AppCSXCAD", self._csx_path])
        if prompt:
            self._prompt_terminate()

    def view_field(self, index: int = 0) -> None:
        """
        View the field dump corresponding to the given index.
        """
        if index > len(self._field_dumps) - 1:
            raise ValueError("Invalid field dump index provided.")
        self._field_dumps[index].view()

    def _prompt_terminate(self) -> None:
        """
        """
        ans = input("Continue simulation (y/n)? ")
        ans = ans.lower()
        if ans == "n":
            print("Terminating simulation.")
            sys.exit(0)
        elif ans != "y":
            print("Please answer y/n.")
            self._prompt_terminate()

    def post_mesh(self):
        """
        Adjust the simulation for the generated mesh.
        """
        self._align_ports_to_mesh()
        self.save_csx()
        self._mesh_errors()

    def _mesh_errors(self) -> None:
        """
        """
        for port in self.ports:
            if port.pml_overlap():
                raise RuntimeError(
                    "Probe or feed overlaps PML. Please fix your simulation. "
                    "CSX file has been saved so you can view the overlap."
                )

    def save_csx(self, path: str = None):
        """
        """
        if path is None:
            path = self.sim_dir + "/" + self._file_name() + ".csx"
        self.csx.Write2XML(path)
        self._csx_path = path

    def _file_name(self):
        """
        """
        return os.path.splitext(os.path.basename(sys.argv[0]))[0]

    def _calc_ports(self):
        """
        """
        [port.calc() for port in self.ports]

    def s_param(self, i: int, j: int) -> np.array:
        """
        Calculate the S-parameter, S_{ij}.

        :param i: First subscript of S.  Must be in the range [1,
                  num_ports]
        :param j: Second subscript of S.  Must be in the range [1,
                  num_ports]
        """
        num_ports = self._num_ports()
        if i > num_ports or j > num_ports or i < 1 or j < 1:
            raise ValueError(
                "Invalid S-parameter requested. Ensure that i and j are in "
                "the proper range for the network."
            )

        i -= 1
        j -= 1
        s = (
            self.ports[i].reflected_voltage()
            / self.ports[j].incident_voltage()
        )
        s = 20 * np.log10(np.abs(s))
        return s

    def _num_ports(self) -> int:
        """
        """
        return len(self.ports)

    def center_frequency(self) -> float:
        """
        """
        idx = int(len(self.freq) / 2)
        return self.freq[idx]

    def half_bandwidth(self) -> float:
        """
        """
        return self.freq[-1] - self.center_frequency()

    def max_frequency(self) -> float:
        """
        """
        return self.center_frequency() + self.half_bandwidth()

    def add_port(self, port) -> None:
        """
        """
        self._ports.append(port)
        self._order_ports()

    def add_field_dump(self, field_dump) -> None:
        """
        """
        self._field_dumps.append(field_dump)

    def _order_ports(self) -> None:
        """
        """
        self._ports.sort(key=lambda port: port.number)

    def _align_ports_to_mesh(self) -> None:
        """
        """
        if self.ports is not None:
            [port.snap_to_mesh(mesh=self.mesh) for port in self.ports]

    def register_mesh(self, mesh) -> None:
        """
        """
        self._mesh = mesh

    def register_nf2ff(self, nf2ff) -> None:
        """
        """
        self._nf2ff = nf2ff


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
