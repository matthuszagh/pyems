import numpy as np
from pyems.coordinate import Coordinate3
from pyems.utilities import array_index
from pyems.calc import wavelength


class NF2FF:
    """
    """

    def __init__(self, sim):
        """
        """
        self._sim = sim
        self._box = self._construct_box()
        self.sim.register_nf2ff(self.box)

        # set later
        self._phi = None
        self._theta = None
        self._gain = None
        self._enorm = None

    @property
    def sim(self):
        """
        """
        return self._sim

    @property
    def box(self):
        """
        """
        return self._box

    def gain(self):
        """
        """
        if self._gain is None:
            raise RuntimeError("Must call calc before retrieving values.")
        return 10 * np.log10(self._gain)

    def radiation_pattern(self, theta: float = None, phi: float = None):
        """
        Calculate the radiation pattern for a series of angles.  If a
        value for theta or phi is provided, the radiation pattern will
        be restricted to that theta or phi value.  For instance,
        providing phi=0 will give the radiation pattern for all theta
        at phi=0.
        """
        if self._enorm is None:
            raise RuntimeError("Must call calc before retrieving values.")
        if theta is None and phi is None:
            return (
                20 * np.log10(self._enorm[:, :] / np.amax(self._enorm[:, :]))
                + self.gain()
            )
        elif theta is not None and phi is None:
            theta_idx = array_index(theta, self._theta)
            return (
                20
                * np.log10(
                    self._enorm[theta_idx, :]
                    / np.amax(self._enorm[theta_idx, :])
                )
                + self.gain()
            )
        elif theta is None and phi is not None:
            phi_idx = array_index(phi, self._phi)
            return (
                20
                * np.log10(
                    self._enorm[:, phi_idx] / np.amax(self._enorm[:, phi_idx])
                )
                + self.gain()
            )
        else:
            theta_idx = array_index(theta, self._theta)
            phi_idx = array_index(phi, self._phi)
            return (
                20
                * np.log10(
                    self._enorm[theta_idx, phi_idx]
                    / np.amax(self._enorm[theta_idx, phi_idx])
                )
                + self.gain()
            )

    def directivity(self, effective_aperture: float) -> float:
        """
        """
        return (
            effective_aperture
            * 4
            * np.pi
            / np.power(wavelength(self.sim.reference_frequency, 1), 2)
        )

    def _construct_box(self):
        """
        """
        if self.sim.mesh is None:
            raise RuntimeError(
                "Mesh must be created before initializing a "
                "near-field to far-field transformation."
            )
        box = self.sim.mesh.sim_box(include_pml=False)
        return self.sim.fdtd.CreateNF2FFBox(start=box.start(), stop=box.stop())

    def calc(
        self,
        theta: np.array,
        phi: np.array,
        radius: float = 1,
        center: Coordinate3 = Coordinate3(0, 0, 0),
        verbose: int = 1,
    ):
        """
        """
        self._theta = theta
        self._phi = phi
        res = self._box.CalcNF2FF(
            sim_path=self.sim.sim_dir,
            freq=self.sim.reference_frequency,
            theta=theta,
            phi=phi,
            radius=radius,
            center=center.coordinate_list(),
            verbose=verbose,
        )
        self._calc_gain(res)
        self._calc_enorm(res)

    def _calc_gain(self, nf2ff_res) -> None:
        """
        """
        self._gain = nf2ff_res.Dmax[0]

    def _calc_enorm(self, nf2ff_res) -> None:
        """
        """
        self._enorm = nf2ff_res.E_norm[0]
