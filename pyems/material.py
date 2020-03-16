from typing import List, Tuple
import numpy as np
from pyems.utilities import sort_table_by_col, table_interp_val
from pyems.physical_constant import EPS0


def loss_to_kappa(loss: float, freq: float, epsr: float) -> float:
    """
    """
    return loss * EPS0 * epsr * 2 * np.pi * freq


class Dielectric:
    """
    """

    def __init__(
        self, epsr: List[Tuple[float, float]], loss: List[Tuple[float, float]]
    ):
        """
        :param epsr: Dielectric constant.  List of frequency (Hz) and
            associated dielectric constant.
        :param loss: Loss tangent.  List of frequency (Hz) and
            associated loss tangent.
        """
        self._epsr = sort_table_by_col(np.array(epsr), col=0)
        self._loss = sort_table_by_col(np.array(loss), col=0)
        self._kappa = self._kappa()

    def epsr_at_freq(self, freq: float) -> float:
        """
        Approximate the dielectric at a given frequency given the
        provided epsr values.

        :param freq: frequency of interest (Hz)

        :returns: dielectric constant
        """
        return float(table_interp_val(self._epsr, 1, freq, 0, True))

    def kappa_at_freq(self, freq: float) -> float:
        """
        Approximate the dielectric conductivity at a given frequency
        given the provided epsr and loss values.
        """
        return float(table_interp_val(self._kappa, 1, freq, 0, True))

    def _kappa(self) -> List[Tuple[float, float]]:
        """
        """
        kappas = []
        for freq, loss in self._loss:
            kappas.append(
                (freq, loss_to_kappa(loss, freq, self.epsr_at_freq(freq)))
            )
        return kappas


common_dielectrics = {
    "FR408": Dielectric(
        epsr=[
            (100e6, 3.72),
            (1e9, 3.69),
            (2e9, 3.68),
            (5e9, 3.64),
            (10e9, 3.65),
        ],
        loss=[
            (100e6, 0.0072),
            (1e9, 0.0091),
            (2e9, 0.0092),
            (5e9, 0.0098),
            (10e9, 0.0095),
        ],
    ),
    "PTFE": Dielectric(epsr=[(1, 2.06)], loss=[(1, 0.0002)]),
}
