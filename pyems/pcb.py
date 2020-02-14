from typing import List, Tuple
import numpy as np
from pyems.utilities import (
    sort_table_by_col,
    table_interp_val,
)


class PCB:
    """
    A PCB structure and material properties for use in an openems simulation.
    """

    def __init__(
        self,
        layers: int,
        sub_epsr: List[Tuple[float, float]],
        sub_rho: float,
        layer_sep: List[float],
        layer_thickness: List[float],
        metal_conductivity: float,
    ):
        """
        :param layers: number of conductive layers
        :param sub_epsr: substrate dielectric constant.  dictionary of
            frequency (Hz) and associated dielectric.
        :param sub_rho: volume resistivity (ohm*mm)
        :param layer_sep: separations (in m) between adjacent copper
            layers.  A list where the first value is the separation
            between the top layer and second layer, etc.  This is
            equivalently the substrate thickness.
        :param layer_thickness: thickness of each conductive layer (in
            m).  Again proceeds from top to bottom layer.
        :param metal_conductivity: Metal layer conductivity in S/m.
        """
        self.layers = layers
        self.sub_epsr = sort_table_by_col(np.array(sub_epsr), col=0)
        self.sub_rho = sub_rho
        self.layer_sep = layer_sep
        self.layer_thick = layer_thickness
        self.metal_kappa = metal_conductivity

    def epsr_at_freq(self, freq: float):
        """
        Approximate the dielectric at a given frequency given the
        provided epsr values.

        :param freq: frequency of interest (Hz)

        :returns: dielectric constant
        """
        return float(table_interp_val(self.sub_epsr, 1, freq, 0, True))

    def substrate_resistivity(self):
        """
        """
        return self.sub_rho

    def substrate_conductivity(self):
        """
        """
        return 1 / self.sub_rho

    def layer_thickness(self):
        """
        """
        return self.layer_thick

    def layer_separation(self, unit: float):
        """
        """
        return self.layer_sep / unit

    def metal_conductivity(self):
        """
        """
        return self.metal_kappa


common_pcbs = {
    "oshpark4": PCB(
        layers=4,
        sub_epsr=[
            [100e6, 3.72],
            [1e9, 3.69],
            [2e9, 3.68],
            [5e9, 3.64],
            [10e9, 3.65],
        ],
        sub_rho=4.4e14,
        layer_sep=np.multiply(1e-3, [0.1702, 1.1938, 0.1702]),
        layer_thickness=np.multiply(1e-3, [0.0356, 0.0178, 0.0178, 0.0356]),
        metal_conductivity=5.8e7,
    )
}
