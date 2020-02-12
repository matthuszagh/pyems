from bisect import bisect_left
from typing import List, Tuple
import numpy as np


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
        :param layer_sep: separations (in mm) between adjacent copper
            layers.  A list where the first value is the separation
            between the top layer and second layer, etc.  This is
            equivalently the substrate thickness.
        :param layer_thickness: thickness of each conductive layer (in
            mm).  Again proceeds from top to bottom layer.
        :param metal_conductivity: Metal layer conductivity in S/m.
        """
        self.layers = layers
        self.sub_epsr = sorted(sub_epsr, key=lambda epsr: epsr[0])
        self.sub_rho = sub_rho
        self.layer_sep = layer_sep
        self.layer_thick = layer_thickness
        self.metal_kappa = metal_conductivity

    def epsr_at_freq(self, freq: float):
        """
        Approximate the dielectric at a given frequency given the
        provided epsr values.

        :param freq: frequency of interest (Hz)

        :param returns: dielectric constant
        """
        if freq <= self.sub_epsr[0][0]:
            return self.sub_epsr[0][1]
        elif freq >= self.sub_epsr[-1][0]:
            return self.sub_epsr[-1][1]

        # perform linear interpolation
        tup_low = bisect_left([x[0] for x in self.sub_epsr], freq)
        if self.sub_epsr[tup_low][0] == freq:
            return self.sub_epsr[tup_low][1]

        tup_high = tup_low
        tup_low -= 1
        xlow = self.sub_epsr[tup_low][0]
        xhigh = self.sub_epsr[tup_high][0]
        ylow = self.sub_epsr[tup_low][1]
        yhigh = self.sub_epsr[tup_high][1]
        slope = (yhigh - ylow) / (xhigh - xlow)
        return ylow + (slope * (freq - xlow))

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

    def layer_separation(self):
        """
        """
        return self.layer_sep

    def metal_conductivity(self):
        """
        """
        return self.metal_kappa


common_pcbs = {
    "oshpark4": PCB(
        layers=4,
        sub_epsr=[
            (100e6, 3.72),
            (1e9, 3.69),
            (2e9, 3.68),
            (5e9, 3.64),
            (10e9, 3.65),
        ],
        sub_rho=4.4e14,
        layer_sep=[0.1702, 1.1938, 0.1702],
        layer_thickness=[0.0356, 0.0178, 0.0178, 0.0356],
        metal_conductivity=5.8e7,
    )
}
