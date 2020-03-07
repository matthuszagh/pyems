from typing import List, Tuple
import numpy as np
from pyems.utilities import (
    sort_table_by_col,
    table_interp_val,
)

C0 = 299792458
MUE0 = 4e-7 * np.pi
EPS0 = 1 / (MUE0 * (C0 ** 2))


def loss_to_kappa(loss: float, freq: float, epsr: float) -> float:
    """
    """
    return loss * EPS0 * epsr * 2 * np.pi * freq


# TODO this might need to be adapted in the future to support a
# different dielectric for each substrate layer.
class PCBProperties:
    """
    A PCB structure and material properties for use in an openems simulation.
    """

    def __init__(
        self,
        substrate_epsr: List[Tuple[float, float]],
        substrate_loss: List[Tuple[float, float]],
        copper_thickness: List[float],
        substrate_thickness: List[float],
        metal_conductivity: float,
        via_plating_thickness: float,
    ):
        """
        :param substrate_epsr: substrate dielectric constant.
            dictionary of frequency (Hz) and associated dielectric.
        :param substrate_loss: Substrate loss tangent.  Dictionary of
            frequency (Hz) and associated loss tangent.
        :param copper_thickness: thickness of each conductive layer
            (in m).  Again proceeds from top to bottom layer.
        :param substrate_thickness: separations (in m) between
            adjacent copper layers.  A list where the first value is
            the separation between the top layer and second layer,
            etc.  This is equivalently the substrate thickness.
        :param metal_conductivity: Metal layer conductivity in S/m.
        :param via_plating_thickness: Thickness of the via plating (in
            m).
        """
        self._substrate_epsr = sort_table_by_col(
            np.array(substrate_epsr), col=0
        )
        self._substrate_loss = sort_table_by_col(
            np.array(substrate_loss), col=0
        )
        self._substrate_kappa = self._kappa()
        self.copper_thick = copper_thickness
        self.substrate_thick = substrate_thickness
        self.metal_kappa = metal_conductivity
        self.via_plating_thick = via_plating_thickness

    def epsr_at_freq(self, freq: float) -> float:
        """
        Approximate the dielectric at a given frequency given the
        provided epsr values.

        :param freq: frequency of interest (Hz)

        :returns: dielectric constant
        """
        return float(table_interp_val(self._substrate_epsr, 1, freq, 0, True))

    def kappa_at_freq(self, freq: float) -> float:
        """
        Approximate the substrate conductivity at a given frequency
        given the provided epsr and loss values.
        """
        return float(table_interp_val(self._substrate_kappa, 1, freq, 0, True))

    def _kappa(self) -> List[Tuple[float, float]]:
        """
        """
        kappas = []
        for freq, loss in self._substrate_loss:
            kappas.append(
                (freq, loss_to_kappa(loss, freq, self.epsr_at_freq(freq)))
            )
        return kappas

    def copper_thickness(self, index: int, unit: float = 1) -> float:
        """
        Get the thickness of a copper layer.

        :param index: The index of the copper layer for which the
            thickness should be given.  The index ranges from 0 (for
            the top layer) to 1 less than the number of layers (bottom
            index).
        """
        if index < 0 or index >= len(self.copper_thick):
            raise ValueError("Invalid index provided.")

        return self.copper_thick[index] / unit

    def substrate_thickness(self, index: int, unit: float = 1) -> float:
        """
        Get the thickness of a substrate layer.

        :param index: The index of the substrate layer for which the
            thickness should be given.  The index ranges from 0 (for
            the substrate layer between the top and next copper
            layers) to 1 less than the number of substrate layers.
        """
        if index < 0 or index >= len(self.substrate_thick):
            raise ValueError("Invalid index provided.")

        return self.substrate_thick[index] / unit

    def via_plating_thickness(self, unit: float = 1) -> float:
        """
        """
        return self.via_plating_thick / unit

    def metal_conductivity(self) -> float:
        """
        """
        return self.metal_kappa

    def num_copper_layers(self) -> int:
        """
        Number of copper layers.

        :returns: The number of copper layers.
        """
        return len(self.copper_thick)

    def num_layers(self) -> int:
        """
        Total number of PCB layers.

        :returns: The total number of PCB layers, including all copper
                  and substrate layers.
        """
        return len(self.copper_thick) + len(self.substrate_thick)

    def copper_layer_dist(
        self,
        layer: int,
        unit: float = 1,
        ref_layer: int = 0,
        zero_thickness: bool = True,
    ) -> float:
        """
        Get the distance of a copper layer from some other layer.

        :param layer: The index of the copper layer for which the
            separation from the reference layer should be calculated.
            Indexed from 0 to 1 less than the number of copper layers.
        :param unit: Dimensional unit.  Leave the default for meters.
        :param ref_layer: The reference layer from which to compute
            the distance.  This defaults to the top layer.
        :param zero_thickness: Assume all copper layers contribute
            zero thickness.  This is the default since OpenEMS yields
            faster and more accurate simulations if zero-thickness
            ConductingSheet's are used for thin metals than if
            non-zero thickness PEC's are used.  If you're using
            non-zero thickness metals, set this to false.  When false,
            we measure to the top layer of the target layer from the
            top of the reference layer.
        """
        num_copper_layers = self.num_copper_layers()
        if (
            layer >= num_copper_layers
            or layer < 0
            or ref_layer >= num_copper_layers
            or ref_layer < 0
        ):
            raise ValueError("Invalid layer or ref_layer index.")

        if layer < ref_layer:
            raise ValueError(
                "layer should have at least as high a reference index as "
                "the reference layer."
            )

        thickness = np.sum(self.substrate_thick[ref_layer:layer])
        if not zero_thickness:
            thickness += np.sum(self.copper_thick[ref_layer:layer])

        return thickness / unit


# trace specs:
# min trace width = 5mil (0.127mm)
# min trace spacing = 5mil (0.127mm)
#
# drill specs:
# min annular ring = 4mil (0.1016mm)
# min drill size = 10mil (0.254mm)
common_pcbs = {
    "oshpark4": PCBProperties(
        substrate_epsr=[
            (100e6, 3.72),
            (1e9, 3.69),
            (2e9, 3.68),
            (5e9, 3.64),
            (10e9, 3.65),
        ],
        substrate_loss=[
            (100e6, 0.0072),
            (1e9, 0.0091),
            (2e9, 0.0092),
            (5e9, 0.0098),
            (10e9, 0.0095),
        ],
        copper_thickness=[0.0356e-3, 0.0178e-3, 0.0178e-3, 0.0356e-3],
        substrate_thickness=[0.1702e-3, 1.1938e-3, 0.1702e-3],
        metal_conductivity=5.8e7,
        via_plating_thickness=0.0254e-3,
    )
}
