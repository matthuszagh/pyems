from typing import Tuple, List


def pml_num_cells(boundary: str) -> int:
    """
    Number of cells in a PML boundary.  For instance, PML_8 would
    return 8.  Any non-PML boundary will return 0.
    """
    split = boundary.split("_")
    if split[0] == "PML":
        return int(split[1])
    else:
        return 0


class BoundaryConditions:
    """
    """

    def __init__(
        self,
        boundary: Tuple[Tuple[str, str], Tuple[str, str], Tuple[str, str]],
    ):
        """
        """
        self.boundary = boundary

    def as_list(self) -> List[str]:
        """
        """
        return [
            self.boundary[0][0],
            self.boundary[0][1],
            self.boundary[1][0],
            self.boundary[1][1],
            self.boundary[2][0],
            self.boundary[2][1],
        ]

    def pml_bounds(
        self,
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """
        """
        return (
            (
                pml_num_cells(self.boundary[0][0]),
                pml_num_cells(self.boundary[0][1]),
            ),
            (
                pml_num_cells(self.boundary[1][0]),
                pml_num_cells(self.boundary[1][1]),
            ),
            (
                pml_num_cells(self.boundary[2][0]),
                pml_num_cells(self.boundary[2][1]),
            ),
        )
