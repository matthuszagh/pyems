from typing import List
from pyems.structure import Structure
from pyems.coordinate import Coordinate2


def module_header(name: str) -> str:
    """
    """
    return "(module " + name + " (layer F.Cu) (clearance 0) (attr virtual)\n"


def polygon(poly: List[Coordinate2]) -> str:
    """
    """
    poly_str = "  (fp_poly (pts "
    for coord in poly:
        poly_str += (
            "(xy "
            + "{:.6f}".format(coord.x)
            + " "
            + "{:.6f}".format(-coord.y)
            + ") "
        )

    poly_str += ") (layer F.Cu) (width 0))\n"
    return poly_str


def write_footprint(structure: Structure, name: str, fpath: str):
    """
    Write a Structure as a Kicad footprint.

    :param structure: Structure from which to create a footprint.
    :param name: Footprint name.
    :param fpath: String denoted the path where the footprint file
        should be saved.
    """
    if structure.polygons is None:
        raise RuntimeError("Unable to write footprint for structure.")

    with open(fpath, "w") as fout:
        fout.write(module_header(name))
        for poly in structure.polygons:
            fout.write(polygon(poly))
        fout.write(")\n")
