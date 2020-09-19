"""
Collection of CSXCAD wrapper functions.

The primary purpose of this file is to avoid openems floating point
rounding errors by rounding all coordinates passed to csxcad to a
limited precision.  I've implemented the rounding capability within
the CSXCAD wrapper functions to ensure that rounding is only performed
immediately before handing off coordinates to CSXCAD.  Incremental
rounding would accumulate rounding errors and defeat the purpose of
this file completely.
"""

from typing import Callable, List, Union, Optional
import numpy as np
from CSXCAD.CSXCAD import ContinuousStructure
from CSXCAD.CSRectGrid import CSRectGrid
from CSXCAD.CSProperties import CSProperties
from CSXCAD.CSPrimitives import (
    CSPrimitives,
    CSPrimPolygon,
    CSPrimBox,
    CSPrimCylinder,
    CSPrimCylindricalShell,
)
from CSXCAD.CSTransform import CSTransform
from pyems.coordinate import (
    Box3,
    C3Tuple,
    Axis,
    c3_maybe_tuple,
    Coordinate3,
    Coordinate2,
)
from pyems.utilities import apply_transform

# numeric decimal precision to avoid floating point errors due to
# rounding.
PREC = 10


colors = {
    "enig": "#EBC884",
    "soldermask": "#027240",
    "aluminum": "#A8B1B6",
    "copper": "#CD7F32",
    "ptfe": "#969C92",
}


def csxcad_nearest(val_or_arr):
    """
    """
    return np.around(val_or_arr, PREC)


def add_material(
    csx: ContinuousStructure,
    name: str,
    epsilon: float = 1.0,
    mue: float = 1.0,
    kappa: float = 0.0,
    sigma: float = 0.0,
    color: Optional[str] = None,
    alpha: int = 255,
) -> CSProperties:
    """
    :param epsilon: relative electric permeability.
    :param mue: relative magnetic permeability.
    :param kappa: electric conductivity.
    :param sigma: magnetic conductivity.
    """
    prop = csx.AddMaterial(
        name, epsilon=epsilon, mue=mue, kappa=kappa, sigma=sigma
    )
    if not color is None:
        prop.SetColor(color, alpha)

    return prop


def add_metal(
    csx: ContinuousStructure,
    name: str,
    color: Optional[str] = colors["aluminum"],
) -> CSProperties:
    """
    """
    prop = csx.AddMetal(name)
    if not color is None:
        prop.SetColor(color)

    return prop


def add_conducting_sheet(
    csx: ContinuousStructure,
    name: str,
    conductivity: float,
    thickness: float,
    color: Optional[str] = colors["enig"],
) -> CSProperties:
    """
    """
    prop = csx.AddConductingSheet(
        name, conductivity=conductivity, thickness=thickness,
    )
    if not color is None:
        prop.SetColor(color)

    return prop


def construct_box(
    prop: CSProperties,
    box: Box3,
    priority: int,
    transform: CSTransform = None,
) -> CSPrimitives:
    """
    """
    if transform is None:
        box = _add_box(
            prop=prop, priority=priority, start=box.start(), stop=box.stop()
        )
        return box

    if box.has_zero_dim():
        fp_warning(construct_box)

    box = _add_box(
        prop=prop,
        priority=priority,
        start=box.origin_start(),
        stop=box.origin_stop(),
    )
    apply_transform(box, transform)
    translate = CSTransform()
    translate.AddTransform("Translate", box.center().coordinate_list())
    apply_transform(box, translate)
    return box


def _circle_points(
    center: C3Tuple, radius: float, normal: Axis, poly_faces: int
) -> List[Coordinate2]:
    """
    """
    pts1 = np.multiply(radius, np.cos(np.linspace(0, 2 * np.pi, poly_faces)))
    pts2 = np.multiply(radius, np.sin(np.linspace(0, 2 * np.pi, poly_faces)))
    if normal.intval() == 0:
        pts1 += center.y
        pts2 += center.z
    elif normal.intval() == 1:
        pts1 += center.x
        pts2 += center.z
    else:
        pts1 += center.x
        pts2 += center.y

    lst = []
    for pt1, pt2 in zip(pts1, pts2):
        lst.append(Coordinate2(pt1, pt2))

    return lst


def construct_circle(
    prop: CSProperties,
    center: C3Tuple,
    radius: float,
    normal: Axis,
    priority: int,
    poly_faces: float = 60,
    transform: CSTransform = None,
) -> CSPrimitives:
    """
    :param normal: Normal direction to the surface of the circle. 0, 1, or 2.
    :param poly_faces: A circle is actually drawn as a polygon.  This
        specifies the number of polygon faces.  Obviously, the greater
        the number of faces, the more accurate the circle.
    """
    center = c3_maybe_tuple(center)
    prim = construct_polygon(
        prop=prop,
        points=_circle_points(
            center=center, radius=radius, normal=normal, poly_faces=poly_faces,
        ),
        normal=normal,
        elevation=center[normal.intval()],
        priority=priority,
        transform=transform,
    )

    return prim


def _poly_points(points: List[Coordinate2]) -> List[List[float]]:
    """
    Convert a set of coordinates to the format expected by CSXCAD.

    CSXCAD expects a list of 2 lists of positions, where the first
    inner list describes the first coordinate positions and the second
    inner list describes the second coordinate positions.  Each polygon
    point is given by the these coordinates combined with the elevation.
    """
    list1 = []
    list2 = []
    for point in points:
        list1.append(point.x)
        list2.append(point.y)

    return [list1, list2]


def construct_polygon(
    prop: CSProperties,
    points: List[Coordinate2],
    normal: Axis,
    elevation: float,
    priority: int,
    transform: CSTransform = None,
):
    """
    """
    poly_points = _poly_points(points)
    if transform is None:
        prim = _add_polygon(
            prop=prop,
            priority=priority,
            points=poly_points,
            norm_dir=normal.intval(),
            elevation=elevation,
        )
        return prim

    fp_warning(construct_polygon)

    first_coord_center = np.average(poly_points[0])
    second_coord_center = np.average(poly_points[1])
    if normal.intval() == 0:
        center = Coordinate3(
            elevation, first_coord_center, second_coord_center
        )
    elif normal.intval() == 1:
        center = Coordinate3(
            first_coord_center, elevation, second_coord_center
        )
    else:
        center = Coordinate3(
            first_coord_center, second_coord_center, elevation
        )

    centered_pts = [
        np.subtract(pts, cent)
        for pts, cent in zip(
            poly_points, [first_coord_center, second_coord_center]
        )
    ]

    prim = _add_polygon(
        prop=prop,
        priority=priority,
        points=centered_pts,
        norm_dir=normal.intval(),
        elevation=0,
    )
    apply_transform(prim, transform)
    tr = CSTransform()
    tr.AddTransform("Translate", center.coordinate_list())
    apply_transform(prim, tr)

    return prim


def construct_cylinder(
    prop: CSProperties,
    start: C3Tuple,
    stop: C3Tuple,
    radius: float,
    priority: int,
    transform: CSTransform = None,
) -> CSPrimCylinder:
    """
    """
    start = c3_maybe_tuple(start)
    stop = c3_maybe_tuple(stop)

    start = start.coordinate_list()
    stop = stop.coordinate_list()
    position = np.average([start, stop], axis=0)
    start = np.subtract(start, position)
    stop = np.subtract(stop, position)
    cyl = prop.AddCylinder(
        start=start, stop=stop, radius=radius, priority=priority
    )
    apply_transform(cyl, transform)

    translate = CSTransform()
    translate.AddTransform("Translate", position)
    apply_transform(cyl, translate)

    return cyl


def construct_cylindrical_shell(
    prop: CSProperties,
    start: C3Tuple,
    stop: C3Tuple,
    inner_radius: float,
    outer_radius: float,
    priority: int,
    transform: CSTransform = None,
) -> CSPrimCylindricalShell:
    """
    """
    start = c3_maybe_tuple(start)
    stop = c3_maybe_tuple(stop)

    start = start.coordinate_list()
    stop = stop.coordinate_list()
    position = np.average([start, stop], axis=0)
    start = np.subtract(start, position)
    stop = np.subtract(stop, position)
    cyl = prop.AddCylindricalShell(
        start=start,
        stop=stop,
        radius=np.average([inner_radius, outer_radius]),
        shell_width=outer_radius - inner_radius,
        priority=priority,
    )
    apply_transform(cyl, transform)

    translate = CSTransform()
    translate.AddTransform("Translate", position)
    apply_transform(cyl, translate)

    return cyl


def _remove_prim_coord_dups(
    coords: List[Union[Coordinate2, Coordinate3]]
) -> List[Union[Coordinate2, Coordinate3]]:
    """
    """
    unique_coords = []
    for coord in coords:
        if not coord in unique_coords:
            unique_coords.append(coord)

    return unique_coords


def prim_coords(prim: CSPrimitives) -> List[Coordinate3]:
    """
    Retrieve all coordinates from a primitive.
    """
    if isinstance(prim, CSPrimBox):
        return _box_coords(prim)
    elif isinstance(prim, CSPrimPolygon):
        return _poly_coords(prim)

    raise ValueError(
        "prim_coords not yet implemented for {}.".format(prim.GetTypeName())
    )


def prim_coords2(prim: CSPrimitives) -> List[Coordinate2]:
    """
    Retrieve a 2D xy projection of all coordinates from a primitive.
    """
    coords3 = prim_coords(prim)
    coords2 = [Coordinate2(coord.x, coord.y) for coord in coords3]

    return _remove_prim_coord_dups(coords2)


def _box_coords(box: CSPrimBox) -> List[Coordinate3]:
    """
    """
    start = box.GetStart()
    stop = box.GetStop()
    box3 = Box3(tuple(start), tuple(stop))
    corners = box3.corners()
    new_corners = []
    for corner in corners:
        transform = box.GetTransform()
        if transform is not None:
            corner = corner.transform(transform)
            corner = corner.round_prec(PREC)
        new_corners.append(corner)

    return _remove_prim_coord_dups(new_corners)


def _poly_coords(poly: CSPrimPolygon) -> List[Coordinate3]:
    """
    """
    coords = poly.GetCoords()
    elev = poly.GetElevation()
    norm_dir = poly.GetNormDir()
    coord_list = []
    for coord1, coord2 in zip(coords[0], coords[1]):
        if norm_dir == 0:
            coord_list.append(Coordinate3(elev, coord1, coord2))
        elif norm_dir == 1:
            coord_list.append(Coordinate3(coord1, elev, coord2))
        else:
            coord_list.append(Coordinate3(coord1, coord2, elev))

    new_coords = []
    for coord in coord_list:
        transform = poly.GetTransform()
        if transform is not None:
            coord = coord.transform(transform)
            coord = coord.round_prec(PREC)
        new_coords.append(coord)

    return _remove_prim_coord_dups(new_coords)


def fp_warning(func: Callable):
    """
    Warn the user that they've requested a transformation for a planar
    structure which may cause misalignment between the mesh and
    structure.
    """
    # raise RuntimeWarning(
    #     (
    #         "Transformation requested in `{}`. If the desired "
    #         "transformation is a 3D rotation that will make the planar "
    #         "structure have a nonzero length in every dimension then this "
    #         "is acceptable. If, however, the transformation is a "
    #         "translation, scale, or rotation which is a multiple of 90 "
    #         "degrees, then this will cause issues with mesh alignment. If "
    #         "you do not understand this message then you have probably "
    #         "done something wrong."
    #     ).format(func.__name__)
    # )
    pass


def add_line(grid: CSRectGrid, dim: int, val: float):
    """
    Add a grid line, rounding to limited precision.
    """
    grid.AddLine(dim, csxcad_nearest(val))


def _add_linpoly(
    prop: CSProperties,
    priority: int,
    points: List[float],
    norm_dir: int,
    elevation: float,
    length: float,
) -> CSPrimitives:
    """
    """
    if np.isclose(length, 0):
        points = csxcad_nearest(points)
        elevation = csxcad_nearest(elevation)

    prim = prop.AddLinPoly(
        priority=priority,
        points=points,
        norm_dir=norm_dir,
        elevation=elevation,
        length=length,
    )
    return prim


def _add_polygon(
    prop: CSProperties,
    priority: int,
    points: List[float],
    norm_dir: int,
    elevation: float,
) -> CSPrimitives:
    """
    """
    points = csxcad_nearest(points)
    elevation = csxcad_nearest(elevation)

    prim = prop.AddPolygon(
        priority=priority,
        points=points,
        norm_dir=norm_dir,
        elevation=elevation,
    )
    return prim


def _add_box(
    prop: CSProperties, priority: int, start: List[float], stop: List[float],
) -> CSPrimitives:
    """
    """
    # Round unconditionally even though doing so may be unnecessary
    # (some dimensions will have nonzero length). This shouldn't have
    # a downside and checking would take longer.
    start = csxcad_nearest(start)
    stop = csxcad_nearest(stop)

    prim = prop.AddBox(priority=priority, start=start, stop=stop)
    return prim
