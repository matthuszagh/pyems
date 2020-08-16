from __future__ import annotations
from typing import List, Union, Tuple, Optional
from copy import deepcopy
from functools import partial
import numpy as np


def val_inside(val: float, bound1: float, bound2: float) -> bool:
    """
    """
    min_bound = np.min([bound1, bound2])
    max_bound = np.max([bound1, bound2])
    if val >= min_bound and val <= max_bound:
        return True
    return False


class Coordinate2:
    """
    """

    def __init__(self, x: float, y: float):
        """
        """
        self._x = x
        self._y = y

    @property
    def x(self) -> float:
        """
        """
        return self._x

    @property
    def y(self) -> float:
        """
        """
        return self._y

    @x.setter
    def x(self, value: float) -> None:
        """
        """
        self._x = value

    @y.setter
    def y(self, value: float) -> None:
        """
        """
        self._y = value

    def __getitem__(self, key):
        """
        """
        return self._int_to_coord(key)

    def __setitem__(self, key, val):
        """
        """
        if key == 0:
            self.x = val
        elif key == 1:
            self.y = val
        else:
            raise ValueError("Invalid index.")

    def __eq__(self, other: Coordinate2) -> bool:
        """
        """
        if self.x == other.x and self.y == other.y:
            return True
        return False

    def _int_to_coord(self, val) -> float:
        """
        """
        if val == 0:
            return self.x
        elif val == 1:
            return self.y
        else:
            raise ValueError("Invalid index.")

    def coordinate_list(self) -> List[float]:
        """
        Retrieve a list of coordinate values for use by openems, which
        requires lists of coordinates.
        """
        return [self._x, self._y]

    def transform(self, transform: CSTransform) -> Coordinate2:
        """
        Transform the coordinate.  This does not update the coordinate, it
        simply returns a new transformed coordinate.  If you want to
        replace the old coordinate you must assign it to the result of
        this function.
        """
        clist = self.coordinate_list()
        clist.append(0)
        tclist = transform.Transform(clist)
        return Coordinate2(tclist[0], tclist[1])

    def round_prec(self, prec: int) -> Coordinate2:
        """
        """
        clist = self.coordinate_list()
        clist = np.around(clist, prec)
        return Coordinate2(clist[0], clist[1])


def list_center2(coords: List[Coordinate2]) -> Coordinate2:
    """
    Compute the center of a list of 2D coordinates.
    """
    pts = [coord.coordinate_list() for coord in coords]
    center = np.average(pts, axis=0)
    return Coordinate2(center[0], center[1])


def line2_angle(coord: Coordinate2, center: Coordinate2) -> float:
    """
    Compute the angle between a coordinate and a center coordinate.
    The 0 angle occurs at the positive x-axis.  That is, when the
    x-value of ``coord`` is greater than that of ``center`` but they
    have the same y-value.
    """
    xdiff = coord.x - center.x
    ydiff = coord.y - center.y
    ang = np.arctan2(ydiff, xdiff)
    if ang < 0:
        ang += 2 * np.pi

    return ang


def reorder_counterclockwise2(coords: List[Coordinate2]) -> List[Coordinate2]:
    """
    """
    center = list_center2(coords)
    func = partial(line2_angle, center=center)
    ordered_coords = sorted(coords, key=func)
    return ordered_coords


class Coordinate3(Coordinate2):
    """
    """

    def __init__(self, x: float, y: float, z: float):
        """
        """
        self._z = z
        super().__init__(x, y)

    @property
    def z(self):
        """
        """
        return self._z

    @z.setter
    def z(self, value: float):
        """
        """
        self._z = value

    def coordinate_list(self) -> List[float]:
        """
        Retrieve a list of coordinate values for use by openems, which
        requires lists of coordinates.
        """
        return [self._x, self._y, self._z]

    def transform(self, transform: CSTransform) -> Coordinate3:
        """
        Transform the coordinate.  This does not update the coordinate, it
        simply returns a new transformed coordinate.  If you want to
        replace the old coordinate you must assign it to the result of
        this function.
        """
        clist = self.coordinate_list()
        tclist = transform.Transform(clist)
        return Coordinate3(tclist[0], tclist[1], tclist[2])

    def round_prec(self, prec: int) -> Coordinate3:
        """
        """
        clist = self.coordinate_list()
        clist = np.around(clist, prec)
        return Coordinate3(clist[0], clist[1], clist[2])

    def __setitem__(self, key, val):
        """
        """
        if key == 0:
            self.x = val
        elif key == 1:
            self.y = val
        elif key == 2:
            self.z = val
        else:
            raise ValueError("Invalid index.")

    def __eq__(self, other: Coordinate3):
        """
        """
        if self.x == other.x and self.y == other.y and self.z == other.z:
            return True
        return False

    def _int_to_coord(self, val):
        """
        """
        if val == 0:
            return self.x
        elif val == 1:
            return self.y
        elif val == 2:
            return self.z
        else:
            raise ValueError("Invalid index.")


class Axis:
    """
    An abstraction of a coordinate axis.  It may also optionally
    specity a direction for cases where it is significant whether the
    axis points in the positive or negative direction relative to the
    normal coordinate axis.  When a direction is not specified, the
    positive direction is taken.
    """

    def __init__(self, axis, direction: int = 1):
        """
        """
        if type(axis) is str:
            self._axis = self._str_to_int(axis)
        else:
            self._axis = axis

        self._direction = direction
        self._check_direction()

    @property
    def axis(self) -> int:
        """
        """
        return self._axis

    @property
    def direction(self) -> int:
        """
        """
        return self._direction

    def is_positive_direction(self) -> bool:
        """
        """
        return self._direction == 1

    def as_list(self) -> List[int]:
        """
        """
        lst = [0, 0, 0]
        lst[self._axis] = self._direction
        return lst

    def _check_direction(self) -> None:
        """
        """
        if self._direction != 1 and self._direction != -1:
            raise ValueError("Invalid direction. Valid values are +1 and -1.")

    def _str_to_int(self, val: str) -> int:
        """
        """
        lval = val.lower()
        if lval == "x":
            return 0
        elif lval == "y":
            return 1
        elif lval == "z":
            return 2
        else:
            raise ValueError(
                "Invalid string axis value. Valid values are "
                "'x', 'y', and 'z' (case insensitive)."
            )

    def _int_to_str(self, val: int) -> str:
        """
        """
        if val == 0:
            return "x"
        elif val == 1:
            return "y"
        elif val == 2:
            return "z"
        else:
            raise ValueError(
                "Invalid integer axis value. Valid values are 0, 1, and 2."
            )

    def intval(self) -> int:
        """
        """
        return self._axis

    def strval(self) -> str:
        """
        """
        return self._int_to_str(self._axis)


C2Tuple = Union[Coordinate2, Tuple[float, float]]
C2TupleOp = Optional[Union[Coordinate2, Tuple[float, float]]]
C3Tuple = Union[Coordinate3, Tuple[float, float, float]]
C3TupleOp = Optional[Union[Coordinate3, Tuple[float, float, float]]]


def c2_maybe_tuple(coord: C2Tuple) -> Coordinate2:
    """
    Convenience function for classes and functions accepting a
    Coordinate2 parameter.  This removes the burden from the user of
    always importing and writing Coordinate2, which is tedious when a
    tuple precisely specifies the intention.
    """
    if isinstance(coord, Tuple):
        if not len(coord) == 2:
            raise ValueError(
                "Tuples passed as Coordinate2 must have length 2."
            )
        coord = Coordinate2(coord[0], coord[1])

    return coord


def c3_maybe_tuple(coord: C3Tuple) -> Coordinate3:
    """
    Convenience function for classes and functions accepting a
    Coordinate3 parameter.  This removes the burden from the user of
    always importing and writing Coordinate3, which is tedious when a
    tuple precisely specifies the intention.
    """
    if isinstance(coord, Tuple):
        if not len(coord) == 3:
            raise ValueError(
                "Tuples passed as Coordinate3 must have length 3."
            )
        coord = Coordinate3(coord[0], coord[1], coord[2])

    return coord


class Box2:
    """
    """

    def __init__(self, min_corner: C2Tuple, max_corner: C2Tuple):
        """
        """
        self._min_corner = c2_maybe_tuple(min_corner)
        self._max_corner = c2_maybe_tuple(max_corner)

    @property
    def min_corner(self) -> Coordinate2:
        """
        """
        return deepcopy(self._min_corner)

    @property
    def max_corner(self) -> Coordinate2:
        """
        """
        return deepcopy(self._max_corner)

    def start(self) -> List[float]:
        """
        List object expected by OpenEMS interface.
        """
        return self.min_corner.coordinate_list()

    def stop(self) -> List[float]:
        """
        List object expected by OpenEMS interface.
        """
        return self.max_corner.coordinate_list()

    def origin_start(self) -> List[float]:
        """
        The hypothetical start coordinate for the box if the box were
        centered at the origin.
        """
        return self.start() - self.center().coordinate_list()

    def origin_stop(self) -> List[float]:
        """
        The hypothetical stop coordinate for the box if the box were
        centered at the origin.
        """
        return self.stop() - self.center().coordinate_list()

    def as_list(self) -> List[List[float]]:
        """
        """
        return [self.start(), self.stop()]

    def corners(self) -> List[Coordinate2]:
        """
        """
        # TODO should be a more concise way to do this.
        corner1 = deepcopy(self.min_corner)

        corner2 = deepcopy(self.min_corner)
        corner2.y = self.max_corner.y

        corner3 = deepcopy(self.max_corner)

        corner4 = deepcopy(self.min_corner)
        corner4.x = self.max_corner.x

        return [corner1, corner2, corner3, corner4]

    def center(self) -> Coordinate2:
        """
        """
        return Coordinate2(
            np.average([self.min_corner.x, self.max_corner.x]),
            np.average([self.min_corner.y, self.max_corner.y]),
        )

    def length(self) -> float:
        """
        """
        return np.abs(self.max_corner.x - self.min_corner.x)

    def width(self) -> float:
        """
        """
        return np.abs(self.max_corner.y - self.min_corner.y)

    def negative_direction(self) -> bool:
        """
        """
        return self.max_corner.x < self.min_corner.x

    def has_zero_dim(self) -> bool:
        """
        Return True if at least 1 dimension of the box has zero size.
        """
        if np.isclose(self.max_corner.x - self.min_corner.x, 0) or np.isclose(
            self.max_corner.y - self.min_corner.y, 0
        ):
            return True
        return False


class Box3:
    """
    """

    def __init__(self, min_corner: C3Tuple, max_corner: C3Tuple):
        """
        """
        self._min_corner = c3_maybe_tuple(min_corner)
        self._max_corner = c3_maybe_tuple(max_corner)

    def __getitem__(self, key):
        """
        """
        return self._int_to_corner(key)

    def _int_to_corner(self, val: int) -> Coordinate3:
        """
        """
        if val == 0:
            return self.min_corner
        elif val == 1:
            return self.max_corner
        else:
            raise ValueError("Invalid Box3 index.")

    @property
    def min_corner(self) -> Coordinate3:
        """
        """
        return self._min_corner

    @min_corner.setter
    def min_corner(
        self, val: Union[Coordinate3, Tuple[float, float, float]]
    ) -> None:
        """
        """
        if isinstance(val, Tuple):
            if not len(val) == 3:
                raise ValueError("Tuples passed to Box3 must have length 3.")
            val = Coordinate3(val[0], val[1], val[2])

        self._min_corner = val

    @property
    def max_corner(self) -> Coordinate3:
        """
        """
        return self._max_corner

    @max_corner.setter
    def max_corner(
        self, val: Union[Coordinate3, Tuple[float, float, float]]
    ) -> None:
        """
        """
        if isinstance(val, Tuple):
            if not len(val) == 3:
                raise ValueError("Tuples passed to Box3 must have length 3.")
            val = Coordinate3(val[0], val[1], val[2])

        self._max_corner = val

    def set_increasing(self) -> None:
        """
        Rearrange min and max coordinates so that all min_corner
        coordinates are less than or equal to all max_corner
        coordinates.
        """
        for dim in range(3):
            if self.min_corner[dim] > self.max_corner[dim]:
                old_max = self.max_corner[dim]
                self.max_corner[dim] = self.min_corner[dim]
                self.min_corner[dim] = old_max

    def corners(self) -> List[Coordinate3]:
        """
        """
        # TODO should be a more concise way to do this.
        corner1 = deepcopy(self.min_corner)

        corner2 = deepcopy(self.min_corner)
        corner2.y = self.max_corner.y

        corner3 = deepcopy(self.min_corner)
        corner3.x = self.max_corner.x

        corner4 = deepcopy(self.max_corner)
        corner4.z = self.min_corner.z

        corner5 = deepcopy(self.min_corner)
        corner5.z = self.max_corner.z

        corner6 = deepcopy(self.max_corner)
        corner6.x = self.min_corner.x

        corner7 = deepcopy(self.max_corner)
        corner7.y = self.min_corner.y

        corner8 = deepcopy(self.max_corner)

        return [
            corner1,
            corner2,
            corner3,
            corner4,
            corner5,
            corner6,
            corner7,
            corner8,
        ]

    def inside(self, point: Coordinate3) -> bool:
        """
        """
        for dim in range(3):
            if not val_inside(
                point[dim], self.min_corner[dim], self.max_corner[dim]
            ):
                return False
        return True

    def start(self) -> List[float]:
        """
        List object expected by OpenEMS interface.
        """
        return self.min_corner.coordinate_list()

    def stop(self) -> List[float]:
        """
        List object expected by OpenEMS interface.
        """
        return self.max_corner.coordinate_list()

    def origin_start(self) -> List[float]:
        """
        The hypothetical start coordinate for the box if the box were
        centered at the origin.
        """
        return self.start() - self.center().coordinate_list()

    def origin_stop(self) -> List[float]:
        """
        The hypothetical stop coordinate for the box if the box were
        centered at the origin.
        """
        return self.stop() - self.center().coordinate_list()

    def as_list(self) -> List[List[float]]:
        """
        """
        return [self.start(), self.stop()]

    def center(self) -> Coordinate3:
        """
        """
        return Coordinate3(
            np.average([self.min_corner.x, self.max_corner.x]),
            np.average([self.min_corner.y, self.max_corner.y]),
            np.average([self.min_corner.z, self.max_corner.z]),
        )

    def length(self) -> float:
        """
        """
        return np.abs(self.max_corner.x - self.min_corner.x)

    def width(self) -> float:
        """
        """
        return np.abs(self.max_corner.y - self.min_corner.y)

    def height(self) -> float:
        """
        """
        return np.abs(self.max_corner.z - self.min_corner.z)

    def negative_direction(self) -> bool:
        """
        """
        return self.max_corner.x < self.min_corner.x

    def has_zero_dim(self) -> bool:
        """
        Return True if at least 1 dimension of the box has zero size.
        """
        if (
            np.isclose(self.max_corner.x - self.min_corner.x, 0)
            or np.isclose(self.max_corner.y - self.min_corner.y, 0)
            or np.isclose(self.max_corner.z - self.min_corner.z, 0)
        ):
            return True
        return False


def box_overlap(box1: Box3, box2: Box3) -> bool:
    """
    """
    box1_corners = box1.corners()
    for corner in box1_corners:
        if box2.inside(corner):
            return True

    box2_corners = box2.corners()
    for corner in box2_corners:
        if box1.inside(corner):
            return True

    return False
