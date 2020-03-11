from typing import List
from copy import deepcopy
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
        """
        return [self._x, self._y]


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
        """
        return [self._x, self._y, self._z]

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
    """

    def __init__(self, val):
        """
        """
        if type(val) is str:
            self._val = self._str_to_int(val)
        else:
            self._val = val

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
        return self._val

    def strval(self) -> str:
        """
        """
        return self._int_to_str(self._val)


class Box2:
    """
    """

    def __init__(self, min_corner: Coordinate2, max_corner: Coordinate2):
        """
        """
        self._min_corner = min_corner
        self._max_corner = max_corner

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

    def has_zero_dim(self) -> bool:
        """
        Return True if at least 1 dimension of the box has zero size.
        """
        if (
            self.max_corner.x - self.min_corner.x == 0
            or self.max_corner.y - self.min_corner.y == 0
        ):
            return True
        return False


class Box3:
    """
    """

    def __init__(self, min_corner: Coordinate3, max_corner: Coordinate3):
        """
        """
        self._min_corner = min_corner
        self._max_corner = max_corner

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
    def min_corner(self, val: Coordinate3) -> None:
        """
        """
        self._min_corner = val

    @property
    def max_corner(self) -> Coordinate3:
        """
        """
        return self._max_corner

    @max_corner.setter
    def max_corner(self, val: Coordinate3) -> None:
        """
        """
        self._max_corner = val

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

    def has_zero_dim(self) -> bool:
        """
        Return True if at least 1 dimension of the box has zero size.
        """
        if (
            self.max_corner.x - self.min_corner.x == 0
            or self.max_corner.y - self.min_corner.y == 0
            or self.max_corner.z - self.min_corner.z == 0
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
