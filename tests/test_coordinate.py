import numpy as np
from pyems.coordinate import Box3, Coordinate3


def test_corner():
    box = Box3(Coordinate3(0, 0, 0), Coordinate3(1, 1, 1))
    corners = box.corners()
    corners = np.array([corner.coordinate_list() for corner in corners])
    assert np.array_equal(
        corners,
        np.array(
            [
                Coordinate3(0, 0, 0).coordinate_list(),
                Coordinate3(0, 1, 0).coordinate_list(),
                Coordinate3(1, 0, 0).coordinate_list(),
                Coordinate3(1, 1, 0).coordinate_list(),
                Coordinate3(0, 0, 1).coordinate_list(),
                Coordinate3(0, 1, 1).coordinate_list(),
                Coordinate3(1, 0, 1).coordinate_list(),
                Coordinate3(1, 1, 1).coordinate_list(),
            ]
        ),
    )
