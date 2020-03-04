from typing import List
from CSXCAD.CSXCAD import ContinuousStructure
from pyems.automesh import Mesh
from pyems.utilities import max_priority


class Feed:
    """
    Excitation feed.
    """

    unique_ctr = 0

    def __init__(
        self,
        csx: ContinuousStructure,
        box: List[List[float]],
        excite_direction: List[float],
        excite_type: int = None,
        resistance: float = None,
        transform_args=None,
        weight_func=None,
        delay: int = 0,
    ):
        """
        :param csx: CSX object.
        :param box: Rectangular box giving feed dimensions.  [[x1, y1,
            z1], [x2, y2, z2]]
        :param excite_direction: The direction that the excitation
            propagates.  Provide a list of 3 values corresponding to
            x, y, and z.  For instance, [0, 0, 1] would propagate in
            the +z direction.
        :param excite_type: Excitation type.  See `SetExcitation`.
            Leave as the default None, if you don't want an
            excitation.
        :param resistance: Feed resistance.  If left as None, which is
            the default, the feed will have infinite impedance.  In
            this case make sure to terminate the structure in PMLs.
        :param transform_args: Any transformations to apply to feed.
        :param weight_func: Excitation weighting function.  See
            `SetWeightFunction`.
        :param delay: Excitation delay in seconds.
        """
        self.csx = csx
        self.box = box
        self.resistance = resistance
        self.excite_direction = excite_direction
        self.excite_type = excite_type
        self.transform_args = transform_args
        self.weight_func = weight_func
        self.delay = delay
        self.excitation_box = None
        self.res_box = None

        self.set_feed()

    def set_feed(self) -> None:
        """
        Set excitation feed.
        """
        if self.excite_type is not None:
            excitation = self.csx.AddExcitation(
                name="excite_" + str(self._get_inc_ctr()),
                exc_type=self.excite_type,
                exc_val=self.excite_direction,
                delay=self.delay,
            )
            if self.weight_func:
                excitation.SetWeightFunction(self.weight_func)

            self.excitation_box = excitation.AddBox(
                start=self.box[0], stop=self.box[1], priority=max_priority()
            )

            if self.transform_args is not None:
                self.excitation_box.AddTransform(*self.transform_args)

        if self.resistance:
            res = self.csx.AddLumpedElement(
                name="resist_" + str(self._get_ctr()),
                ny=self._resist_dir(),
                caps=True,
                R=self.resistance,
            )
            self.res_box = res.AddBox(start=self.box[0], stop=self.box[1])
            self.res_box.AddTransform(*self.transform_args)

    def snap_to_mesh(self, mesh) -> None:
        """
        Align feed with the provided mesh.  It is necessary to call
        this function in order to get correct simulation results.

        :param mesh: Mesh object.
        """
        for dim in [0, 1, 2]:
            self._snap_dim(mesh, dim)

    def _snap_dim(self, mesh: Mesh, dim: int) -> None:
        """
        Align feed to mesh for a given dimension.  This function will
        only have an effect when the provided dimension has zero size.

        :param mesh: Mesh object.
        :param dim: Dimension.  0, 1, 2 for x, y, z.
        """
        if self.box[0][dim] == self.box[1][dim]:
            start = self.excitation_box.GetStart()
            stop = self.excitation_box.GetStop()
            _, pos = mesh.nearest_mesh_line(dim, start[dim])
            start[dim] = pos
            stop[dim] = pos
            self.excitation_box.SetStart(start)
            self.excitation_box.SetStop(stop)
            if self.resistance:
                self.res_box.SetStart(start)
                self.res_box.SetStop(stop)

    def _resist_dir(self) -> int:
        """
        AddLumpedElement requires a direction in the form of 0, 1, or
        2. Get this value from the excitation direction.
        """
        # TODO doesn't work when excite_direction has multiple directions.
        return abs(self.excite_direction[1] + (2 * self.excite_direction[2]))

    @classmethod
    def _get_ctr(cls):
        """
        """
        return cls.unique_ctr

    @classmethod
    def _inc_ctr(cls):
        """
        """
        cls.unique_ctr += 1

    @classmethod
    def _get_inc_ctr(cls):
        """
        """
        ctr = cls._get_ctr()
        cls._inc_ctr()
        return ctr
