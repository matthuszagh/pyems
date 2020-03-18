from typing import List, Tuple
import numpy as np
from pyems.simulation import Simulation
from pyems.mesh import Mesh
from pyems.utilities import max_priority
from pyems.coordinate import Box3, box_overlap


class Feed:
    """
    Excitation feed.
    """

    unique_ctr = 0

    def __init__(
        self,
        sim: Simulation,
        box: Box3,
        excite_direction: List[float],
        excite_type: int = None,
        impedance: complex = None,
        weight_func=None,
        delay: int = 0,
    ):
        """
        :param sim: Simulation to which this feed is added.
        :param box: Feed box.
        :param excite_direction: The direction that the excitation
            propagates.  Provide a list of 3 values corresponding to
            x, y, and z.  For instance, [0, 0, 1] would propagate in
            the +z direction.
        :param excite_type: Excitation type.  See `SetExcitation`.
            Leave as the default None, if you don't want an
            excitation.
        :param impedance: Feed impedance.  If left as None, which is
            the default, the feed will have infinite impedance.  In
            this case make sure to terminate the structure in PMLs.
        :param weight_func: Excitation weighting function.  See
            `SetWeightFunction`.
        :param delay: Excitation delay in seconds.
        """
        self._sim = sim
        self._box = box
        self.impedance = impedance
        self.excite_direction = excite_direction
        self.excite_type = excite_type
        self.weight_func = weight_func
        self.delay = delay
        self.excitation_box = None
        self.res_box = None

        self.set_feed()

    @property
    def sim(self) -> Simulation:
        """
        """
        return self._sim

    @property
    def box(self) -> Box3:
        """
        """
        return self._box

    def set_feed(self) -> None:
        """
        Set excitation feed.
        """
        if self.excite_type is not None:
            excitation = self.sim.csx.AddExcitation(
                name="excite_" + str(self._get_inc_ctr()),
                exc_type=self.excite_type,
                exc_val=self.excite_direction,
                delay=self.delay,
            )
            if self.weight_func:
                excitation.SetWeightFunction(self.weight_func)

            self.excitation_box = excitation.AddBox(
                start=self.box.start(),
                stop=self.box.stop(),
                priority=max_priority(),
            )

        if self.impedance:
            rval, cval, lval = self._impedance_rcl()
            res = self.sim.csx.AddLumpedElement(
                name="resist_" + str(self._get_ctr()),
                ny=self._resist_dir(),
                caps=True,
                R=rval,
                C=cval,
                L=lval,
            )
            self.res_box = res.AddBox(
                start=self.box.start(), stop=self.box.stop()
            )

    def pml_overlap(self) -> bool:
        """
        """
        if self.excitation_box is None and self.res_box is None:
            return False

        pml_boxes = self.sim.mesh.pml_boxes()
        for pml_box in pml_boxes:
            if box_overlap(self.box, pml_box):
                return True
        return False

    def _impedance_rcl(self) -> Tuple[float, float, float]:
        """
        """
        if np.is_complex(self.impedance):
            raise RuntimeWarning(
                "Only feed resistances are currently supported."
            )

        return (np.real(self.impedance), 0, 0)

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
        if self.box.min_corner[dim] == self.box.max_corner[dim]:
            start = self.excitation_box.GetStart()
            stop = self.excitation_box.GetStop()
            _, pos = mesh.nearest_mesh_line(dim, start[dim])
            start[dim] = pos
            stop[dim] = pos
            self.excitation_box.SetStart(start)
            self.excitation_box.SetStop(stop)
            if self.impedance:
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
