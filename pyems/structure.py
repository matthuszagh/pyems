"""
A collection of macro structures.  These are frequently used
combinations of primitives, such as a via, microstrip line, etc.  This
allows you, for instance, to add a parameterized via, rather than a
cylindrical shell, air cylinder, circular pads, etc.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from CSXCAD.CSTransform import CSTransform
from CSXCAD.CSProperties import CSProperties
from CSXCAD.CSPrimitives import CSPrimitives
from pyems.pcb import PCBProperties
from pyems.utilities import apply_transform, append_transform
from pyems.coordinate import Coordinate2, Box2, Coordinate3, Axis, Box3
from pyems.simulation import Simulation
from pyems.port import MicrostripPort, CoaxPort, DifferentialMicrostripPort
import pyems.calc as calc
from pyems.priority import priorities
from pyems.material import Dielectric


def construct_circle(
    prop: CSProperties,
    center: Coordinate3,
    radius: float,
    normal: Axis,
    priority: int,
    poly_faces: float = 60,
) -> CSPrimitives:
    """
    :param normal: Normal direction to the surface of the circle. 0, 1, or 2,
    :param poly_faces: A circle is actually drawn as a polygon.  This
        specifies the number of polygon faces.  Obviously, the greater
        the number of faces, the more accurate the circle.
    """
    prim = prop.AddLinPoly(
        priority=priority,
        points=np.multiply(
            radius,
            [
                np.cos(np.linspace(0, 2 * np.pi, poly_faces)),
                np.sin(np.linspace(0, 2 * np.pi, poly_faces)),
            ],
        ),
        norm_dir=normal.intval(),
        elevation=0,
        length=0,
    )
    tr = CSTransform()
    tr.AddTransform("Translate", center.coordinate_list())
    apply_transform(prim, tr)

    return prim


def _transformed_coordinate(coord, transform_origin, transform: CSTransform):
    """
    Transform a coordinate about `transform_origin`, then translate it
    to the correct position.  The coordinate can either be Coordinate2
    or Coordinate3.
    """
    twod = False
    if type(coord) == Coordinate2:
        twod = True
        coord = Coordinate3(coord.x, coord.y, 0)
        transform_origin = Coordinate3(
            transform_origin.x, transform_origin.y, 0
        )

    tcoord = [
        coord.x - transform_origin.x,
        coord.y - transform_origin.y,
        coord.z - transform_origin.z,
    ]
    if transform is not None:
        tcoord = transform.Transform(tcoord)

    res_coord = Coordinate3(
        tcoord[0] + transform_origin.x,
        tcoord[1] + transform_origin.y,
        tcoord[2] + transform_origin.z,
    )
    if twod:
        return Coordinate2(res_coord.x, res_coord.y)
    else:
        return res_coord


def _set_box(
    prop: CSProperties,
    start: List[float],
    stop: List[float],
    position: Coordinate3,
    transform: CSTransform,
    priority: int,
) -> List[Coordinate2]:
    """
    Add a box by first constructing the box at the origin, then
    transforming it and finally translating it for the desired
    position.
    """
    box = prop.AddBox(priority=priority, start=start, stop=stop)
    if transform is not None:
        apply_transform(box, transform)
    translate = CSTransform()
    translate.AddTransform("Translate", position.coordinate_list())
    apply_transform(box, translate)

    tr_box = Box2(
        Coordinate2(start[0], start[1]), Coordinate2(stop[0], stop[1])
    )
    corners = tr_box.corners()
    tr_coordinates = []
    for corner in corners:
        corner_list = corner.coordinate_list()
        corner_list.append(0)
        if transform is not None:
            corner_list = transform.Transform(corner_list)
        corner_list = translate.Transform(corner_list)
        tr_coordinates.append(Coordinate2(corner_list[0], corner_list[1]))

    return tr_coordinates


def _polygon_points(points: List[Coordinate2]) -> List[List[float]]:
    """
    Convert a set of coordinates to the format expected by CSXCAD.

    CSXCAD expects a list of 2 lists of positions, where the first
    inner list describes the x-coordinate positions and the second
    inner list describes the y-coordinate positions.  Each polygon
    point is given by the x- and y-coordinate with matching list
    position.
    """
    list1 = []
    list2 = []
    for point in points:
        list1.append(point.x)
        list2.append(point.y)

    return [list1, list2]


def _set_polygon(
    prop: CSProperties,
    points: List[Coordinate2],
    elevation: float,
    position: Coordinate3,
    transform: CSTransform,
    priority: int,
) -> List[Coordinate2]:
    """
    :param points: A list 2D coordinates describing the xy points of
        the polygon.  The z-coordinate point is given by `elevation`.
        Select the points relative to the origin such that `transform`
        will be applied about the origin.  After `transform` is
        applied, the origin will be translated to `position`.

    :returns: Fully transformed polygon points.
    """
    poly = prop.AddPolygon(
        points=_polygon_points(points),
        norm_dir=2,
        elevation=elevation,
        priority=priority,
    )
    translate_vec = position.coordinate_list()
    translate = CSTransform()
    translate.AddTransform("Translate", translate_vec)
    apply_transform(poly, transform)
    apply_transform(poly, translate)

    transformed_coordinates = []
    for point in points:
        point_list = point.coordinate_list()
        point_list.append(0)
        if transform is not None:
            point_list = transform.Transform(point_list)
        point_list = translate.Transform(point_list)
        transformed_coordinates.append(
            Coordinate2(point_list[0], point_list[1])
        )

    return transformed_coordinates


def _set_cylinder(
    prop: CSProperties,
    start: Coordinate3,
    stop: Coordinate3,
    radius: float,
    transform: CSTransform,
    priority: int,
) -> None:
    """
    """
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


def _set_cylindrical_shell(
    prop: CSProperties,
    start: Coordinate3,
    stop: Coordinate3,
    inner_radius: float,
    outer_radius: float,
    transform: CSTransform,
    priority: int,
) -> None:
    """
    """
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


def _via_noconnect_layers(
    layers: List[int], noconnect_layers: List[int]
) -> List[int]:
    """
    """
    valid_layers = []
    for noconnect_layer in noconnect_layers:
        if noconnect_layer in layers:
            valid_layers.append(noconnect_layer)
        else:
            raise RuntimeWarning(
                "Via no-connect layer specified for layer where via "
                "isn't present. No-connect for layer {} will be "
                "ignored. Check your code.".format(noconnect_layer)
            )
    return valid_layers


class Structure(ABC):
    """
    Base class for all other structures.  Provides the capability to
    position and transform any structure.
    """

    unique_index = 0

    def __init__(self, sim: Simulation):
        """
        :param sim: The Simulation to which this object will be
            added.
        """
        self._sim = sim
        self._polygons = None

    @abstractmethod
    def construct(self, position) -> None:
        """
        Build the structure.  For each substructure this is a 3-stage
        process.  The substructure should be constructed as though the
        entire structure were being constructed at the origin.  Then,
        any transformations should be applied.  Finally, the structure
        should be translated to its final position.  This makes
        transformations easier to apply.

        Some structures do not support transforms, in which case the
        structure can be built directly at its final position.
        """
        pass

    @property
    def sim(self) -> Simulation:
        """
        """
        return self._sim

    @property
    def polygons(self) -> List:
        """
        """
        return self._polygons

    @classmethod
    def _get_ctr(cls):
        """
        Retrieve unique counter.
        """
        return cls.unique_index

    @classmethod
    def _inc_ctr(cls):
        """
        Increment unique counter.
        """
        cls.unique_index += 1

    @classmethod
    def _get_inc_ctr(cls):
        """
        Retrieve and increment unique counter.
        """
        ctr = cls._get_ctr()
        cls._inc_ctr()
        return ctr


class PCB(Structure):
    """
    Printed circuit board structure.

    All copper layers are filled automatically, although this can be
    overridden during initialization.  Use priorities to replace
    copper with other properties.
    """

    def __init__(
        self,
        sim: Simulation,
        pcb_prop: PCBProperties,
        length: float,
        width: float,
        position: Coordinate3 = Coordinate3(0, 0, 0),
        layers: range = None,
        omit_copper: List[int] = [],
    ):
        """
        :param pcb_prop: PCBProperties object that discribes this PCB.
        :param length: Length (using the dimensional unit set) of the
            circuit board in the x-direction.
        :param width: Width of the circuit board in the y-direction.
        :param position: The position of the top middle of the PCB.
        :param layers: A python range object specifying the
            layers to include.  For instance, you often only want to
            consider the top two conductive layers and the substrate
            between them.  In this case you'd pass range(3) for layers
            0, 1, and 2.  The default, None, includes all layers.
        :param omit_copper: A list of all copper layers not to fill
            with a ground plane.  By default, all copper layers are
            filled and can be overridden with higher priority
            primitives.  This ignores substrate layers unlike
            layers so to omit a ground plane on the 2nd layer
            you'd pass [1].
        """
        self._pcb_prop = pcb_prop
        self._length = length
        self._width = width
        self._position = position
        if layers is None:
            self._layers = range(self.pcb_prop.num_layers())
        else:
            self._layers = layers
        self._omit_copper = omit_copper
        super().__init__(sim)

        if self.position is not None:
            self.construct(self.position)

    @property
    def layers(self) -> range:
        """
        """
        return self._layers

    @property
    def pcb_prop(self) -> PCBProperties:
        """
        """
        return self._pcb_prop

    @property
    def position(self) -> Coordinate3:
        """
        """
        return self._position

    @property
    def width(self) -> float:
        """
        """
        return self._width

    @property
    def length(self) -> float:
        """
        """
        return self._length

    def copper_layer_elevation(self, layer: int) -> float:
        """
        """
        return self.position.z - self.pcb_prop.copper_layer_dist(
            layer, unit=self.sim.unit, ref_layer=self.layers[0]
        )

    def copper_layers(self) -> range:
        """
        Range object specifying all copper layers in the PCB.  This
        includes copper layers for which the copper pour has been
        omitted.
        """
        return range(int(self.layers[0] / 2), int(self.layers[-1] / 2) + 1)

    def copper_pours(self) -> List[int]:
        """
        List of all copper layers with a copper pour.
        """
        layers = list(self.copper_layers())
        for omission in self._omit_copper:
            layers.remove(omission)
        return layers

    def _is_copper_layer(self, layer_index: int) -> bool:
        """
        """
        return layer_index % 2 == 0

    def construct(self, position: Coordinate3) -> None:
        """
        """
        self._position = position
        zpos = 0
        for layer in self.layers:
            zpos = self._construct_layer(zpos, layer)

    def _construct_layer(self, zpos: float, layer_index: int) -> float:
        """
        """
        if self._is_copper_layer(layer_index):
            return self._construct_copper_layer(zpos, layer_index)
        else:
            return self._construct_substrate_layer(zpos, layer_index)

    def _construct_copper_layer(self, zpos: float, layer_index: int) -> float:
        """
        """
        copper_index = self._copper_index(layer_index)
        if copper_index in self._omit_copper:
            return zpos

        layer_prop = self.sim.csx.AddConductingSheet(
            self._layer_name(layer_index),
            conductivity=self.pcb_prop.metal_conductivity(),
            thickness=self.pcb_prop.copper_thickness(
                self._copper_index(layer_index)
            ),
        )

        xbounds = self._x_bounds()
        ybounds = self._y_bounds()
        layer_prop.AddBox(
            priority=priorities["ground"],
            start=[xbounds[0], ybounds[0], zpos],
            stop=[xbounds[1], ybounds[1], zpos],
        )

        return zpos

    def _construct_substrate_layer(
        self, zpos: float, layer_index: int
    ) -> None:
        """
        """
        ref_freq = self.sim.reference_frequency
        layer_prop = self.sim.csx.AddMaterial(
            self._layer_name(layer_index),
            epsilon=self.pcb_prop.substrate.epsr_at_freq(ref_freq),
            kappa=self.pcb_prop.substrate.kappa_at_freq(ref_freq),
        )
        xbounds = self._x_bounds()
        ybounds = self._y_bounds()
        zbounds = (
            zpos
            - self.pcb_prop.copper_layer_dist(
                self._copper_index(layer_index + 1),
                unit=self.sim.unit,
                ref_layer=self._copper_index(layer_index - 1),
            ),
            zpos,
        )
        layer_prop.AddBox(
            priority=priorities["substrate"],
            start=[xbounds[0], ybounds[0], zbounds[0]],
            stop=[xbounds[1], ybounds[1], zbounds[1]],
        )

        return zbounds[0]

    def _x_bounds(self) -> Tuple[float, float]:
        """
        """
        return (
            self.position.x - (self._length / 2),
            self.position.x + (self._length / 2),
        )

    def _y_bounds(self) -> Tuple[float, float]:
        """
        """
        return (
            self.position.y - (self._width / 2),
            self.position.y + (self._width / 2),
        )

    def _copper_index(self, layer_index: int) -> int:
        """
        The copper index for a given layer index.
        """
        if not self._is_copper_layer(layer_index):
            raise ValueError(
                "Tried to compute the copper layer index for a "
                "non-copper layer."
            )
        return int(layer_index / 2)

    def _substrate_index(self, layer_index: int) -> int:
        """
        The substrate index for a given layer index.
        """
        if self._is_copper_layer(layer_index):
            raise ValueError(
                "Tried to compute the substrate layer index for a "
                "copper layer."
            )
        return int(layer_index / 2)

    def _layer_name(self, layer_index: int) -> str:
        """
        A name to use when constructing a property.
        """
        if self._is_copper_layer(layer_index):
            return "copper_layer_" + str(self._copper_index(layer_index))
        else:
            return "substrate_layer_" + str(self._substrate_index(layer_index))


class Via(Structure):
    """
    Via structure.
    """

    unique_index = 0

    def __init__(
        self,
        pcb: PCB,
        position: Coordinate2,
        drill: float,
        annular_ring: float,
        antipad: float,
        layers: range = None,
        noconnect_layers: List[int] = [],
        fill: bool = False,
    ):
        """
        :param pcb: PCB object to which the via should be added.
        :param position: The (x,y) coordinates of the via.  These are
            absolute coordinates (e.g. not relative to some point on
            the PCB).  If you provide None, the via will not be
            constructed immediately and you will have to manually call
            construct later.  This is useful when you don't know the
            via position when you declare it.
        :param drill: Drill diameter.
        :param annular_ring: Width of the annular ring.
        :param antipad: Gap width between annular ring and surrounding
            copper pour for no-connect layers.
        :param layers: The PCB copper layers spanned by the via.  The
            default value of None causes the via to span all layers.
            If you pass a python range object the via will only span
            the copper layers included in that range.  For instance,
            range(1, 3) will create a buried via spanning the first
            and second copper layers.  Note that the layer indices are
            relative to the layers used in the pcb object, not all the
            layers of the PCBProperties object.  Therefore, if the PCB
            object omits the first two layers (first copper and
            substrate) of the PCBProperties object, for instance, the
            0th layer here will correspond to layer index 2 of the
            PCBProperties object.
        :param noconnect_layers: A list of copper layers for which the
            via will not be connected to the surrounding copper pour.
            This adds an antipad for these layers.  The index values
            ignore substrate layers, so 0 denotes the first copper
            layer, 1 denotes the 2nd, etc.  This should be set for all
            layers that connect to a signal trace, unless the copper
            pour has been removed from that layer.
        :param fill: The default of False fills the via with air and
            creates a dimensionally-accurate representation of the
            via.  If this is instead set to True, the metal plating
            will be extended to fill the entire via (i.e. the via will
            be represented as a solid metal cylinder).  OpenEMS
            struggles a bit with curved thin metals and so setting
            this to True may improve simulation results and
            efficiency.  I don't have enough data to definitively say
            which is better.  Until I have better information the
            default will be the dimensionally-accurate version.
        """
        self._pcb = pcb
        self._position = position
        self._drill = drill
        self._annular_ring = annular_ring
        self._antipad = antipad
        if layers is None:
            self._layers = self.pcb.copper_layers()
        else:
            self._layers = range(
                layers[0] + self.pcb.copper_layers()[0],
                layers[-1] + self.pcb.copper_layers()[0] + 1,
            )
        self._noconnect_layers = _via_noconnect_layers(
            self._layers, noconnect_layers
        )
        self._fill = fill
        self._index = None

        if self.position is not None:
            self.construct(self.position)

    @property
    def pcb(self) -> PCB:
        """
        """
        return self._pcb

    @property
    def layers(self) -> range:
        """
        """
        return self._layers

    @property
    def position(self) -> Coordinate2:
        """
        """
        return self._position

    def construct(self, position: Coordinate2) -> None:
        """
        """
        self._position = position
        self._index = self._get_inc_ctr()
        self._construct_via()
        self._construct_pads()
        self._construct_antipads()

    def _construct_via(self) -> None:
        """
        """
        start = [
            self.position.x,
            self.position.y,
            self.pcb.copper_layer_elevation(self.layers[0]),
        ]
        stop = [
            self.position.x,
            self.position.y,
            self.pcb.copper_layer_elevation(self.layers[-1]),
        ]
        via_prop = self.pcb.sim.csx.AddMetal(self._via_name())
        via_prim = via_prop.AddCylinder(
            priority=priorities["trace"],
            start=start,
            stop=stop,
            radius=self._shell_radius(),
        )
        apply_transform(via_prim, self.pcb.transform)

        if not self._fill:
            air_prop = self.pcb.sim.csx.AddMaterial(
                self._air_name(), epsilon=1
            )
            air_prim = air_prop.AddCylinder(
                priority=priorities["via_fill"],
                start=start,
                stop=stop,
                radius=self._drill_radius(),
            )
            apply_transform(air_prim, self.pcb.transform)

    def _construct_pads(self) -> None:
        """
        """
        for layer in self.layers:
            zpos = self.pcb.copper_layer_elevation(layer)
            pad_prop = self.pcb.sim.csx.AddConductingSheet(
                self._pad_name(layer),
                conductivity=self.pcb.pcb_prop.metal_conductivity(),
                thickness=self.pcb.pcb_prop.copper_thickness(layer),
            )
            pad_prim = construct_circle(
                prop=pad_prop,
                center=Coordinate3(self.position.x, self.position.y, zpos),
                radius=self.pad_radius(),
                normal=Axis("z"),
                priority=priorities["trace"],
            )
            apply_transform(pad_prim, self.pcb.transform)

    def _construct_antipads(self) -> None:
        """
        """
        for layer in self._noconnect_layers:
            zpos = self.pcb.copper_layer_elevation(layer)
            antipad_prop = self.pcb.sim.csx.AddMaterial(
                self._antipad_name(layer),
                epsilon=self.pcb.pcb_prop.substrate.epsr_at_freq(
                    self.pcb.sim.reference_frequency
                ),
                kappa=self.pcb.pcb_prop.substrate.kappa_at_freq(
                    self.pcb.sim.reference_frequency
                ),
            )
            antipad_prim = construct_circle(
                prop=antipad_prop,
                center=Coordinate3(self.position.x, self.position.y, zpos),
                radius=self._antipad_radius(),
                normal=Axis("z"),
                priority=priorities["keepout"],
            )
            apply_transform(antipad_prim, self.pcb.transform)

    def _air_name(self) -> str:
        """
        """
        return "air_" + str(self._index)

    def _via_name(self) -> str:
        """
        """
        return "via_" + str(self._index)

    def _pad_name(self, layer) -> str:
        """
        """
        return self._via_name() + "_pad_layer_" + str(layer)

    def _antipad_name(self, layer) -> str:
        """
        """
        return self._via_name() + "_antipad_layer_" + str(layer)

    def _drill_radius(self) -> float:
        """
        """
        return self._drill / 2

    def _shell_radius(self) -> float:
        """
        Radius of drill hole plus plating thickness.
        """
        return self._drill_radius() + self.pcb.pcb_prop.via_plating_thickness(
            unit=self.pcb.sim.unit
        )

    def pad_radius(self) -> float:
        """
        """
        return self._drill_radius() + self._annular_ring

    def _antipad_radius(self) -> float:
        """
        """
        return self.pad_radius() + self._antipad


class ViaFence(Structure):
    """
    Via fence structure.

    A via fence is a series of vias that can be used to
    electromagnetically guard regions of a PCB.
    """

    def __init__(
        self,
        pcb: PCB,
        position: Coordinate2,
        length: float,
        spacing: float,
        via: Via = None,
        transform: CSTransform = None,
    ):
        """
        A via fence, by default, will always be in the x-direction.
        For arbitrary directions, use the transform argument.
        """
        raise RuntimeError(
            "TODO Via fence is not yet implemented.  In particular, it needs "
            "a way of getting all via primitives to be able to transform "
            "them."
        )
        self._pcb = pcb
        self._position = position
        self._length = length
        self._spacing = spacing
        self._via = via
        self._transform = transform

        if self._position is not None:
            self.construct(self._position)

    def construct(
        self, position: Coordinate2, transform: CSTransform = None
    ) -> None:
        """
        """
        self._position = position
        self._transform = append_transform(self._transform, transform)

        vias = self._construct_zero()
        translate = CSTransform()
        translate_vec = [self._position.x, self._position.y, 0]
        translate.AddTransform("Translate", translate_vec)
        for via in vias:
            apply_transform(via, self._transform)
            apply_transform(via, translate)

    def _construct_zero(self) -> List[CSPrimitives]:
        """
        """
        posx = -self._length / 2 + self._spacing / 2
        stopx = self._length / 2 - self._spacing / 2

        while posx < stopx:
            self._via.construct(Coordinate2(posx, 0))
            posx += self._spacing


class ViaWall(Structure):
    """
    A via wall is similar to a via fence except that it approximates
    the series of vias as a single metal rectangular box.  This is
    more computationally efficient than a ViaFence and for most
    applications just as accurate.

    It's possible to make the width 0 and the mesh will try to put a
    mesh line directly at the 0-width position.  However, due to
    floating point errors this will sometimes ignore the wall, so it's
    generally recommended to use a nonzero width.
    """

    unique_index = 0

    def __init__(
        self,
        pcb: PCB,
        position: Coordinate2,
        length: float,
        width: float,
        antipad_width: float = None,
        layers: range = None,
        noconnect_layers: List[int] = [],
        transform: CSTransform = None,
    ):
        """
        """
        self._pcb = pcb
        self._position = position
        self._length = length
        self._width = width
        if layers is None:
            self._layers = self._pcb.copper_layers()
        else:
            self._layers = range(
                layers[0] + self._pcb.copper_layers()[0],
                layers[-1] + self._pcb.copper_layers()[0] + 1,
            )
        self._noconnect_layers = _via_noconnect_layers(
            self._layers, noconnect_layers
        )
        if antipad_width is None and len(self._noconnect_layers) != 0:
            raise ValueError(
                "Must specify antipad_width when using noconnect_layers."
            )
        self._antipad_width = antipad_width
        self._transform = transform
        self._index = None

        if self._position is not None:
            self.construct(self._position)

    def construct(
        self, position: Coordinate2, transform: CSTransform = None
    ) -> None:
        """
        """
        self._position = position
        self._transform = append_transform(self._transform, transform)
        self._index = self._get_inc_ctr()
        self._construct_via_wall()
        self._construct_antipads()

    def _construct_via_wall(self) -> None:
        """
        """
        prop = self._pcb.sim.csx.AddMetal(self._via_wall_name())
        _set_box(
            prop=prop,
            start=self._start(),
            stop=self._stop(),
            position=Coordinate3(self._position.x, self._position.y, 0),
            transform=self._transform,
            priority=priorities["ground"],
        )

    def _construct_antipads(self) -> None:
        """
        """
        if len(self._noconnect_layers) == 0:
            return

        ref_freq = self._pcb.sim.reference_frequency
        prop = self._pcb.sim.csx.AddMaterial(
            self._antipad_name(),
            epsilon=self._pcb.pcb_prop.substrate.epsr_at_freq(ref_freq),
            kappa=self._pcb.pcb_prop.substrate.kappa_at_freq(ref_freq),
        )
        for layer in self._noconnect_layers:
            zpos = self._pcb.copper_layer_elevation(layer)
            start = [
                -self._length / 2,
                -self._width / 2 - self._antipad_width,
                zpos,
            ]
            stop = [
                self._length / 2,
                self._width / 2 + self._antipad_width,
                zpos,
            ]
            _set_box(
                prop=prop,
                start=start,
                stop=stop,
                position=Coordinate3(self._position.x, self._position.y, 0),
                transform=self._transform,
                priority=priorities["keepout"],
            )

    def _start(self) -> List[float]:
        """
        """
        return [
            -self._length / 2,
            -self._width / 2,
            self._pcb.copper_layer_elevation(self._layers[0]),
        ]

    def _stop(self) -> List[float]:
        """
        """
        return [
            self._length / 2,
            self._width / 2,
            self._pcb.copper_layer_elevation(self._layers[-1]),
        ]

    def _via_wall_name(self) -> str:
        """
        """
        return "Via_Wall_" + str(self._index)

    def _antipad_name(self) -> str:
        """
        """
        return "Via_Wall_Antipad_" + str(self._index)


class Microstrip(Structure):
    """
    Microstrip transmission line structure.  This can also be set to
    act as a port for excitation and/or later analysis.  When used as
    a port, the microstrip cannot be transformed, since ports do not
    support transformations.
    """

    unique_index = 0

    def __init__(
        self,
        pcb: PCB,
        position: Coordinate2,
        length: float,
        width: float,
        propagation_axis: Axis,
        trace_layer: int = 0,
        gnd_layer: int = 1,
        gnd_gap: Tuple[float, float] = (None, None),
        terminal_gap: Tuple[float, float] = (None, None),
        port_number: int = None,
        excite: bool = False,
        invert_excitation: bool = False,
        feed_impedance: float = None,
        feed_shift: float = 0.2,
        ref_impedance: float = None,
        measurement_shift: float = 0.5,
        transform: CSTransform = None,
    ):
        """
        :param pcb: PCB object to which the microstrip line should be
            added.
        :param position: Center position of the microstrip trace.  The
            z-coordinate is determined by the PCB layer.
        :param length: Length of microstrip trace.
        :param width: Width of microstrip trace.
        :param propagation_axis: Axis and direction of signal
            propagation.  This determines the microstrip trace
            direction and sets the signal feed and probes correctly
            for a port.
        :param trace_layer: PCB layer of the signal trace.  Uses
            copper layer index values.
        :param gnd_layer: PCB layer of the ground plane.  Uses copper
            layer index values.  Can be set to None when not using a
            port.
        :param gnd_gap: Gap distance between trace edge and
            surrounding coplanar ground plane.  This is passed as a
            tuple of two floats, specifying the gap for each side of
            the microstrip.  The first value gives the gap distance of
            the lower edge (smaller coordinate system value) and the
            second value gives the gap distance at the upper edge.  In
            other words, the order is not affected by the propagation
            direction.  If either value is left as the default value
            of None, no gap will be set.  Ensure the copper plane is
            removed from the trace layer if this is the case.
        :param terminal_gap: Adds ground gaps to the ends of the
            microstrip trace.  Provided as a tuple of 2 floats, where
            the first value gives the gap at the lower edge and the
            second value gives the gap at the upper edge.  Like
            `gnd_gap` the order is independent of the propagation
            axis.  A value of None sets no terminal gap.
        :param port_number: If the microstrip line is a port, this
            specifies the port number.  If you leave this as None, the
            Microstrip line will not be treated as a port (i.e. you
            can't use it for an excitation and can't measure values
            with it).
        :param excite: Set to True if the microstrip is a port and
            should have an associated excitation.
        :param invert_excitation: If True, flip the excitation to be
            in the opposite direction.
        :param feed_impedance: The feeding impedance value.  The
            default value of None creates an infinite impedance.  If
            you use the default value ensure that the port is
            terminated by a PML.  When performing a characteristic
            impedance measurement use the default value and PML, which
            gives better results than attempting to use a matching
            impedance.
        :param feed_shift: The amount by which to shift the feed
            as a fraction of the total port length.  The final position
            will be influenced by this value but adjusted for the mesh
            used.
        :param ref_impedance: The impedance used to calculate the port
            voltage and current values.  If left as the default value
            of None, the calculated characteristic impedance is used
            to calculate these values.
        :param measurement_shift: The amount by which to shift the
            measurement probes as a fraction of the total port length.
            By default, the measurement port is placed halfway between
            the start and stop.  Like `feed_shift`, the final position
            will be adjusted for the mesh used.  This is important
            since voltage probes need to lie on mesh lines and current
            probes need to be placed equidistant between them.
        :param transform: CSTransform to apply to microstrip.
        """
        self._pcb = pcb
        self._position = position
        self._length = length
        self._width = width
        self._propagation_axis = propagation_axis
        self._check_propagation_axis()
        self._trace_layer = trace_layer
        self._gnd_layer = gnd_layer
        self._gnd_gap = gnd_gap
        self._terminal_gap = terminal_gap
        self._port_number = port_number
        self._excite = excite
        self._invert_excitation = invert_excitation
        self._feed_impedance = feed_impedance
        self._feed_shift = feed_shift
        self._ref_impedance = ref_impedance
        self._measurement_shift = measurement_shift
        self._transform = transform
        self._index = None
        self._polygons = []

        self._check_ref_impedance()
        self._check_coplanar_gap()

        if self.position is not None:
            self.construct(self.position)

    @property
    def port_number(self) -> int:
        """
        """
        return self._port_number

    @property
    def pcb(self) -> int:
        """
        """
        return self._pcb

    @property
    def position(self) -> Coordinate2:
        """
        """
        return self._position

    @property
    def transform(self) -> CSTransform:
        """
        """
        return self._transform

    def construct(
        self, position: Coordinate2, transform: CSTransform = None
    ) -> None:
        """
        """
        self._position = position
        self._transform = append_transform(self._transform, transform)
        self._index = self._get_inc_ctr()
        if self.port_number is not None:
            if self.transform is not None:
                raise ValueError("Ports do not support transforms.")
            self._construct_port()
        else:
            self._construct_trace()

        self._construct_gap()

    def _construct_port(self) -> None:
        """
        """
        box = self._port_box()
        MicrostripPort(
            sim=self.pcb.sim,
            box=box,
            propagation_axis=self._propagation_axis,
            excitation_axis=self._excitation_axis(),
            number=self.port_number,
            thickness=self.pcb.pcb_prop.copper_thickness(self._trace_layer),
            conductivity=self.pcb.pcb_prop.metal_conductivity(),
            excite=self._excite,
            feed_impedance=self._feed_impedance,
            feed_shift=self._feed_shift,
            ref_impedance=self._ref_impedance,
            measurement_shift=self._measurement_shift,
        )
        trace_box = Box2(
            Coordinate2(box.min_corner.x, box.min_corner.y),
            Coordinate2(box.max_corner.x, box.max_corner.y),
        )
        self.polygons.append(trace_box.corners())

    def _construct_trace(self) -> None:
        """
        """
        trace_prop = self.pcb.sim.csx.AddConductingSheet(
            self._microstrip_name(),
            conductivity=self.pcb.pcb_prop.metal_conductivity(),
            thickness=self.pcb.pcb_prop.copper_thickness(self._trace_layer),
        )
        trace_z = self._trace_z()
        prop_axis = self._propagation_axis.axis
        perp_axis = self._trace_perpendicular_axis().axis

        start = [0, 0, trace_z]
        stop = [0, 0, trace_z]
        start[prop_axis] = -self._length / 2
        stop[prop_axis] = self._length / 2
        start[perp_axis] = -self._width / 2
        stop[perp_axis] = self._width / 2

        pos = Coordinate3(self.position.x, self.position.y, 0)
        poly_points = _set_box(
            prop=trace_prop,
            start=start,
            stop=stop,
            position=pos,
            transform=self.transform,
            priority=priorities["trace"],
        )
        self.polygons.append(poly_points)

    def _construct_gap(self) -> None:
        """
        """
        if all(gap is None for gap in self._gnd_gap) and all(
            gap is None for gap in self._terminal_gap
        ):
            return

        freq = self.pcb.sim.reference_frequency
        gap_prop = self.pcb.sim.csx.AddMaterial(
            self._gap_name(),
            epsilon=self.pcb.pcb_prop.substrate.epsr_at_freq(freq),
            kappa=self.pcb.pcb_prop.substrate.kappa_at_freq(freq),
        )
        trace_z = self._trace_z()
        prop_axis = self._propagation_axis.axis
        perp_axis = self._trace_perpendicular_axis().axis

        start = [0, 0, trace_z]
        stop = [0, 0, trace_z]
        start[prop_axis] = -self._length / 2
        stop[prop_axis] = self._length / 2
        start[perp_axis] = -self._width / 2
        stop[perp_axis] = self._width / 2

        if self._gnd_gap[0] is not None:
            start[perp_axis] -= self._gnd_gap[0]
        if self._gnd_gap[1] is not None:
            stop[perp_axis] += self._gnd_gap[1]
        if self._terminal_gap[0] is not None:
            start[prop_axis] -= self._terminal_gap[0]
        if self._terminal_gap[1] is not None:
            stop[prop_axis] += self._terminal_gap[1]

        pos = Coordinate3(self.position.x, self.position.y, 0)
        _set_box(
            prop=gap_prop,
            start=start,
            stop=stop,
            position=pos,
            transform=self.transform,
            priority=priorities["keepout"],
        )

    def _check_propagation_axis(self) -> None:
        """
        """
        if self._propagation_axis.axis == 2:
            raise ValueError(
                "Invalid propagation axis. Must be in either "
                "the x or y directions."
            )

    def _check_ref_impedance(self) -> None:
        """
        """
        if self._port_number is not None and self._ref_impedance is None:
            raise RuntimeWarning(
                "Reference impedance not set for port {}".format(
                    self._port_number
                )
            )

    def _check_coplanar_gap(self) -> None:
        """
        """
        if (
            self._gnd_gap[0] is None or self._gnd_gap[1] is None
        ) and self._trace_layer in self._pcb.copper_pours():
            raise RuntimeWarning(
                "Ground gaps have not been set on the trace layer "
                "where a copper pour has been set. This is most "
                "likely an error. Please check your simulation."
            )

    def _excitation_axis(self) -> None:
        """
        """
        direction = int(np.sign(self._gnd_layer - self._trace_layer))
        if self._invert_excitation:
            direction *= -1
        return Axis("z", direction)

    def _propagation_direction(self) -> int:
        """
        Get the direction of the signal propagation.
        """
        return self._propagation_axis.direction

    def _microstrip_name(self) -> str:
        """
        """
        return "microstrip_" + str(self._index)

    def _gap_name(self) -> str:
        """
        """
        return "microstrip_gap_" + str(self._index)

    def _port_box(self) -> Box3:
        """
        """
        prop_axis = self._propagation_axis.axis
        perp_axis = self._trace_perpendicular_axis().axis
        excite_axis = self._excitation_axis().axis

        prop_bounds = self._prop_bounds()
        perp_bounds = self._trace_perpendicular_bounds()

        box = Box3(
            Coordinate3(None, None, None), Coordinate3(None, None, None)
        )
        box.min_corner[prop_axis] = prop_bounds[0]
        box.max_corner[prop_axis] = prop_bounds[1]
        box.min_corner[perp_axis] = perp_bounds[0]
        box.max_corner[perp_axis] = perp_bounds[1]
        box.min_corner[excite_axis] = self._gnd_z()
        box.max_corner[excite_axis] = self._trace_z()

        return box

    def _prop_bounds(self) -> Tuple[float, float]:
        """
        Minimum and maximum trace propagation positions.  This
        accounts for the direction, so if the direction is negative,
        the minimum position will be larger than the maximum position.
        """
        prop_axis = self._propagation_axis.axis
        min_val = self.position[prop_axis] - (
            self._propagation_direction() * self._length / 2
        )
        max_val = self.position[prop_axis] + (
            self._propagation_direction() * self._length / 2
        )
        return (min_val, max_val)

    def _trace_perpendicular_bounds(self) -> Tuple[float, float]:
        """
        Minimum and maximum trace y positions.  This accounts for the
        direction, so if the direction is negative, the minimum y will
        be larger than the maximum y.  This shouldn't actually matter,
        but is implemented this way for consistency with _x_bounds.
        """
        trace_perp_axis = self._trace_perpendicular_axis().axis
        min_val = self.position[trace_perp_axis] - (
            self._propagation_direction() * self._width / 2
        )
        max_val = self.position[trace_perp_axis] + (
            self._propagation_direction() * self._width / 2
        )
        return (min_val, max_val)

    def _trace_perpendicular_axis(self) -> Axis:
        """
        """
        trace_axes = [0, 1]
        trace_axes.remove(self._propagation_axis.axis)
        return Axis(trace_axes[0])

    def _trace_z(self) -> float:
        """
        """
        return self.pcb.copper_layer_elevation(self._trace_layer)

    def _gnd_z(self) -> float:
        """
        """
        return self.pcb.copper_layer_elevation(self._gnd_layer)


class DifferentialMicrostrip(Structure):
    """
    """

    unique_index = 0

    def __init__(
        self,
        pcb: PCB,
        position: Coordinate2,
        length: float,
        width: float,
        gap: float,
        propagation_axis: Axis,
        trace_layer: int = 0,
        gnd_layer: int = 1,
        gnd_gap: Tuple[float, float] = (None, None),
        terminal_gap: Tuple[float, float] = (None, None),
        port_number: int = None,
        excite: bool = False,
        feed_impedance: float = None,
        feed_shift: float = 0.2,
        ref_impedance: float = None,
        measurement_shift: float = 0.5,
        transform: CSTransform = None,
    ):
        """
        :param gap: Separation between microstrip lines, measured from
            the inner trace edges.
        """
        self._pcb = pcb
        self._position = position
        self._length = length
        self._width = width
        self._gap = gap
        self._propagation_axis = propagation_axis
        self._check_propagation_axis()
        self._trace_layer = trace_layer
        self._gnd_layer = gnd_layer
        self._gnd_gap = gnd_gap
        self._terminal_gap = terminal_gap
        self._port_number = port_number
        self._excite = excite
        self._feed_impedance = feed_impedance
        self._feed_shift = feed_shift
        self._ref_impedance = ref_impedance
        self._measurement_shift = measurement_shift
        self._transform = transform
        self._index = None

        if self._position is not None:
            self.construct(self._position)

    def construct(
        self, position: Coordinate2, transform: CSTransform = None
    ) -> None:
        """
        """
        self._position = position
        self._transform = append_transform(self._transform, transform)
        self._index = self._get_inc_ctr()
        if self._port_number is not None:
            if self._transform is not None:
                raise ValueError("Ports do not support transforms.")
            DifferentialMicrostripPort(
                sim=self._pcb.sim,
                box=self._port_box(),
                propagation_axis=self._propagation_axis,
                excitation_axis=self._excite_axis(),
                number=self._port_number,
                gap=self._gap,
                thickness=self._pcb.pcb_prop.copper_thickness(
                    self._trace_layer
                ),
                conductivity=self._pcb.pcb_prop.metal_conductivity(),
                excite=self._excite,
                feed_impedance=self._feed_impedance,
                feed_shift=self._feed_shift,
                ref_impedance=self._ref_impedance,
                measurement_shift=self._measurement_shift,
            )
        else:
            positions = self._trace_positions()
            for i, pos in enumerate(positions):
                Microstrip(
                    pcb=self._pcb,
                    position=pos,
                    length=self._length,
                    width=self._width,
                    propagation_axis=self._propagation_axis,
                    trace_layer=self._trace_layer,
                    gnd_layer=self._gnd_layer,
                    gnd_gap=(None, None),
                    terminal_gap=(None, None),
                    transform=self._transform,
                )

        self._construct_gap()

    def _construct_gap(self) -> None:
        """
        """
        if self._trace_layer not in self._pcb.copper_pours():
            return

        freq = self._pcb.sim.reference_frequency
        gap_prop = self._pcb.sim.csx.AddMaterial(
            self._gap_name(),
            epsilon=self._pcb.pcb_prop.substrate.epsr_at_freq(freq),
            kappa=self._pcb.pcb_prop.substrate.kappa_at_freq(freq),
        )
        elevation = self._trace_elevation()
        prop_axis = self._propagation_axis.axis
        excite_axis = self._excite_axis().axis
        box = Box3(
            Coordinate3(None, None, elevation),
            Coordinate3(None, None, elevation),
        )

        box.min_corner[prop_axis] = -self._length / 2
        if self._terminal_gap[0] is not None:
            box.min_corner[prop_axis] -= self._terminal_gap[0]
        box.max_corner[prop_axis] = self._length / 2
        if self._terminal_gap[1] is not None:
            box.max_corner[prop_axis] += self._terminal_gap[1]

        box.min_corner[excite_axis] = -self._gap / 2
        if self._gnd_gap[0] is not None:
            box.min_corner[excite_axis] -= self._width + self._gnd_gap[0]
        box.max_corner[excite_axis] = self._gap / 2
        if self._gnd_gap[1] is not None:
            box.max_corner[excite_axis] += self._width + self._gnd_gap[1]

        _set_box(
            prop=gap_prop,
            start=box.start(),
            stop=box.stop(),
            position=Coordinate3(self._position.x, self._position.y, 0),
            transform=self._transform,
            priority=priorities["keepout"],
        )

    def _trace_positions(self) -> Tuple[Coordinate2, Coordinate2]:
        """
        """
        lower_pos = Coordinate2(None, None)
        upper_pos = Coordinate2(None, None)
        prop_axis = self._propagation_axis.axis
        excite_axis = self._excite_axis().axis

        lower_pos[prop_axis] = self._position[prop_axis]
        upper_pos[prop_axis] = self._position[prop_axis]

        lower_pos[excite_axis] = (
            self._position[excite_axis] - self._gap / 2 - self._width / 2
        )
        upper_pos[excite_axis] = (
            self._position[excite_axis] + self._gap / 2 + self._width / 2
        )

    def _check_propagation_axis(self) -> None:
        """
        """
        if self._propagation_axis.axis == 2:
            raise ValueError(
                "Invalid propagation axis. Must be in either "
                "the x or y directions."
            )

    def _normal_axis(self) -> Axis:
        """
        """
        return Axis("z")

    def _excite_axis(self) -> Axis:
        """
        """
        axes = [0, 1]
        axes.remove(self._propagation_axis.axis)
        return Axis(axes[0])

    def _port_box(self) -> Box3:
        """
        """
        elevation = self._trace_elevation()
        box = Box3(
            Coordinate3(None, None, elevation),
            Coordinate3(None, None, elevation),
        )
        prop_axis = self._propagation_axis.axis
        excite_axis = self._excite_axis().axis
        box.min_corner[prop_axis] = (
            self._position[prop_axis] - self._length / 2
        )
        box.max_corner[prop_axis] = (
            self._position[prop_axis] + self._length / 2
        )
        box.min_corner[excite_axis] = (
            self._position[excite_axis] - self._width - self._gap / 2
        )
        box.max_corner[excite_axis] = (
            self._position[excite_axis] + self._width + self._gap / 2
        )

        return box

    def _trace_elevation(self) -> float:
        """
        """
        return self._pcb.copper_layer_elevation(self._trace_layer)

    def _gap_name(self) -> str:
        """
        """
        return "differential_microstrip_gap_" + str(self._index)


class MicrostripCoupler(Structure):
    """
    """

    unique_index = 0

    def __init__(
        self,
        pcb: PCB,
        position: Coordinate2,
        trace_layer: int,
        gnd_layer: int,
        trace_width: float,
        trace_gap: float,
        length: float,
        miter: float = None,
        gnd_gap: Tuple[float, float] = (None, None),
        transform: CSTransform = None,
    ):
        """
        :param pcb: PCB to which this microstrip coupler is added.
        :param position: Center point of the microstrip coupler.  This
            is halfway along the length in the x-direction and in the
            middle of the trace gap in the y-direction.  If the
            position is set to None, `construct` will have to be
            called manually for the coupler to be instantiated.
        :param trace_layer: PCB copper layer on which the traces are
            placed.
        :param gnd_layer: PCB copper layer of the reference ground
            plane.
        :param trace_width: Microstrip trace width.
        :param trace_gap: Distance between coupled line traces.  The
            distance is measured from the inner edge of each trace.
        :param length: Length of the coupled portion of the microstrip
            traces.
        :param miter: The amount to miter the corners are ports three
            and four.  If left as None, an optimal miter estimate will
            be used.  See the `miter` parameter of `Miter`'s
            constructur for details.
        :param gnd_gap: Distance between the microstrip trace outer
            edges and the coplanar ground plane.  A value can be
            provided for each side, starting with the smaller y-value.
            If the copper pour has been omitted from the trace layer,
            leave this as the default None.
        :param transform: Transform to apply to the coupler.
        """
        self._pcb = pcb
        self._position = position
        self._trace_layer = trace_layer
        self._gnd_layer = gnd_layer
        self._trace_width = trace_width
        self._trace_gap = trace_gap
        self._length = length
        self._miter = miter
        self._gnd_gap = gnd_gap
        self._transform = transform
        self._index = None
        self._port_positions = [None, None, None, None]
        self._polygons = []

        if self._position is not None:
            self.construct(self._position)

    def construct(
        self, position: Coordinate2, transform: CSTransform = None
    ) -> None:
        """
        """
        self._position = position
        self._transform = append_transform(self._transform, transform)
        self._index = self._get_inc_ctr()

        self._construct_traces()
        self._construct_trace_gap()
        self._construct_miters()

    def _construct_traces(self) -> None:
        """
        """
        y_dist = self._y_dist()
        for i, ypos in enumerate(
            [self._position.y - y_dist, self._position.y + y_dist]
        ):
            if i == 0:
                gnd_gap = (self._gnd_gap[i], None)
            else:
                gnd_gap = (None, self._gnd_gap[i])

            microstrip = Microstrip(
                pcb=self._pcb,
                position=Coordinate2(self._position.x, ypos),
                length=self._length,
                width=self._trace_width,
                propagation_axis=Axis("x"),
                trace_layer=self._trace_layer,
                gnd_layer=None,
                gnd_gap=gnd_gap,
            )
            self._polygons += microstrip.polygons

    def _construct_trace_gap(self) -> None:
        """
        """
        if self._trace_layer not in self._pcb.copper_pours():
            return

        ref_freq = self._pcb.sim.reference_frequency
        prop = self._pcb.sim.csx.AddMaterial(
            self._gap_name(),
            epsilon=self._pcb.pcb_prop.substrate.epsr_at_freq(ref_freq),
            kappa=self._pcb.pcb_prop.substrate.kappa_at_freq(ref_freq),
        )
        zpos = self._pcb.copper_layer_elevation(self._trace_layer)
        _set_box(
            prop=prop,
            start=[-self._length / 2, -self._trace_gap / 2, zpos],
            stop=[self._length / 2, self._trace_gap / 2, zpos],
            position=Coordinate3(self._position.x, self._position.y, 0),
            transform=self._transform,
            priority=priorities["keepout"],
        )

    def _construct_miters(self) -> None:
        """
        """
        miter = Miter(
            pcb=self._pcb,
            position=None,
            pcb_layer=self._trace_layer,
            gnd_layer=self._gnd_layer,
            trace_width=self._trace_width,
            gap=self._gnd_gap[1],
            miter=self._miter,
            transform=self._transform,
        )
        tr = CSTransform()
        tr.AddTransform("RotateAxis", "z", 90)
        miter.construct(
            position=Coordinate2(
                self._position.x
                - self._length / 2
                - miter.length()
                + self._trace_width / 2,
                self._position.y
                - self._y_dist()
                - miter.length()
                + self._trace_width / 2,
            ),
            transform=tr,
        )
        self._polygons += miter.polygons

        miter = Miter(
            pcb=self._pcb,
            position=Coordinate2(
                self._position.x + self._length / 2,
                self._position.y - self._y_dist(),
            ),
            pcb_layer=self._trace_layer,
            gnd_layer=self._gnd_layer,
            trace_width=self._trace_width,
            gap=self._gnd_gap[1],
            miter=self._miter,
            transform=self._transform,
        )

        self._polygons += miter.polygons

        self._set_port_positions(miter.length())

    def _set_port_positions(self, miter_length: float) -> None:
        """
        """
        self._port_positions[0] = _transformed_coordinate(
            coord=Coordinate2(
                self._position.x - self._length / 2,
                self._position.y + self._y_dist(),
            ),
            transform_origin=self._position,
            transform=self._transform,
        )

        self._port_positions[1] = _transformed_coordinate(
            coord=Coordinate2(
                self._position.x + self._length / 2,
                self._position.y + self._y_dist(),
            ),
            transform_origin=self._position,
            transform=self._transform,
        )

        self._port_positions[2] = _transformed_coordinate(
            coord=Coordinate2(
                self._position.x
                - self._length / 2
                - miter_length
                + self._trace_width / 2,
                self._position.y
                - self._y_dist()
                - miter_length
                + self._trace_width / 2,
            ),
            transform_origin=self._position,
            transform=self._transform,
        )

        self._port_positions[3] = _transformed_coordinate(
            coord=Coordinate2(
                self._position.x
                + self._length / 2
                + miter_length
                - self._trace_width / 2,
                self._position.y
                - self._y_dist()
                - miter_length
                + self._trace_width / 2,
            ),
            transform_origin=self._position,
            transform=self._transform,
        )

    def _gap_name(self) -> str:
        """
        """
        return "Microstrip_Coupler_gap_" + str(self._index)

    def _y_dist(self) -> float:
        """
        Y-distance from coupler center to center of each microstrip
        trace.
        """
        return self._trace_gap / 2 + self._trace_width / 2

    def port_positions(
        self,
    ) -> Tuple[Coordinate2, Coordinate2, Coordinate2, Coordinate2]:
        """
        Center trace positions of the coupler's ports 1, 2, 3, and 4.
        """
        return (
            self._port_positions[0],
            self._port_positions[1],
            self._port_positions[2],
            self._port_positions[3],
        )


class Taper(Structure):
    """
    Trace with different widths at the start and end.  Can be used to
    smoothly transition between a trace of one width to a trace of
    another width.

    The taper proceeds in the positive x-direction, where width1 is
    the width at the lower x-value and width2 is the width at the
    higher x-value.  This can be adjusted with a transformation.
    """

    unique_index = 0

    def __init__(
        self,
        pcb: PCB,
        position: Coordinate2,
        pcb_layer: int,
        width1: float,
        width2: float,
        length: float,
        gap: float = None,
        transform: CSTransform = None,
    ):
        """
        :param pcb: PCB object to which the taper is added.
        :param position: Taper midpoint.  If set to None, the taper
            will need to be constructed later manually with construct.
        :param pcb_layer: PCB copper layer on which the taper should
            be placed.
        :param width1: Leftmost width.
        :param width2: Rightmost width.
        :param length: Taper length.  The width of the taper
            increases/decreases linearly along the length.
        :param gap: Distance between taper and surrounding coplanar
            ground plane.  If gap is set to None, no gap is used.
            Ensure coplanar copper pour is removed if this is used.
        :param transform: Transform applied to taper.
        """
        self._pcb = pcb
        self._position = position
        self._pcb_layer = pcb_layer
        self._width1 = width1
        self._width2 = width2
        self._length = length
        self._gap = gap
        self._transform = transform

        if self.position is not None:
            self.construct(self.position)

    @property
    def pcb(self) -> PCB:
        """
        """
        return self._pcb

    @property
    def position(self) -> Coordinate2:
        """
        """
        return self._position

    @property
    def transform(self) -> CSTransform:
        """
        """
        return self._transform

    @property
    def length(self) -> float:
        """
        """
        return self._length

    @property
    def width1(self) -> float:
        """
        """
        return self._width1

    @property
    def width2(self) -> float:
        """
        """
        return self._width2

    def construct(
        self, position: Coordinate2, transform: CSTransform = None
    ) -> None:
        """
        """
        self._transform = append_transform(self.transform, transform)
        self._position = position
        self._construct_taper()
        self._construct_gap()

    def _construct_taper(self) -> None:
        """
        """
        taper_prop = self.pcb.sim.csx.AddConductingSheet(
            self._taper_name(),
            conductivity=self.pcb.pcb_prop.metal_conductivity(),
            thickness=self.pcb.pcb_prop.copper_thickness(self._pcb_layer),
        )
        pts = self._trapezoid_points(self.width1, self.width2)
        zpos = self._taper_elevation()
        _set_polygon(
            prop=taper_prop,
            points=pts,
            elevation=zpos,
            position=Coordinate3(self.position.x, self.position.y, 0),
            transform=self.transform,
            priority=priorities["trace"],
        )

    def _construct_gap(self) -> None:
        """
        """
        if self._gap is None:
            return

        ref_freq = self.pcb.sim.reference_frequency
        gap_prop = self.pcb.sim.csx.AddMaterial(
            self._gap_name(),
            epsilon=self.pcb.pcb_prop.substrate.epsr_at_freq(ref_freq),
            kappa=self.pcb.pcb_prop.substrate.kappa_at_freq(ref_freq),
        )
        pts = self._trapezoid_points(
            self.width1 + (2 * self._gap), self.width2 + (2 * self._gap)
        )
        zpos = self._taper_elevation()
        _set_polygon(
            prop=gap_prop,
            points=pts,
            elevation=zpos,
            position=Coordinate3(self.position.x, self.position.y, 0),
            transform=self.transform,
            priority=priorities["keepout"],
        )

    def _trapezoid_points(
        self, width1: float, width2: float
    ) -> List[Coordinate2]:
        """
        Returns 4 trapezoid corners in the order bottom left, top
        left, bottom right, top right.
        """
        xmin = -self._length / 2
        xmax = self._length / 2
        yl1 = -width1 / 2
        yl2 = width1 / 2
        yr1 = -width2 / 2
        yr2 = width2 / 2

        return [
            Coordinate2(xmin, yl1),
            Coordinate2(xmin, yl2),
            Coordinate2(xmax, yr2),
            Coordinate2(xmax, yr1),
        ]

    def _taper_name(self) -> str:
        """
        """
        return "taper_" + str(self._get_ctr())

    def _gap_name(self) -> str:
        """
        """
        return "taper_" + str(self._get_ctr()) + "_gap"

    def _taper_elevation(self) -> float:
        """
        """
        return self.pcb.copper_layer_elevation(self._pcb_layer)


class Miter(Structure):
    """
    Microstrip mitered bend.  Currently only supports 90degree bends.
    By default, this will connect to the right side of a microstrip
    travelling in the +x-direction and the top of a microstrip
    travelling in the +y-direction.  Transforms can be used for any
    other configuration.  Transforms will be applied relative to the
    `position` argument.
    """

    unique_index = 0

    def __init__(
        self,
        pcb: PCB,
        position: Coordinate2,
        pcb_layer: int,
        gnd_layer: int,
        trace_width: float,
        gap: float,
        miter: float = None,
        transform: CSTransform = None,
    ):
        """
        :param pcb: PCB object to which the taper is added.
        :param position: Midpoint of the trace at which the mitered
            corner begins.
        :param pcb_layer: PCB copper layer on which the miter should
            be placed.
        :param gnd_layer: PCB copper layer of the ground plane.
        :param trace_width: Microstrip trace width.
        :param miter: Distance between original, unmitered corner and
            mitered edge.  The point on the mitered edge is chosen
            such that distance line from the original corner to the
            new edge is perpendicular to the mitered edge.  If left as
            None, the Douville and James optimal miter is computed.
        :param gap: Distance between taper and surrounding coplanar
            ground plane.  If gap is set to None, no gap is used.
            Ensure coplanar copper pour is removed if this is used.
        :param transform: Transform applied to miter.
        """
        self._pcb = pcb
        self._position = position
        self._pcb_layer = pcb_layer
        self._gnd_layer = gnd_layer
        self._trace_width = trace_width
        if miter is None:
            height = np.abs(
                self.pcb.copper_layer_elevation(pcb_layer)
                - self.pcb.copper_layer_elevation(gnd_layer)
            )
            self._miter = calc.miter(trace_width, height)
        else:
            self._miter = miter
        self._gap = gap
        self._transform = transform
        self._index = self._get_inc_ctr()
        self._polygons = []

        if self.position is not None:
            self.construct(self.position)

    @property
    def pcb(self) -> PCB:
        """
        """
        return self._pcb

    @property
    def position(self) -> Coordinate2:
        """
        """
        return self._position

    @property
    def transform(self) -> CSTransform:
        """
        """
        return self._transform

    @property
    def miter(self) -> float:
        """
        """
        return self._miter

    def corner_length(self) -> float:
        """
        """
        return self._trace_width * np.sqrt(2)

    def end_point(self) -> Coordinate2:
        """
        Coordinate of the end of the miter.  Analogous to
        `self.position` but for the end of the miter.
        """
        inset_len = self._trace_width - self.overlap_length()
        xpos = self.position.x + self._trace_width / 2 + inset_len
        ypos = self.position.y - (self._trace_width / 2) - inset_len

        return Coordinate2(xpos, ypos)

    def construct(
        self, position: Coordinate2, transform: CSTransform = None
    ) -> None:
        """
        """
        self._transform = append_transform(self._transform, transform)
        self._position = position
        self._construct_trace()
        self._construct_gap()

    def _construct_trace(self) -> None:
        """
        """
        prop = self.pcb.sim.csx.AddConductingSheet(
            self._trace_name(),
            conductivity=self.pcb.pcb_prop.metal_conductivity(),
            thickness=self.pcb.pcb_prop.copper_thickness(self._pcb_layer),
        )
        pos = Coordinate3(self.position.x, self.position.y, 0)
        poly_points = _set_polygon(
            prop=prop,
            points=self._trace_points(),
            elevation=self.pcb.copper_layer_elevation(self._pcb_layer),
            position=pos,
            transform=self.transform,
            priority=priorities["trace"],
        )
        self.polygons.append(poly_points)

    def _construct_gap(self) -> None:
        """
        """
        if self._gap is None:
            return

        ref_freq = self.pcb.sim.reference_frequency
        prop = self.pcb.sim.csx.AddMaterial(
            self._gap_name(),
            epsilon=self.pcb.pcb_prop.substrate.epsr_at_freq(ref_freq),
            kappa=self.pcb.pcb_prop.substrate.kappa_at_freq(ref_freq),
        )
        pos = Coordinate3(self.position.x, self.position.y, 0)
        _set_polygon(
            prop=prop,
            points=self._gap_points(),
            elevation=self.pcb.copper_layer_elevation(self._pcb_layer),
            position=pos,
            transform=self.transform,
            priority=priorities["keepout"],
        )

    def length(self) -> float:
        """
        Length of miter in x- or y-dimension (it is the same for each).
        """
        return self.inset_length() + self._trace_width

    def _trace_points(self) -> List[Coordinate2]:
        """
        List of miter x and y-coordinates such that self.position is
        taken as the origin.  See _set_polygon for how these points
        are used.
        """
        inset_len = self.inset_length()
        pts = []
        # 1st point from top left, proceeding counterclockwise
        pts.append(Coordinate2(0, self._trace_width / 2))
        # 2
        pts.append(Coordinate2(0, -self._trace_width / 2))
        # 3
        pts.append(Coordinate2(inset_len, -self._trace_width / 2))
        # 4
        pts.append(Coordinate2(inset_len, -self._trace_width / 2 - inset_len))
        # 5
        pts.append(
            Coordinate2(
                inset_len + self._trace_width,
                -self._trace_width / 2 - inset_len,
            )
        )

        return pts

    def _gap_points(self) -> List[Coordinate2]:
        """
        """
        inset_len = self.inset_length()
        pts = []
        # 1
        pts.append(0, self._trace_width / 2 + self._gap)
        # 2
        pts.append(0, -self._trace_width / 2 - inset_len)
        # 3
        pts.append(
            self._trace_width + inset_len + self._gap,
            -self._trace_width / 2 - inset_len,
        )
        # 4
        pts.append(
            self._trace_width + inset_len + self._gap,
            -self._trace_width / 2 - inset_len + (self._gap / np.sqrt(2)),
        )
        # 5
        pts.append(self._gap / np.sqrt(2), self._trace_width / 2 + self._gap)

        return pts

    def overlap_length(self) -> float:
        """
        """
        corner_len = self.corner_length()
        if self._miter > corner_len:
            raise RuntimeError("Miter is larger than corner length.")
        return (self.corner_length() - self._miter) * np.sqrt(2)

    def inset_length(self) -> float:
        """
        """
        return self._trace_width - self.overlap_length()

    def _trace_name(self) -> str:
        """
        """
        return "miter_trace_" + str(self._index)

    def _gap_name(self) -> str:
        """
        """
        return "miter_gap_" + str(self._index)


class SMDPassiveDimensions:
    """
    """

    def __init__(self, length: float, width: float, height: float):
        """
        """
        self._unit_set = False
        self._length = length
        self._width = width
        self._height = height

    @property
    def length(self) -> float:
        """
        """
        if not self._unit_set:
            raise RuntimeError(
                "Set SMD passive unit before accessing dimensions."
            )
        return self._length

    @property
    def width(self) -> float:
        """
        """
        if not self._unit_set:
            raise RuntimeError(
                "Set SMD passive unit before accessing dimensions."
            )
        return self._width

    @property
    def height(self) -> float:
        """
        """
        if not self._unit_set:
            raise RuntimeError(
                "Set SMD passive unit before accessing dimensions."
            )
        return self._height

    def set_unit(self, unit: float) -> None:
        """
        """
        if self._unit_set:
            return
        self._length /= unit
        self._width /= unit
        self._height /= unit
        self._unit_set = True


common_smd_passives = {
    "0201C": SMDPassiveDimensions(length=0.6e-3, width=0.3e-3, height=0.3e-3),
    "0402C": SMDPassiveDimensions(length=1e-3, width=0.5e-3, height=0.5e-3),
    "0603C": SMDPassiveDimensions(length=1.6e-3, width=0.8e-3, height=0.8e-3),
}


class SMDPassive(Structure):
    """
    Small surface-mount capacitor, resistor, or inductor.

    SMD Passives do not support transforms, since the resistive,
    capacitive and inductive elements must be specified in a direction
    parallel to a coordinate axis.
    """

    unique_index = 0

    def __init__(
        self,
        pcb: PCB,
        position: Coordinate2,
        axis: Axis,
        dimensions: SMDPassiveDimensions,
        pad_width: float,
        pad_length: float,
        gap: float = None,
        c: float = None,
        r: float = None,
        l: float = None,
        pcb_layer: int = 0,
        gnd_cutout_width: float = 0,
        gnd_cutout_length: float = 0,
    ):
        """
        :param pcb: PCB object to which this SMD will be added.
        :param position: Position of the center of the SMD passive on
            the PCB.  This can be set to None, in which case construct
            will need to be called manually to create the SMD.
        :param axis: Signal propagation axis.  Necessary to set the
            resistive, capacitive and inductive values.
        :param dimensions: SMD passive dimensions.  Use meters, rather
            than simulation default unit.
        :param pad_width: SMD pad width.
        :param pad_length: SMD pad length.
        :param gap: Coplanar ground plane keepout distance.  The
            distance from the pad edge to adjacent ground plane.  If
            set to None, no gap is used.  Ensure coplanar copper pour
            is removed if this is used.
        :param c: Capacitance value (in farads).
        :param r: Resistance value (in ohms).
        :param l: Inductance value (in henrys)
        :param pcb_layer: PCB copper layer where the SMD is placed.
            Must be set to the top or bottom layer.
        :param gnd_cutout_width: Width of the ground cutout, as a
            proportion of the pad width.
        :param gnd_cutout_length: Length of the ground cutout, as a
            proportion of length between the ends of the pads.
        """
        self._pcb = pcb
        self._position = position
        if axis.axis == 2:
            raise ValueError("Axis must either point in x or y-directions.")
        self._axis = axis
        self._dimensions = dimensions
        self._dimensions.set_unit(self.pcb.sim.unit)
        self._pad_width = pad_width
        self._pad_length = pad_length
        self._gap = gap
        self._c = c
        self._r = r
        self._l = l
        if self._l is not None:
            raise RuntimeWarning(
                "Setting an inductance value has no effect. OpenEMS "
                "does not support lumped inductances yet."
            )
        self._pcb_layer = pcb_layer
        self._check_pcb_layer()
        self._gnd_cutout_width = gnd_cutout_width * pad_width
        self._gnd_cutout_length = gnd_cutout_length * (
            pad_length + self._dimensions.length
        )

        if self.position is not None:
            self.construct(self.position)

    @property
    def position(self) -> Coordinate2:
        """
        """
        return self._position

    @property
    def dimensions(self) -> SMDPassiveDimensions:
        """
        """
        return self._dimensions

    @property
    def pcb(self) -> PCB:
        """
        """
        return self._pcb

    def construct(self, position: Coordinate2) -> None:
        """
        """
        self._position = position
        self._construct_pads()
        self._construct_smd()
        self._construct_gap()
        self._construct_cutout()

    def _construct_pads(self) -> None:
        """
        """
        pad_prop = self.pcb.sim.csx.AddConductingSheet(
            self._pad_name(),
            conductivity=self.pcb.pcb_prop.metal_conductivity(),
            thickness=self.pcb.pcb_prop.copper_thickness(self._pcb_layer),
        )
        prop_axis = self._axis.axis
        orth_axis = self._orthogonal_axis().axis
        zpos = self._pad_elevation()
        for pad_middle in [
            -self.dimensions.length / 2,
            self.dimensions.length / 2,
        ]:
            start = [None, None, zpos]
            stop = [None, None, zpos]
            start[prop_axis] = pad_middle - self._pad_length / 2
            stop[prop_axis] = pad_middle + self._pad_length / 2
            start[orth_axis] = -self._pad_width / 2
            stop[orth_axis] = self._pad_width / 2
            _set_box(
                prop=pad_prop,
                start=start,
                stop=stop,
                position=Coordinate3(self.position.x, self.position.y, 0),
                transform=None,
                priority=priorities["trace"],
            )

    def _construct_smd(self) -> None:
        """
        """
        smd_prop = self.pcb.sim.csx.AddLumpedElement(
            self._smd_name(), ny=self._axis.axis, caps=False
        )
        if self._r is not None:
            smd_prop.SetResistance(self._r)
        if self._c is not None:
            smd_prop.SetCapacity(self._c)
        if self._l is not None:
            smd_prop.SetInductance(self._l)

        prop_axis = self._axis.axis
        orth_axis = self._orthogonal_axis().axis
        start = [None, None, self._pad_elevation()]
        stop = [None, None, self._pad_elevation() + self.dimensions.height]
        start[prop_axis] = -self.dimensions.length / 2
        stop[prop_axis] = self.dimensions.length / 2
        start[orth_axis] = -self.dimensions.width / 2
        stop[orth_axis] = self.dimensions.width / 2
        _set_box(
            prop=smd_prop,
            start=start,
            stop=stop,
            position=Coordinate3(self.position.x, self.position.y, 0),
            transform=None,
            priority=priorities["component"],
        )

    def _construct_gap(self) -> None:
        """
        """
        if self._gap is None:
            return

        prop_axis = self._axis.axis
        orth_axis = self._orthogonal_axis().axis
        zpos = self._pad_elevation()
        start = [None, None, zpos]
        stop = [None, None, zpos]
        start[prop_axis] = -(self.dimensions.length / 2) - (
            self._pad_length / 2
        )
        stop[prop_axis] = (self.dimensions.length / 2) + (self._pad_length / 2)
        start[orth_axis] = -(self._pad_width / 2) - self._gap
        stop[orth_axis] = (self._pad_width / 2) + self._gap

        ref_freq = self.pcb.sim.reference_frequency
        gap_prop = self.pcb.sim.csx.AddMaterial(
            self._gap_name(),
            epsilon=self.pcb.pcb_prop.substrate.epsr_at_freq(ref_freq),
            kappa=self.pcb.pcb_prop.substrate.kappa_at_freq(ref_freq),
        )
        _set_box(
            prop=gap_prop,
            start=start,
            stop=stop,
            position=Coordinate3(self.position.x, self.position.y, 0),
            transform=None,
            priority=priorities["keepout"],
        )

    def _construct_cutout(self) -> None:
        """
        """
        if self._gnd_cutout_length == 0 or self._gnd_cutout_width == 0:
            return

        prop_axis = self._axis.axis
        orth_axis = self._orthogonal_axis().axis
        zpos = self._gnd_elevation()
        start = [None, None, zpos]
        stop = [None, None, zpos]
        start[prop_axis] = -(self._gnd_cutout_length / 2)
        stop[prop_axis] = self._gnd_cutout_length / 2
        start[orth_axis] = -(self._gnd_cutout_width / 2)
        stop[orth_axis] = self._gnd_cutout_width / 2

        ref_freq = self.pcb.sim.reference_frequency
        cutout_prop = self.pcb.sim.csx.AddMaterial(
            self._cutout_name(),
            epsilon=self.pcb.pcb_prop.substrate.epsr_at_freq(ref_freq),
            kappa=self.pcb.pcb_prop.substrate.kappa_at_freq(ref_freq),
        )
        _set_box(
            prop=cutout_prop,
            start=start,
            stop=stop,
            position=Coordinate3(self.position.x, self.position.y, 0),
            transform=None,
            priority=priorities["keepout"],
        )

    def _orthogonal_axis(self) -> Axis:
        """
        Axis in the PCB plane orthogonal to the propagation axis.
        """
        prop_axis = self._axis.axis
        axes = [Axis("x"), Axis("y")]
        del axes[prop_axis]
        return axes[0]

    def _smd_name(self) -> str:
        """
        """
        return "SMDPassive_" + str(self._get_ctr())

    def _pad_name(self) -> str:
        """
        """
        return self._smd_name() + "_pad"

    def _gap_name(self) -> str:
        """
        """
        return self._smd_name() + "_gap"

    def _cutout_name(self) -> str:
        """
        """
        return self._smd_name() + "_cutout"

    def _pad_elevation(self) -> float:
        """
        """
        return self.pcb.copper_layer_elevation(self._pcb_layer)

    def _gnd_elevation(self) -> float:
        """
        """
        if self._upside_down():
            gnd_layer = self._pcb_layer - 1
        else:
            gnd_layer = self._pcb_layer + 1
        return self.pcb.copper_layer_elevation(gnd_layer)

    def _check_pcb_layer(self) -> None:
        """
        """
        if (
            self._pcb_layer != self.pcb.copper_layers()[0]
            and self._pcb_layer != self.pcb.copper_layers()[-1]
        ):
            raise ValueError("Invalid copper layer for SMD passive.")

    def _upside_down(self) -> bool:
        """
        """
        if self._pcb_layer == self.pcb.copper_layers()[0]:
            return False
        else:
            return True


class WaveguideDimensions:
    """
    """

    def __init__(self, a: float, b: float):
        """
        """
        self._unit_set = False
        self._a = a
        self._b = b

    @property
    def a(self) -> float:
        """
        """
        if not self._unit_set:
            raise RuntimeError(
                "Set waveguide unit before accessing dimensions."
            )
        return self._a

    @property
    def b(self) -> float:
        """
        """
        if not self._unit_set:
            raise RuntimeError(
                "Set waveguide unit before accessing dimensions."
            )
        return self._b

    def set_unit(self, unit: float) -> None:
        """
        """
        if self._unit_set:
            return
        self._a /= unit
        self._b /= unit
        self._unit_set = True


# See https://www.everythingrf.com/tech-resources/waveguides-sizes
standard_waveguides = {"WR159": WaveguideDimensions(40.386e-3, 20.193e-3)}


class Coax(Structure):
    """
    Coaxial cable structure.
    """

    unique_index = 0

    def __init__(
        self,
        sim: Simulation,
        position: Coordinate3,
        length: float,
        radius: float,
        core_radius: float,
        shield_thickness: float,
        dielectric: Dielectric,
        propagation_axis: Axis,
        port_number: int = None,
        excite: bool = False,
        feed_impedance: float = None,
        feed_shift: float = 0.2,
        ref_impedance: float = None,
        measurement_shift: float = 0.5,
        delay: float = 0,
        transform: CSTransform = None,
    ):
        """
        :param sim: Simulation to which the coaxial cable is added.
        :param position: Coaxial cable center.  If set to None, the
            coaxial cable will not be immediately constructed.  In
            this case, the cable must be manually constructed with
            `construct`.
        :param length: Length of cable.
        :param radius: For a cross-section of the coaxial cable, this
            is the distance from the center to the inside edge of the
            outer conducting shield.
        :param core_radius: Copper core radius.
        :param shield_thickness: Outer conductive shield thickness.
        :param dielectric: Dielectric material used between inner
            copper core and outer shield.
        :param propagation_axis: Signal propagation axis and
            direction.
        :param port_number: If the coaxial cable is a port, this
            specifies the port number.  If you leave this as None, the
            coaxial cable will not be treated as a port (i.e. you
            can't use it for an excitation and can't measure values
            with it).
        :param excite: Set to True if the coaxial cable is a port and
            should have an associated excitation.
        :param feed_impedance: See CoaxPort.
        :param feed_shift: See CoaxPort.
        :param ref_impedance: See CoaxPort.
        :param measurement_shift: See CoaxPort.
        :param delay:
        :param transform: CSTransform to apply to coaxial cable.
        """
        super().__init__(sim=sim)
        self._position = position
        self._length = length
        self._radius = radius
        self._core_radius = core_radius
        self._shield_thickness = shield_thickness
        self._dielectric = dielectric
        self._propagation_axis = propagation_axis
        self._port_number = port_number
        self._excite = excite
        self._feed_impedance = feed_impedance
        self._feed_shift = feed_shift
        self._ref_impedance = ref_impedance
        self._measurement_shift = measurement_shift
        self._delay = delay
        self._transform = transform

        self._index = None

        if self._position is not None:
            self.construct(self._position)

    def construct(
        self, position: Coordinate3, transform: CSTransform = None
    ) -> None:
        """
        """
        self._index = self._get_inc_ctr()
        self._position = position
        self._transform = append_transform(self._transform, transform)
        self._construct_core()
        self._construct_dielectric()
        self._construct_shield()

    def _construct_core(self) -> None:
        """
        """
        if self._port_number is not None:
            self._construct_port_core()
        else:
            self._construct_nonport_core()

    def _construct_port_core(self) -> None:
        """
        """
        if self._transform is not None:
            raise ValueError("Ports do not support transforms.")

        if self._propagation_axis.direction == 1:
            start = self._start()
            stop = self._stop()
        else:
            start = self._stop()
            stop = self._start()

        CoaxPort(
            sim=self.sim,
            number=self._port_number,
            start=start,
            stop=stop,
            radius=self._radius,
            core_radius=self._core_radius,
            excite=self._excite,
            feed_shift=self._feed_shift,
            feed_impedance=self._feed_impedance,
            measurement_shift=self._measurement_shift,
            delay=self._delay,
            ref_impedance=self._ref_impedance,
        )

    def _construct_nonport_core(self) -> None:
        """
        """
        core_prop = self.sim.csx.AddMetal(self._core_name())
        _set_cylinder(
            prop=core_prop,
            start=self._start(),
            stop=self._stop(),
            radius=self._core_radius,
            transform=self._transform,
            priority=priorities["trace"],
        )

    def _construct_dielectric(self) -> None:
        """
        """
        ref_freq = self.sim.reference_frequency
        dielectric_prop = self.sim.csx.AddMaterial(
            self._dielectric_name(),
            epsilon=self._dielectric.epsr_at_freq(ref_freq),
            kappa=self._dielectric.kappa_at_freq(ref_freq),
        )
        _set_cylindrical_shell(
            prop=dielectric_prop,
            start=self._start(),
            stop=self._stop(),
            inner_radius=self._core_radius,
            outer_radius=self._radius,
            transform=self._transform,
            priority=priorities["substrate"],
        )

    def _construct_shield(self) -> None:
        """
        """
        shield_prop = self.sim.csx.AddMetal(self._shield_name())
        _set_cylindrical_shell(
            prop=shield_prop,
            start=self._start(),
            stop=self._stop(),
            inner_radius=self._radius,
            outer_radius=self._radius + self._shield_thickness,
            transform=self._transform,
            priority=priorities["ground"],
        )

    def _start(self) -> Coordinate3:
        """
        """
        prop_axis = self._propagation_axis.intval()
        pos = self._position.coordinate_list()
        pos[prop_axis] -= self._length / 2
        if self._transform is not None:
            pos = self._transform.Transform(pos)
        return Coordinate3(pos[0], pos[1], pos[2])

    def _stop(self) -> Coordinate3:
        """
        """
        prop_axis = self._propagation_axis.intval()
        pos = self._position.coordinate_list()
        pos[prop_axis] += self._length / 2
        if self._transform is not None:
            pos = self._transform.Transform(pos)
        return Coordinate3(pos[0], pos[1], pos[2])

    def _core_name(self) -> str:
        """
        """
        return "Coax_core_" + str(self._index)

    def _dielectric_name(self) -> str:
        """
        """
        return "Coax_dielectric_" + str(self._index)

    def _shield_name(self) -> str:
        """
        """
        return "Coax_shield_" + str(self._index)
