"""
A collection of macro structures.  These are frequently used
combinations of primitives, such as a via, microstrip line, etc.  This
allows you, for instance, to add a parameterized via, rather than a
cylindrical shell, air cylinder, circular pads, etc.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple
from copy import deepcopy
import numpy as np
from CSXCAD.CSTransform import CSTransform
from CSXCAD.CSProperties import CSProperties
from CSXCAD.CSPrimitives import CSPrimitives
from pyems.pcb import PCBProperties
from pyems.utilities import apply_transform, append_transform
from pyems.coordinate import Coordinate2, Coordinate3, Axis, Box2, Box3
from pyems.simulation import Simulation
from pyems.port import MicrostripPort, CoaxPort
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


def _set_box(
    prop: CSProperties,
    start: List[float],
    stop: List[float],
    position: Coordinate3,
    transform: CSTransform,
    priority: int,
) -> None:
    """
    Add a box by first constructing the box at the origin, then
    transforming it and finally translating it for the desired
    position.
    """
    box = prop.AddBox(priority=priority, start=start, stop=stop)
    apply_transform(box, transform)
    translate = CSTransform()
    translate.AddTransform("Translate", position.coordinate_list())
    apply_transform(box, translate)


def _set_polygon(
    prop: CSProperties,
    points: List[List[float]],
    elevation: float,
    position: Coordinate3,
    transform: CSTransform,
    priority: int,
) -> None:
    """
    """
    poly = prop.AddPolygon(
        points=points, norm_dir=2, elevation=elevation, priority=priority,
    )
    translate_vec = position.coordinate_list()
    translate = CSTransform()
    translate.AddTransform("Translate", translate_vec)
    apply_transform(poly, transform)
    apply_transform(poly, translate)


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
        layer_prop = self.sim.csx.AddMaterial(
            self._layer_name(layer_index),
            epsilon=self.pcb_prop.substrate.epsr_at_freq(
                self.sim.center_frequency()
            ),
            kappa=self.pcb_prop.substrate.kappa_at_freq(
                self.sim.center_frequency()
            ),
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
        self._noconnect_layers = self._set_noconnect_layers(noconnect_layers)
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

    def _set_noconnect_layers(self, layers: List[int]) -> List[int]:
        """
        """
        valid_layers = []
        for layer in layers:
            if layer in self.layers:
                valid_layers.append(layer)
            else:
                raise RuntimeWarning(
                    "Via no-connect layer specified for layer where via "
                    "isn't present. No-connect for layer {} will be "
                    "ignored. Check your code.".format(layer)
                )
        return valid_layers

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
                    self.pcb.sim.center_frequency()
                ),
                kappa=self.pcb.pcb_prop.substrate.kappa_at_freq(
                    self.pcb.sim.center_frequency()
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


class Microstrip(Structure):
    """
    Microstrip transmission line structure.  This can also be set to
    act as a port for later analysis.  When used as a port, the
    microstrip cannot be transformed, since ports do not support
    transformations.
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
        gnd_gap: float = None,
        via_gap: float = None,
        terminal_gaps: Tuple[bool] = (False, False),
        via: Via = None,
        via_spacing: float = None,
        shorten_via_wall: Tuple[float, float] = (0, 0),
        port_number: int = None,
        excite: bool = False,
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
            layer index values.
        :param gnd_gap: Gap distance between trace edge and
            surrounding coplanar ground plane.  If left as the default
            value of None, no gap will be set.  Ensure the copper
            plane is removed from the trace layer if this is the case.
        :param via_gap: Gap distance between the start of the coplanar
            ground plane and the surrounding via fence.  If set to
            None, a via fence is not used.
        :param terminal_gaps: Adds ground gaps to the ends of the
            microstrip trace.  Provided as a tuple of 2 booleans,
            indicating whether to add gaps to the start and stop of
            the microstrip, respectively.
        :param via: Via object to use for the via fence.  If set to
            None and a via_gap is specified, an unbroken metal sheet
            is used to approximate the via fence.  This can reduce
            simulation time and is often a good approximation to the
            effect produced by a via fence.  Note that by using the
            approximation you lose the ability to use blind/buried
            vias.
        :param via_spacing: Spacing between consecutive vias in the
            via fence.  This will only have an effect if via_gap is
            specified.  In this case, a value must be provided.
        :param shorten_via_wall: Shorten the via wall by this amount
            at the start and end.
        :param port_number: If the microstrip line is a port, this
            specifies the port number.  If you leave this as None, the
            Microstrip line will not be treated as a port (i.e. you
            can't use it for an excitation and can't measure values
            with it).
        :param excite: Set to True if the microstrip is a port and
            should have an associated excitation.
        :param feed_impedance: See PlanarPort.
        :param feed_shift: See PlanarPort.
        :param ref_impedance: See PlanarPort.
        :param measurement_shift: See PlanarPort.
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
        if gnd_gap is None:
            self._via_gap = None
        else:
            self._via_gap = via_gap
        self._terminal_gaps = terminal_gaps
        self._via = via
        self._via_spacing = via_spacing
        self._shorten_via_wall = shorten_via_wall
        self._port_number = port_number
        self._excite = excite
        self._feed_impedance = feed_impedance
        self._feed_shift = feed_shift
        self._ref_impedance = ref_impedance
        self._measurement_shift = measurement_shift
        self._transform = transform
        self._index = None

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
        self._construct_via_fence()

    def _construct_port(self) -> None:
        """
        """
        MicrostripPort(
            sim=self.pcb.sim,
            box=self._port_box(),
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
        _set_box(
            prop=trace_prop,
            start=start,
            stop=stop,
            position=pos,
            transform=self.transform,
            priority=priorities["trace"],
        )

    def _construct_gap(self) -> None:
        """
        """
        if self._gnd_gap is None:
            return

        freq = self.pcb.sim.center_frequency()
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

        start[perp_axis] -= self._gnd_gap
        stop[perp_axis] += self._gnd_gap
        if self._terminal_gaps[0]:
            if self._propagation_axis.is_positive_direction():
                start[prop_axis] -= self._gnd_gap
            else:
                stop[prop_axis] += self._gnd_gap
        if self._terminal_gaps[1]:
            if self._propagation_axis.is_positive_direction():
                stop[prop_axis] += self._gnd_gap
            else:
                start[prop_axis] -= self._gnd_gap
        pos = Coordinate3(self.position.x, self.position.y, 0)
        _set_box(
            prop=gap_prop,
            start=start,
            stop=stop,
            position=pos,
            transform=self.transform,
            priority=priorities["keepout"],
        )

    def _construct_via_fence(self) -> None:
        """
        """
        if self._via_gap is None:
            return

        perp_low, perp_high = self._via_perp_pos()
        if self._via is None:
            self._construct_via_wall(perp_low)
            self._construct_via_wall(perp_high)
        else:
            # TODO
            raise RuntimeError(
                "Vias for via fence have not yet been properly "
                "setup for transforms."
            )
            prop_axis = self._propagation_axis.axis
            perp_axis = self._trace_perpendicular_axis().axis
            prop_max = self.box.max_corner[prop_axis]
            bound_spacing = self._via_spacing / 2
            prop_pos = self.box.min_corner[prop_axis] + bound_spacing
            via_rad = self._via.pad_radius()
            while prop_pos + via_rad < prop_max:
                pos_low = Coordinate2(None, None)
                pos_low[prop_axis] = prop_pos
                pos_low[perp_axis] = perp_low

                pos_high = Coordinate2(None, None)
                pos_high[prop_axis] = prop_pos
                pos_high[perp_axis] = perp_high

                self._via.construct(pos_low)
                self._via.construct(pos_high)
                prop_pos += self._via_spacing

    def _construct_via_wall(self, perp_pos: float) -> None:
        """
        """
        via_prop = self.pcb.sim.csx.AddConductingSheet(
            self._via_wall_name(),
            conductivity=self.pcb.pcb_prop.metal_conductivity(),
            thickness=self.pcb.pcb_prop.copper_thickness(0),
        )

        prop_axis = self._propagation_axis.axis
        perp_axis = self._trace_perpendicular_axis().axis
        start = [
            None,
            None,
            self.pcb.copper_layer_elevation(self.pcb.copper_layers()[-1]),
        ]
        start[prop_axis] = -self._length / 2 + self._shorten_via_wall[0]
        start[perp_axis] = perp_pos

        stop = [
            None,
            None,
            self.pcb.copper_layer_elevation(self.pcb.copper_layers()[0]),
        ]
        stop[prop_axis] = self._length / 2 - self._shorten_via_wall[1]
        stop[perp_axis] = perp_pos

        pos = Coordinate3(self.position.x, self.position.y, 0)
        _set_box(
            prop=via_prop,
            start=start,
            stop=stop,
            position=pos,
            transform=self.transform,
            priority=priorities["ground"],
        )

    def _check_propagation_axis(self) -> None:
        """
        """
        if self._propagation_axis.axis == 2:
            raise ValueError(
                "Invalid propagation axis. Must be in either "
                "the x or y directions."
            )

    def _excitation_axis(self) -> None:
        """
        """
        direction = int(np.sign(self._gnd_layer - self._trace_layer))
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

    def _via_wall_name(self) -> str:
        """
        """
        return "via_wall_" + str(self._index)

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

    def _via_perp_pos(self) -> Tuple[float, float]:
        """
        """
        return (
            -self._propagation_direction()
            * (self._width / 2 + self._gnd_gap + self._via_gap),
            self._propagation_direction()
            * (self._width / 2 + self._gnd_gap + self._via_gap),
        )

    def _trace_z(self) -> float:
        """
        """
        return self.pcb.copper_layer_elevation(self._trace_layer)

    def _gnd_z(self) -> float:
        """
        """
        return self.pcb.copper_layer_elevation(self._gnd_layer)


class Taper(Structure):
    """
    Trace with different widths at the start and end.  Can be used to
    smoothly transition between a trace of one width to a trace of
    another width.

    The taper proceeds in the positive x-direction, where width1 is
    the width1 is the width at the lower x-value and width2 is the
    width at the higher x-value.  This can be adjusted with a
    transformation.
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
        gap: float,
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

        center_freq = self.pcb.sim.center_frequency()
        gap_prop = self.pcb.sim.csx.AddMaterial(
            self._gap_name(),
            epsilon=self.pcb.pcb_prop.substrate.epsr_at_freq(center_freq),
            kappa=self.pcb.pcb_prop.substrate.kappa_at_freq(center_freq),
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
    ) -> List[List[float]]:
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

        return [[xmin, xmin, xmax, xmax], [yl1, yl2, yr2, yr1]]

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
    Microstrip mitered bend. Currently only supports 90degree bends.
    """

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

    def construct(
        self, position: Coordinate2, transform: CSTransform = None
    ) -> None:
        """
        """
        self._transform = append_transform(self._transform, transform)
        self._position = position
        self._construct_trace()
        self._construct_gap()


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
    "0402C": SMDPassiveDimensions(length=1e-3, width=0.5e-3, height=0.5e-3)
}


class SMDPassive(Structure):
    """
    Small surface-mount capacitor, resistor, or inductor.
    """

    unique_index = 0

    def __init__(
        self,
        pcb: PCB,
        position: Coordinate2,
        dimensions: SMDPassiveDimensions,
        pad_width: float,
        pad_length: float,
        gap: float,
        c: float = 0,
        r: float = 0,
        l: float = 0,
        pcb_layer: int = 0,
        gnd_cutout_width: float = 0,
        gnd_cutout_length: float = 0,
        taper: Taper = None,
        transform: CSTransform = None,
    ):
        """
        :param pcb: PCB object to which this SMD will be added.
        :param position: Position of the center of the SMD passive on
            the PCB.  This can be set to None, in which case construct
            will need to be called manually to create the SMD.
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
        :param taper: Taper the transition between the trace and SMD
            pad.  None means do not add a taper.
        :param transform: Transform applied to SMD.
        """
        self._pcb = pcb
        self._position = position
        self._dimensions = dimensions
        self._dimensions.set_unit(self.pcb.sim.unit)
        self._pad_width = pad_width
        self._pad_length = pad_length
        self._gap = gap
        self._c = c
        self._r = r
        self._l = l
        self._pcb_layer = pcb_layer
        self._check_pcb_layer()
        self._gnd_cutout_width = gnd_cutout_width * pad_width
        self._gnd_cutout_length = gnd_cutout_length * (
            pad_length + self._dimensions.length
        )
        self._taper = taper
        self._transform = transform

        if self.position is not None:
            self.construct(self.position)

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
    def dimensions(self) -> SMDPassiveDimensions:
        """
        """
        return self._dimensions

    @property
    def pcb(self) -> PCB:
        """
        """
        return self._pcb

    def construct(
        self, position: Coordinate2, transform: CSTransform = None
    ) -> None:
        """
        """
        self._transform = append_transform(self._transform, transform)
        self._position = position
        self._construct_pads()
        self._construct_smd()
        self._construct_gap()
        self._construct_cutout()
        self._construct_taper()

    def _construct_pads(self) -> None:
        """
        """
        pad_prop = self.pcb.sim.csx.AddConductingSheet(
            self._pad_name(),
            conductivity=self.pcb.pcb_prop.metal_conductivity(),
            thickness=self.pcb.pcb_prop.copper_thickness(self._pcb_layer),
        )
        zpos = self._pad_elevation()
        for pad_middle in [
            -self.dimensions.length / 2,
            self.dimensions.length / 2,
        ]:
            xmin = pad_middle - self._pad_length / 2
            xmax = pad_middle + self._pad_length / 2
            ymin = -self._pad_width / 2
            ymax = self._pad_width / 2
            _set_box(
                prop=pad_prop,
                start=[xmin, ymin, zpos],
                stop=[xmax, ymax, zpos],
                position=Coordinate3(self.position.x, self.position.y, 0),
                transform=self.transform,
                priority=priorities["trace"],
            )

    def _construct_smd(self) -> None:
        """
        """
        smd_prop = self.pcb.sim.csx.AddLumpedElement(
            self._smd_name(), ny=0, caps=True, R=self._r, C=self._c, L=self._l
        )
        _set_box(
            prop=smd_prop,
            start=[
                -self.dimensions.length / 2,
                -self.dimensions.width / 2,
                self._pad_elevation(),
            ],
            stop=[
                self.dimensions.length / 2,
                self.dimensions.width / 2,
                self._pad_elevation() + self.dimensions.height,
            ],
            position=Coordinate3(self.position.x, self.position.y, 0),
            transform=self.transform,
            priority=priorities["trace"],
        )

    def _construct_gap(self) -> None:
        """
        """
        if self._gap is None:
            return

        xmin = -(self.dimensions.length / 2) - (self._pad_length / 2)
        xmax = (self.dimensions.length / 2) + (self._pad_length / 2)
        ymin = -(self.dimensions.width / 2) - self._gap
        ymax = (self.dimensions.width / 2) + self._gap
        zpos = self._pad_elevation()

        center_freq = self.pcb.sim.center_frequency()
        gap_prop = self.pcb.sim.csx.AddMaterial(
            self._gap_name(),
            epsilon=self.pcb.pcb_prop.substrate.epsr_at_freq(center_freq),
            kappa=self.pcb.pcb_prop.substrate.kappa_at_freq(center_freq),
        )
        _set_box(
            prop=gap_prop,
            start=[xmin, ymin, zpos],
            stop=[xmax, ymax, zpos],
            position=Coordinate3(self.position.x, self.position.y, 0),
            transform=self.transform,
            priority=priorities["keepout"],
        )

    def _construct_cutout(self) -> None:
        """
        """
        if self._gnd_cutout_length == 0 or self._gnd_cutout_width == 0:
            return
        xmin = -(self._gnd_cutout_length / 2)
        xmax = self._gnd_cutout_length / 2
        ymin = -(self._gnd_cutout_width / 2)
        ymax = self._gnd_cutout_width / 2
        zpos = self._gnd_elevation()

        center_freq = self.pcb.sim.center_frequency()
        cutout_prop = self.pcb.sim.csx.AddMaterial(
            self._cutout_name(),
            epsilon=self.pcb.pcb_prop.substrate.epsr_at_freq(center_freq),
            kappa=self.pcb.pcb_prop.substrate.kappa_at_freq(center_freq),
        )
        _set_box(
            prop=cutout_prop,
            start=[xmin, ymin, zpos],
            stop=[xmax, ymax, zpos],
            position=Coordinate3(self.position.x, self.position.y, 0),
            transform=self.transform,
            priority=priorities["keepout"],
        )

    def _construct_taper(self) -> None:
        """
        """
        if self._taper is None:
            return

        pos1 = Coordinate2(
            self.position.x
            - (self.dimensions.length / 2)
            - (self._pad_length / 2)
            - (self._taper.length / 2),
            self.position.y,
        )
        pos2 = Coordinate2(
            self.position.x
            + (self.dimensions.length / 2)
            + (self._pad_length / 2)
            + (self._taper.length / 2),
            self.position.y,
        )

        tr = CSTransform()
        tr.AddTransform("RotateAxis", "z", 180)
        self._taper.construct(pos1)
        self._taper.construct(pos2, tr)

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

        CoaxPort(
            sim=self.sim,
            number=self._port_number,
            start=self._start(),
            stop=self._stop(),
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
        center_freq = self.sim.center_frequency()
        dielectric_prop = self.sim.csx.AddMaterial(
            self._dielectric_name(),
            epsilon=self._dielectric.epsr_at_freq(center_freq),
            kappa=self._dielectric.kappa_at_freq(center_freq),
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
        pos = [0, 0, 0]
        pos[prop_axis] = -self._length / 2
        if self._transform is not None:
            pos = self._transform.Transform(pos)
        return Coordinate3(pos[0], pos[1], pos[2])

    def _stop(self) -> Coordinate3:
        """
        """
        prop_axis = self._propagation_axis.intval()
        pos = [0, 0, 0]
        pos[prop_axis] = self._length / 2
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
