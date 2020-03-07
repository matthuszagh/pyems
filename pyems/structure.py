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
from pyems.coordinate import Coordinate2, Coordinate3, Axis, Box2, Box3
from pyems.simulation_beta import Simulation
from pyems.port import MicrostripPort


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


class Structure(ABC):
    """
    Base class for all other structures.  Provides the capability to
    position and transform any structure.
    """

    unique_index = 0

    def __init__(
        self, sim: Simulation, transform: CSTransform = None,
    ):
        """
        :param sim: The Simulation to which this object will be
            added.
        :param transform: A CSTransform object (incorporating possibly
            multiple transformations) to apply to the structure.  This
            will be applied before the structures position is shifted
            to the position set by position.
        """
        self._sim = sim
        self._transform = transform

    @abstractmethod
    def construct(self, position) -> None:
        """
        Build the structure.
        """
        pass

    @property
    def sim(self) -> Simulation:
        """
        """
        return self._sim

    @property
    def transform(self) -> CSTransform:
        """
        """
        return self._transform

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
        substrate_priority: int = 0,
        copper_priority: int = 1,
        transform: CSTransform = None,
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
        self._substrate_priority = substrate_priority
        self._copper_priority = copper_priority
        super().__init__(sim, transform)

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
    def copper_priority(self) -> int:
        """
        """
        return self._copper_priority

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

    def construct(self, position: Coordinate3) -> None:
        """
        """
        self._position = position
        zpos = self.position.z
        for layer in self.layers:
            zpos = self._construct_layer(zpos, layer)

    def _is_copper_layer(self, layer_index: int) -> bool:
        """
        """
        return layer_index % 2 == 0

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
        xbounds = self._get_x_bounds()
        ybounds = self._get_y_bounds()
        box = layer_prop.AddBox(
            priority=self._copper_priority,
            start=[xbounds[0], ybounds[0], zpos],
            stop=[xbounds[1], ybounds[1], zpos],
        )
        apply_transform(box, self.transform)

        return zpos

    def _construct_substrate_layer(
        self, zpos: float, layer_index: int
    ) -> None:
        """
        """
        layer_prop = self.sim.csx.AddMaterial(
            self._layer_name(layer_index),
            epsilon=self.pcb_prop.epsr_at_freq(self.sim.center_frequency()),
            kappa=self.pcb_prop.kappa_at_freq(self.sim.center_frequency()),
        )
        xbounds = self._get_x_bounds()
        ybounds = self._get_y_bounds()
        zbounds = (
            zpos
            - self.pcb_prop.copper_layer_dist(
                self._copper_index(layer_index + 1),
                unit=self.sim.unit,
                ref_layer=self._copper_index(layer_index - 1),
            ),
            zpos,
        )
        box = layer_prop.AddBox(
            priority=self._substrate_priority,
            start=[xbounds[0], ybounds[0], zbounds[0]],
            stop=[xbounds[1], ybounds[1], zbounds[1]],
        )
        apply_transform(box, self.transform)

        return zbounds[0]

    def _get_x_bounds(self) -> Tuple[float, float]:
        """
        """
        return (
            self.position.x - (self._length / 2),
            self.position.x + (self._length / 2),
        )

    def _get_y_bounds(self) -> Tuple[float, float]:
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

    It doesn't make sense to apply a custom transform to this.
    However, any transform applied to the PCB will automatically be
    applied here.
    """

    unique_index = 0

    def __init__(
        self,
        pcb: PCB,
        position: Coordinate2,
        drill: float,
        annular_ring: float,
        antipad: float,
        metal_priority: int,
        antipad_priority: int,
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
        :param metal_priority: CSXCAD priority for annular ring and
            via plating.
        :param antipad_priority: CSXCAD priority for antipad.  Must be
            set lower than metal_priority, or the via will be ignored.
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
        self._metal_priority = metal_priority
        self._antipad_priority = antipad_priority
        if self._antipad_priority >= self._metal_priority:
            raise RuntimeWarning(
                "Via antipad priority was not set lower than metal "
                "priority. This will cause the via metal to be ignored "
                "on these layers. Is this intentional?"
            )
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
            priority=self._metal_priority,
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
                priority=self._air_priority(),
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
                priority=self._metal_priority,
            )
            apply_transform(pad_prim, self.pcb.transform)

    def _construct_antipads(self) -> None:
        """
        """
        for layer in self._noconnect_layers:
            zpos = self.pcb.copper_layer_elevation(layer)
            antipad_prop = self.pcb.sim.csx.AddMaterial(
                self._antipad_name(layer),
                epsilon=self.pcb.pcb_prop.epsr_at_freq(
                    self.pcb.sim.center_frequency()
                ),
                kappa=self.pcb.pcb_prop.kappa_at_freq(
                    self.pcb.sim.center_frequency()
                ),
            )
            antipad_prim = construct_circle(
                prop=antipad_prop,
                center=Coordinate3(self.position.x, self.position.y, zpos),
                radius=self._antipad_radius(),
                normal=Axis("z"),
                priority=self._antipad_priority,
            )
            apply_transform(antipad_prim, self.pcb.transform)

    def _air_priority(self) -> int:
        """
        """
        return self._metal_priority + 1

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
    act as a port for later analysis.
    """

    unique_index = 0

    def __init__(
        self,
        pcb: PCB,
        box: Box2,
        trace_priority: int,
        gap_priority: int,
        trace_layer: int = 0,
        gnd_layer: int = 1,
        gnd_gap: float = None,
        via_gap: float = None,
        via: Via = None,
        via_spacing: float = None,
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
        :param box: 2D box (x-y plane) of the microstrip trace.  The
            z-coordinate is determined by the PCB layer.  When used as
            a port, the order matters since the probes and signal
            excitation will be setup to travel from the minimum x to
            maximum x.
        :param trace_priority: CSXCAD priority for trace.
        :param gap_priority: CSXCAD priority for gap.  This should be
            lower than the trace priority but higher than the coplanar
            ground priority.
        :param trace_layer: PCB layer of the signal trace.  Uses
            copper layer index values.
        :param gnd_layer: PCB layer of the ground plane.  Uses copper
            layer index values.
        :param gnd_gap: Gap distance between trace edge and
            surrounding coplanar ground plane.  If left as the default
            value of None, the entire coplanar ground plane will be
            removed.
        :param via_gap: Gap distance between the start of the coplanar
            ground plane and the surrounding via fence.  If set to
            None, a via fence is not used.
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
        :param transform: CSTransform to apply to microstrip.  The
            microstrip will automatically inherit any transformation
            applied to the PCB.  This transform will be applied after
            that one.
        """
        self._pcb = pcb
        self._box = box
        self._trace_priority = trace_priority
        self._gap_priority = gap_priority
        if self._gap_priority >= self._trace_priority:
            raise RuntimeWarning(
                "Microstrip gap priority was not set less than trace "
                "priority. This will ignore the trace. Is that intended?"
            )
        self._trace_layer = trace_layer
        self._gnd_layer = gnd_layer
        if gnd_gap is None:
            self._gnd_gap = self.pcb.width / 2 - (self.trace_width() / 2)
            self._via_gap = None
        else:
            self._gnd_gap = gnd_gap
            self._via_gap = via_gap
        self._via = via
        self._via_spacing = via_spacing
        self._port_number = port_number
        self._excite = excite
        self._feed_impedance = feed_impedance
        self._feed_shift = feed_shift
        self._ref_impedance = ref_impedance
        self._measurement_shift = measurement_shift
        self._transform = append_transform(self.pcb.transform, transform)
        self._index = None

        if self.box is not None:
            self.construct(self.box)

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
    def box(self) -> Box2:
        """
        """
        return self._box

    def trace_width(self) -> float:
        """
        """
        return self.box.max_corner.y - self.box.min_corner.y

    def construct(self, box: Box2) -> None:
        """
        """
        self._box = box
        self._index = self._get_inc_ctr()
        if self.port_number is not None:
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
            number=self.port_number,
            thickness=self.pcb.pcb_prop.copper_thickness(self._trace_layer),
            conductivity=self.pcb.pcb_prop.metal_conductivity(),
            excite=self._excite,
            feed_impedance=self._feed_impedance,
            feed_shift=self._feed_shift,
            ref_impedance=self._ref_impedance,
            measurement_shift=self._measurement_shift,
            transform=self.transform,
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
        start = self.box.start()
        start.append(trace_z)
        stop = self.box.stop()
        stop.append(trace_z)
        box = trace_prop.AddBox(
            priority=self._trace_priority, start=start, stop=stop
        )
        apply_transform(box, self.transform)

    def _construct_gap(self) -> None:
        """
        """
        freq = self.pcb.sim.center_frequency()
        gap_prop = self.pcb.sim.csx.AddMaterial(
            self._gap_name(),
            epsilon=self.pcb.pcb_prop.epsr_at_freq(freq),
            kappa=self.pcb.pcb_prop.kappa_at_freq(freq),
        )
        trace_z = self._trace_z()
        start = self.box.start()
        start.append(trace_z)
        stop = self.box.stop()
        stop.append(trace_z)
        start[1] -= self._gnd_gap
        stop[1] += self._gnd_gap
        gap_box = gap_prop.AddBox(
            priority=self._gap_priority, start=start, stop=stop,
        )
        apply_transform(gap_box, self.transform)

    def _construct_via_fence(self) -> None:
        """
        """
        if self._via_gap is None:
            return

        ylow, yhigh = self._via_ypos()
        if self._via is None:
            self._construct_via_wall(ylow)
            self._construct_via_wall(yhigh)
        else:
            # TODO ignores extra transform?
            xmax = self.box.max_corner.x
            bound_spacing = self._via_spacing / 2
            xpos = self.box.min_corner.x + bound_spacing
            via_rad = self._via.pad_radius()
            while xpos + via_rad < xmax:
                pos_low = Coordinate2(xpos, ylow)
                pos_high = Coordinate2(xpos, yhigh)
                self._via.construct(pos_low)
                self._via.construct(pos_high)
                xpos += self._via_spacing

    def _construct_via_wall(self, ypos: float) -> None:
        """
        """
        via_prop = self.pcb.sim.csx.AddConductingSheet(
            self._via_wall_name(),
            conductivity=self.pcb.pcb_prop.metal_conductivity(),
            thickness=self.pcb.pcb_prop.copper_thickness(0),
        )
        start = self.box.start()
        start.append(
            self.pcb.copper_layer_elevation(self.pcb.copper_layers()[-1])
        )
        start[1] = ypos
        stop = self.box.stop()
        stop.append(
            self.pcb.copper_layer_elevation(self.pcb.copper_layers()[0])
        )
        stop[1] = ypos
        box = via_prop.AddBox(
            priority=self.pcb.copper_priority, start=start, stop=stop
        )
        apply_transform(box, self.transform)

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
        box = Box3(
            Coordinate3(
                self.box.min_corner.x, self.box.min_corner.y, self._gnd_z()
            ),
            Coordinate3(
                self.box.max_corner.x, self.box.max_corner.y, self._trace_z()
            ),
        )
        return box

    def _via_ypos(self) -> Tuple[float, float]:
        """
        """
        return (
            self.box.min_corner.y - self._gnd_gap - self._via_gap,
            self.box.max_corner.y + self._gnd_gap + self._via_gap,
        )

    def _trace_z(self) -> float:
        """
        """
        return self.pcb.copper_layer_elevation(self._trace_layer)

    def _gnd_z(self) -> float:
        """
        """
        return self.pcb.copper_layer_elevation(self._gnd_layer)
