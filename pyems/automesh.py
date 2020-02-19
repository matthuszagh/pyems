from typing import List
from bisect import bisect_left, insort_left
import numpy as np
from pyems.utilities import float_cmp
from CSXCAD.CSXCAD import ContinuousStructure


class Mesh:
    """
    Automatic mesh generation for OpenEMS CSX structures.  Probes
    should always be defined after mesh generation.  Additionally,
    voltage probes should be placed on a mesh line and current probes
    should be placed midway between two adjacent mesh lines.
    """

    def __init__(
        self,
        csx: ContinuousStructure,
        lmin: float,
        mres=1 / 20,
        sres=1 / 10,
        smooth: List[float] = [1.5, 1.5, 1.5],
        min_lines: int = 9,
        expand_bounds: List[float] = [20, 20, 20, 20, 20, 20],
        simulation_bounds: List[float] = None,
    ):
        """
        :param csx: the CSXCAD structure (return value of
            CSXCAD.ContinuousStructure()).
        :param lmin: the minimum wavelength associated with the
            expected frequency.
        :param mres: the metal resolution, specified as a factor of
            lmin.
        :param sres: the substrate resolution, specified as a factor
            of lmin.
        :param smooth: the factor by which adjacent cells are allowed
            to differ in size.  This should be a list of 3 factors,
            one for each dimension.  This is useful if, for instance,
            you want the mesh to be smoother in the direction of
            signal propagation.
        :param min_lines: the minimum number of mesh lines for a
            primitive's dimensional length, unless the length of that
            primitive's dimension is precisely 0.
        :param expand bounds: list of 6 elements corresponding to
            [xmin, xmax, ymin, ymax, zmin, zmax] each element gives
            the number of cells to add to the mesh at that boundary.
            The cell size is determined by sres and the actual number
            of cells added may be more (or possibly) less than what is
            specified here due to the thirds rule and smoothing.  This
            essentially defines the simulation box.  It's anticipated
            that the user will only define physical structures (e.g.
            metal layers, substrate, etc.) and will use this to set
            the simulation box.
        :param simulation_bounds: If set this will enforce a strict
            total mesh size and expand_bounds will be ignored.  An
            error will trigger if the internal CSX structures require
            a larger mesh than the one specified here.
        """
        self.csx = csx
        self.lmin = lmin
        self.mres = mres * self.lmin
        self.sres = sres * self.lmin
        self.smooth = smooth
        # mesh lines are added at both boundaries, which gives us an
        # extra mesh line
        self.min_lines = min_lines - 1
        self.expand_bounds = expand_bounds
        self.simulation_bounds = simulation_bounds
        # Sort primitives by decreasing priority.
        self.prims = self.csx.GetAllPrimitives()
        # Keep track of mesh regions already applied. This is an array
        # of 3 elements. The 1st element is a list of ranges in the
        # x-dimension that have already been meshed. The 2nd
        # corresponds to the y-dimension and the 3rd corresponds to
        # the z-dimension.
        self.ranges_meshed = [[], [], []]
        # Keep a list of all metal boundaries. No mesh line is allowed to
        # lie on a boundary, and to the extent possible, should obey the
        # thirds rule about that boundary. Zero-dimension metal structures
        # are not added to this list. The 1st item is the x-dimension
        # list, the 2nd is the y-dimension list and the 3rd is the
        # z-dimension list.
        self.metal_bounds = [[], [], []]
        # Mesh lines that cannot be moved. These are mesh lines that
        # lie directly on a zero-dimension primitive.
        self.const_meshes = [[], [], []]
        # Keep track of the smallest valid resolution value. This
        # allows us to later remove all adjacent mesh lines separated
        # by less than this value.
        self.smallest_res = self.mres
        # The generated mesh.
        self.mesh = self.csx.GetGrid()
        # Set the lines first and draw them last since the API doesn't
        # appear to expose a way to remove individual lines.
        self.mesh_lines = [[], [], []]

    def generate_mesh(self, enforce_thirds=True, smooth=True):
        """
        Start by assuming only two different mesh resolutions: metal
        and substrate/air.  This simplifies the 2/3 rule, where the
        distance to the mesh is 1/3 on the metal side and 2/3 on the
        other side.

        Nonmetal primitives use a mesh resolution of lmin/10 and metal
        primitives use a mesh resolution of lmin/20.  There are two
        exceptions to this: (1) if a dimension of a meterial has
        length 0 (e.g. a planar metal sheet) a single mesh line is
        placed exactly on that line, and (2) a nonzero length material
        must have a minimum of 10 mesh lines .  The 2nd exception will
        not violate the third's rule or smoothness (that adjacent mesh
        lines not differ in separation by more than a factor of 1.5).

        :param enforce_thirds: Enforce thirds rule for metal
            boundaries.  This should always be enabled unless you want
            to debug the mesh generation.
        :param smooth: Smooth mesh lines so that adjacent separations
            do not differ by more than the smoothness factor.  This
            should always be enabled unless you want to debug the mesh
            generation.
        """
        # add metal mesh
        for prim in self.prims:
            if (
                self._type_str(prim) == "Metal"
                or self._type_str(prim) == "ConductingSheet"
            ):
                bounds = self._get_prim_bounds(prim)
                for i in range(3):
                    self._gen_mesh_in_bounds(
                        bounds[i][0], bounds[i][1], self.mres, i, metal=True
                    )

        # add substrate mesh
        for prim in self.prims:
            if self._type_str(prim) == "Material":
                bounds = self._get_prim_bounds(prim)
                for i in range(3):
                    self._gen_mesh_in_bounds(
                        bounds[i][0], bounds[i][1], self.sres, i, metal=False
                    )

        # add simulation box mesh
        if self.simulation_bounds is not None:
            self._check_simulation_bounds_valid()
            for i in range(3):
                self._gen_mesh_in_bounds(
                    self.simulation_bounds[2 * i],
                    self.simulation_bounds[2 * i + 1],
                    self.sres,
                    i,
                    metal=False,
                )
        else:
            for i in range(3):
                self._gen_mesh_in_bounds(
                    self.mesh_lines[i][0]
                    - (self.sres * self.expand_bounds[2 * i]),
                    self.mesh_lines[i][-1]
                    + (self.sres * self.expand_bounds[2 * i + 1]),
                    self.sres,
                    i,
                    metal=False,
                )

        # remove unintended, tightly spaced meshes
        for dim in range(3):
            self._remove_tight_mesh_lines(dim)

        # enforce thirds rule
        if enforce_thirds:
            for dim in range(3):
                self._enforce_thirds(dim)

        # smooth mesh
        if smooth:
            for dim in range(3):
                self._smooth_mesh_lines(dim)

        # set calculated mesh lines
        self._set_mesh_from_lines()

        self._emit_warning()

    def add_line(self, dim: int, pos: float, smooth: bool = True) -> None:
        """
        Add a mesh line.  The mesh line will be fixed.  I.e. smoothing
        / thirds rule cannot shift this line.  This shouldn't
        generally be necessary, and using it might be a sign that
        you're generating the mesh incorrectly.

        :param dim: Dimension to which to add the line.
        :param pos: Position of the new mesh line.
        :param smooth: Resmooth mesh lines after adding.
        """
        insort_left(self.mesh_lines[dim], pos)
        insort_left(self.const_meshes[dim], pos)
        if smooth:
            self._smooth_mesh_lines(dim)
        self._set_mesh_from_lines()

    def nearest_mesh_line(self, dim: int, pos: float) -> (int, float):
        """
        Find the nearest mesh line to a desired position for a given
        dimension.

        :param dim: 0, 1, or 2 for x, y, z.
        :param pos: desired position.

        :returns: (index, position) where index is the array index and
                  position is the actual dimension value.
        """
        lines = self.mesh_lines[dim]
        bisect_pos = bisect_left(self.mesh_lines[dim], pos)
        if bisect_pos == 0:
            return (0, lines[0])
        elif bisect_pos == len(lines):
            return (bisect_pos - 1, lines[bisect_pos - 1])
        else:
            lower = lines[bisect_pos - 1]
            upper = lines[bisect_pos]
            if pos - lower < upper - pos:
                return (bisect_pos - 1, lower)
            else:
                return (bisect_pos, upper)

    def _check_simulation_bounds_valid(self) -> None:
        """
        Ensure the strict bounds are at least as large as the mesh
        bounds formed by other structures.
        """
        for dim in [0, 2, 4]:
            idx = int(dim / 2)
            if idx >= len(self.ranges_meshed):
                continue
            elif 0 >= len(self.ranges_meshed[idx]):
                continue
            if self.simulation_bounds[dim] > self.ranges_meshed[idx][0]:
                raise ValueError("invalid strict bounds chosen.")
        for dim in [1, 3, 5]:
            idx = int(dim / 2)
            if idx >= len(self.ranges_meshed):
                continue
            elif 0 >= len(self.ranges_meshed[idx]):
                continue
            if self.simulation_bounds[dim] < self.ranges_meshed[idx][-1]:
                raise ValueError("invalid strict bounds chosen.")

    # TODO should ensure that inserted mesh lines are not at metal boundaries
    def _enforce_thirds(self, dim):
        """
        Replace mesh lines at metal boundaries with a mesh line
        1/3*res inside the metal boundary and 2/3*res outside.

        :param dim: Dimension for which thirds rule should be
            enforced.
        """
        for i, pos in enumerate(self.mesh_lines[dim]):
            if (
                pos in self.metal_bounds[dim]
                and pos not in self.const_meshes[dim]
            ):
                # don't do anything at the boundary
                if i == 0 or i == len(self.mesh_lines[dim]) - 1:
                    continue
                # # at lower boundary
                # if i == 0:
                #     del self.mesh_lines[dim][i]
                #     insort_left(self.mesh_lines[dim], pos + (self.mres / 3))
                #     self._enforce_thirds(dim)
                # # at upper boundary
                # elif i == len(self.mesh_lines[dim]) - 1:
                #     del self.mesh_lines[dim][i]
                #     insort_left(self.mesh_lines[dim], pos - (self.mres / 3))
                #     self._enforce_thirds(dim)
                else:
                    spacing_left = pos - self.mesh_lines[dim][i - 1]
                    spacing_right = self.mesh_lines[dim][i + 1] - pos
                    del self.mesh_lines[dim][i]
                    # metal-metal boundary
                    if (
                        abs(spacing_left - spacing_right)
                        < self.smallest_res / 10
                    ):
                        new_low = pos - (spacing_left / 2)
                        new_high = pos + (spacing_left / 2)
                    # don't need to add tolerance for float comparison
                    # since metal-metal boundary check already did
                    # that
                    elif spacing_left < spacing_right:
                        new_low = pos - (spacing_left / 3)
                        new_high = pos + (2 * spacing_left / 3)
                    else:
                        new_low = pos - (2 * spacing_right / 3)
                        new_high = pos + (spacing_right / 3)

                    insort_left(self.mesh_lines[dim], new_low)
                    insort_left(self.mesh_lines[dim], new_high)
                    self._enforce_thirds(dim)

    def _remove_tight_mesh_lines(self, dim):
        """
        Remove adjacent mesh lines for dimension @dim with spacing
        less than the smallest valid resolution.

        :param dim: Dimension in which to remove tightly spaced
            meshes.
        """
        last_pos = self.mesh_lines[dim][0]
        for i, pos in enumerate(self.mesh_lines[dim]):
            if i == 0:
                continue
            # we can freely delete duplicates
            if pos == last_pos:
                del self.mesh_lines[dim][i]
                self._remove_tight_mesh_lines(dim)
            # we have to check whether these are zero-dimension
            # structures before deleting them.
            elif (
                pos - last_pos < self.smallest_res
                and abs(pos - last_pos - self.smallest_res)
                > self.smallest_res / 10
                and (
                    pos not in self.const_meshes[dim]
                    or last_pos not in self.const_meshes[dim]
                )
            ):
                if last_pos not in self.const_meshes[dim]:
                    del self.mesh_lines[dim][i - 1]
                else:
                    del self.mesh_lines[dim][i]
                self._remove_tight_mesh_lines(dim)
            else:
                last_pos = pos

    def _set_mesh_from_lines(self):
        """
        Generates the actual CSX mesh structure from mesh_lines.  This
        clears any preexisting mesh.
        """
        for i in range(3):
            self.mesh.ClearLines(i)
        for dim in range(3):
            for line in self.mesh_lines[dim]:
                self.mesh.AddLine(dim, line)

    def _get_mesh(self):
        return self.mesh

    def _type_str(self, prim):
        return prim.GetProperty().GetTypeString()

    def _get_prim_bounds(self, prim):
        orig_bounds = prim.GetBoundBox()
        bounds = [[None, None], [None, None], [None, None]]
        for i in range(3):
            upper = max(orig_bounds[0][i], orig_bounds[1][i])
            lower = min(orig_bounds[0][i], orig_bounds[1][i])
            bounds[i] = [lower, upper]
        return bounds

    def _mesh_res_in_bounds(self, lower, upper, dim):
        """
        Get the mesh resolution in the supplied boundary.

        :param lower: lower boundary.
        :param upper: upper boundary.
        :param dim: Dimension. 0, 1, or 2 for x, y, or z.
        """
        lower_idx = bisect_left(self.mesh_lines[dim], lower)
        upper_idx = min(
            bisect_left(self.mesh_lines[dim], upper) + 1,
            len(self.mesh_lines[dim]),
        )
        spacing = []
        last_pos = self.mesh_lines[dim][lower_idx]
        if lower_idx + 1 == upper_idx:
            return (
                self.mesh_lines[dim][upper_idx]
                - self.mesh_lines[dim][lower_idx]
            )
        else:
            for idx in range(lower_idx + 1, upper_idx):
                spacing.append(self.mesh_lines[dim][idx] - last_pos)
                last_pos = self.mesh_lines[dim][idx]
            return sum(spacing) / len(spacing)

    def _split_bounds(self, lower, upper, dim):
        """
        Split bounds delimited by [lower, upper] into regions where mesh
        already exists and regions where it doesn't yet exist.

        Returns a list of 2 items. The 1st item is a list of bounds
        where the new mesh boundaries are outside the existing
        mesh. The 2nd item is a list of bounds where the new mesh
        overlaps the existing mesh.
        """
        outin_ranges = [[], []]
        if len(self.ranges_meshed[dim]) == 0:
            outin_ranges[0].append([lower, upper])
            return outin_ranges

        for [lower_mesh, upper_mesh] in self.ranges_meshed[dim]:
            if upper <= lower_mesh:
                outin_ranges[0].append([lower, upper])
                # since meshed ranges are sorted, we can ignore the rest.
                return outin_ranges
            elif lower >= upper_mesh:
                continue
            elif lower < lower_mesh:
                outin_ranges[0].append([lower, lower_mesh])
                if upper > upper_mesh:
                    outin_ranges[1].append([lower_mesh, upper_mesh])
                    lower = upper_mesh
                    continue
                else:
                    outin_ranges[1].append([lower_mesh, upper])
                    return outin_ranges
            else:
                outin_ranges[1].append([lower, min(upper, upper_mesh)])
                if upper > upper_mesh:
                    lower = upper_mesh
                    continue
                else:
                    return outin_ranges
        if lower < upper:
            outin_ranges[0].append([lower, upper])

        return outin_ranges

    def _clear_mesh_in_bounds(self, lower, upper, dim):
        """
        """
        for elt in self.mesh_lines[dim]:
            if elt >= lower and elt <= upper:
                self.mesh_lines[dim].remove(elt)

    def _range_union(self, ranges, start_idx=0):
        """
        """
        ranges = sorted(ranges)
        if len(ranges[start_idx:]) <= 1:
            return ranges

        if ranges[start_idx][1] >= ranges[start_idx + 1][0]:
            ranges.append([ranges[start_idx][0], ranges[start_idx + 1][1]])
            del ranges[start_idx : start_idx + 2]
            return self._range_union(ranges[start_idx:])
        else:
            return self._range_union(ranges[start_idx + 1 :])

    def _consolidate_meshed_ranges(self, dim):
        """
        Order meshed ranges and consolidate contiguous ranges.
        """
        self.ranges_meshed[dim] = self._range_union(self.ranges_meshed[dim])

    def _update_ranges(self, lower, upper, dim):
        """
        :param dim: is the dimension: 0, 1, 2 for x, y, or z.
        """
        self.ranges_meshed[dim].append([lower, upper])
        self._consolidate_meshed_ranges(dim)

    def _smooth_mesh_lines(self, dim):
        """
        Ensure adjacent mesh line separations differ by less than the
        smoothness factor.

        If there's enough room between mesh lines, this function will
        recursively add mesh lines corresponding to the largest
        possible separation in line with self.smooth.  When there's
        not enough room, it moves the position of existing lines to be
        in line with smooth.

        TODO should be refactored since logic in if branches are
        basically identical, but switched.

        :param dim: Dimension where mesh should be smoothed.
        """
        for i, pos in enumerate(self.mesh_lines[dim]):
            if i == 0 or i == len(self.mesh_lines[dim]) - 1:
                continue
            left_spacing = pos - self.mesh_lines[dim][i - 1]
            right_spacing = self.mesh_lines[dim][i + 1] - pos
            if (
                left_spacing > (self.smooth[dim] * right_spacing)
                and left_spacing - (self.smooth[dim] * right_spacing)
                > self.smallest_res / 10
            ):
                ratio = left_spacing / right_spacing
                if i == len(self.mesh_lines[dim]) - 2:
                    # if there's no further mesh spacings to worry
                    # about, this ensures we'll only move the mesh
                    # line when that will satisfy smoothness
                    outer_spacing = right_spacing + (
                        1 / (self.smooth[dim] * (self.smooth[dim] + 1))
                    )
                else:
                    outer_spacing = self.mesh_lines[dim][i + 2] - (
                        pos + right_spacing
                    )
                # if this condition satisfied, then we can move the
                # current mesh line without violating smoothness
                # elsewhere. To see how I got this condition, imagine
                # spacings are given by a, b and c in order. Spacings
                # on either side of current position are a and b. s
                # gives smooth factor. Move pos by dx to have a and b
                # satisfy s. It's currently above, so move by just
                # enough to satisfy.
                #
                # (a-dx)/(b+dx) = s
                #
                # simultaneously,
                #
                # (b+dx)/c <= s
                #
                # we need the max a where this works. This occurs at
                #
                # (b+dx)/c = s
                #
                # find a. sagemath tells you that
                #
                # a = cs^2 + cs - b
                if (
                    left_spacing
                    <= outer_spacing
                    * self.smooth[dim]
                    * (self.smooth[dim] + 1)
                    - right_spacing
                ):
                    # adjustment to make left_spacing = smooth * right_spacing
                    adj = (
                        left_spacing - (self.smooth[dim] * right_spacing)
                    ) / (self.smooth[dim] + 1)
                    # TODO need to ensure new mesh line doesn't fall
                    # on metal boundary or violate thirds.
                    if pos not in self.const_meshes[dim]:
                        del self.mesh_lines[dim][i]
                        insort_left(self.mesh_lines[dim], pos - adj)
                    else:
                        insort_left(
                            self.mesh_lines[dim], pos - (left_spacing / 2)
                        )
                # mesh separation is too small to add smooth *
                # spacing, so instead add it halfway
                elif ratio <= self.smooth[dim] * (self.smooth[dim] + 1):
                    insort_left(self.mesh_lines[dim], pos - (left_spacing / 2))
                else:
                    insort_left(
                        self.mesh_lines[dim],
                        pos - (self.smooth[dim] * right_spacing),
                    )
                self._smooth_mesh_lines(dim)
            elif (
                right_spacing > self.smooth[dim] * left_spacing
                and right_spacing - (self.smooth[dim] * left_spacing)
                > self.smallest_res / 10
            ):
                ratio = right_spacing / left_spacing
                if i == 1:
                    outer_spacing = left_spacing + (
                        1 / (self.smooth[dim] * (self.smooth[dim] + 1))
                    )
                else:
                    outer_spacing = (
                        pos - left_spacing - self.mesh_lines[dim][i - 2]
                    )
                if (
                    right_spacing
                    <= outer_spacing
                    * self.smooth[dim]
                    * (self.smooth[dim] + 1)
                    - left_spacing
                ):
                    adj = (
                        right_spacing - (self.smooth[dim] * left_spacing)
                    ) / (self.smooth[dim] + 1)
                    # TODO need to ensure new mesh line doesn't fall
                    # on metal boundary or violate thirds.
                    if pos not in self.const_meshes[dim]:
                        del self.mesh_lines[dim][i]
                        insort_left(self.mesh_lines[dim], pos + adj)
                    else:
                        insort_left(
                            self.mesh_lines[dim], pos + (right_spacing / 2)
                        )
                elif ratio <= self.smooth[dim] * (self.smooth[dim] + 1):
                    insort_left(
                        self.mesh_lines[dim], pos + (right_spacing / 2)
                    )
                else:
                    insort_left(
                        self.mesh_lines[dim],
                        pos + (self.smooth[dim] * left_spacing),
                    )
                self._smooth_mesh_lines(dim)

    def _nearest_divisible_res(self, lower, upper, res):
        """
        Return the nearest resolution to @res that evenly subdivides the
        interval [@lower, @upper].

        This is important because it helps prevent adjacent lines from
        bunching up and unnecessarily increasing the simulation time.
        """
        num_divisions = np.round((upper - lower) / res)
        num_divisions = max(num_divisions, 1)
        return (upper - lower) / num_divisions

    def _gen_mesh_in_bounds(self, lower, upper, res, dim, metal=False):
        """
        Add mesh lines within the provided dimensional boundaries.

        :param lower: Lower dimensional boundary.
        :param upper: Upper dimensional boundary.
        :param res: Desired mesh resolution within the provided
            boundary.
        :param dim: Dimension.  0, 1, or 2.
        :param metal: Set to True if this boundary corresponds to a
            metal structure.
        """
        if lower == upper:
            insort_left(self.mesh_lines[dim], lower)
            insort_left(self.const_meshes[dim], lower)
        else:
            [outer_bounds, inner_bounds] = self._split_bounds(
                lower, upper, dim
            )
            for obound in outer_bounds:
                if obound[1] - obound[0] < self.min_lines * res:
                    res = (obound[1] - obound[0]) / self.min_lines
                else:
                    res = self._nearest_divisible_res(
                        obound[0], obound[1], res
                    )
                self.smallest_res = min(self.smallest_res, res)
                j = obound[0]
                while j <= obound[1]:
                    insort_left(self.mesh_lines[dim], j)
                    j += res
                    # depending on float rounding errors we can miss
                    # the last mesh line
                    if float_cmp(j, obound[1], self.smallest_res / 20):
                        j = obound[1]
                self._update_ranges(obound[0], obound[1], dim)
                if metal:
                    insort_left(self.metal_bounds[dim], obound[0])
                    insort_left(self.metal_bounds[dim], obound[1])
            for ibound in inner_bounds:
                if upper - lower < self.min_lines * res:
                    res = (upper - lower) / self.min_lines
                else:
                    res = self._nearest_divisible_res(
                        ibound[0], ibound[1], res
                    )
                self.smallest_res = min(self.smallest_res, res)
                # only redo the mesh if the desired one is finer than
                # the existing one
                cur_mesh_res = self._mesh_res_in_bounds(
                    ibound[0], ibound[1], dim
                )
                if cur_mesh_res > res and abs(cur_mesh_res - res) > res / 10:
                    self._clear_mesh_in_bounds(ibound[0], ibound[1], dim)
                    j = ibound[0]
                    while j <= ibound[1]:
                        insort_left(self.mesh_lines[dim], j)
                        j += res
                        if float_cmp(j, ibound[1], self.smallest_res / 20):
                            j = ibound[1]
                    self._update_ranges(ibound[0], ibound[1], dim)
                if metal:
                    insort_left(self.metal_bounds[dim], ibound[0])
                    insort_left(self.metal_bounds[dim], ibound[1])

    def _emit_warning(self):
        """
        Display a warning for a possibly bad mesh.
        """
        if self.smallest_res < self._max_dim() / 1000:
            print(
                "The generated mesh appears to be very fine. "
                "Have you used consistent length units?"
            )

    def _max_dim(self):
        """
        Compute the max dimension of the mesh.
        """
        dims = [lines[-1] - lines[0] for lines in self.mesh_lines]
        return np.amax(dims)
