"""Classes relating to planestress geometry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import shapely as shapely
import shapely.affinity as affinity

from planestress.post.plotting import plotting_context
from planestress.pre.material import DEFAULT_MATERIAL, Material
from planestress.pre.mesh import Mesh


if TYPE_CHECKING:
    import matplotlib.axes

    from planestress.pre.load_case import LoadCase
    from planestress.pre.mesh import Field


class Geometry:
    """Class describing a geometric region."""

    def __init__(
        self,
        polygons: shapely.Polygon | shapely.MultiPolygon,
        materials: Material | list[Material] = DEFAULT_MATERIAL,
        embedded_geometry: list[Point | Facet] | None = None,
        tol: int = 12,
    ) -> None:
        """Inits the Geometry class.

        .. note::
            Length of ``materials`` must equal the number of ``polygons``, i.e.
            ``len(polygons.geoms)``.

        Args:
            polygons: A :class:`shapely.Polygon` or :class:`shapely.MultiPolygon`
                describing the geometry. A :class:`~shapely.MultiPolygon` comprises of a
                list of :class:`~shapely.Polygon` objects, that can describe a geometry
                with multiple distinct regions.
            materials: A list of :class:`~planestress.pre.Material` objects describing
                the material properties of each ``polygon`` within the geometry. If a
                single :class:`planestress.pre.Material` is supplied, this material is
                applied to all regions. Defaults to ``DEFAULT_MATERIAL``, i.e. a
                material with unit properties and a Poisson's ratio of zero.
            embedded_geometry: List of ``Point`` or ``Facet`` objects to embed into the
                mesh. Can also be added by using the :meth:`embed_point` or
                :meth:`embed_line` methods. Defaults to ``None``.
            tol: The points in the geometry get rounded to ``tol`` digits. Defaults to
                ``12``.

        Raises:
            ValueError: If the number of ``materials`` does not equal the number of
                ``polygons``.

        Example:
            TODO.
        """
        # convert polygon to multipolygon
        if isinstance(polygons, shapely.Polygon):
            polygons = shapely.MultiPolygon(polygons=[polygons])

        # convert material to list of materials
        if isinstance(materials, Material):
            materials = [materials] * len(polygons.geoms)

        # check materials length
        if len(polygons.geoms) != len(materials):
            raise ValueError(
                f"Length of materials: {len(materials)}, must equal number of polygons:"
                f"{len(polygons.geoms)}."
            )

        # save input data
        self.polygons = polygons
        self.materials = materials
        self.embedded_geometry = [] if embedded_geometry is None else embedded_geometry
        self.tol = tol

        # allocate points, facets, curve loops and surfaces
        self.points: list[Point] = []
        self.facets: list[Facet] = []
        self.curve_loops: list[CurveLoop] = []
        self.surfaces: list[Surface] = []

        # allocate holes (note these are just for plotting purposes - not used by gmsh)
        self.holes: list[Point] = []

        # compile the geometry into points, facets and holes
        self.compile_geometry()

        # TODO - test for overlapping facets

        # allocate mesh
        self.mesh = Mesh()

    def compile_geometry(self) -> None:
        """Creates points, facets and holes from shapely geometry."""
        # note tags in gmsh start at 1
        poly_idx: int = 1
        loop_idx: int = 1

        # loop through each polygon
        for poly in self.polygons.geoms:
            # first create points, facets and holes for each polygon
            poly_points, poly_facets, poly_holes, loop_idx = self.compile_polygon(
                polygon=poly, poly_idx=poly_idx, loop_idx=loop_idx
            )

            # add points to the global list, skipping duplicate points
            for pt in poly_points:
                if pt not in self.points:
                    self.points.append(pt)
                # we have to fix facets that reference the skipped point!
                else:
                    # get the point that we are keeping in the list
                    kept_pt_idx = self.points.index(pt)
                    kept_pt = self.points[kept_pt_idx]

                    # loop through all facets and update removed point reference
                    for fct in poly_facets:
                        fct.update_point(old=pt, new=kept_pt)

            # add facets to the global list
            for fct in poly_facets:
                # conduct checks:
                # if we have a zero length facet, remove from the curve loop
                if fct.zero_length():
                    # get current surface
                    surf = self.surfaces[-1]

                    # loop through curve loops
                    for loop in surf.curve_loops:
                        # if facet in curve loop, remove
                        if fct in loop.facets:
                            loop.facets.remove(fct)
                # if we have a duplicate facet, fix reference to facet in curve loop
                elif fct in self.facets:
                    # get current surface
                    # note duplicate facets only occur between different surfaces
                    # best to get current surface rather than loop through all surfaces
                    surf = self.surfaces[-1]

                    # get the facet that we are keeping in the list
                    kept_fct_idx = self.facets.index(fct)
                    kept_fct = self.facets[kept_fct_idx]

                    # loop through curve loops of current surface and update facet ref
                    for loop in surf.curve_loops:
                        loop.update_facet(old=fct, new=kept_fct)
                # otherwise add new facet
                else:
                    self.facets.append(fct)

            # add holes to list of multipolygon holes
            self.holes.extend(poly_holes)

            # update polygon and loop index
            poly_idx += 1
            loop_idx += 1

        # assign point indexes and poly indexes to points, note tags in gmsh start at 1
        for pt_idx, pt in enumerate(self.points):
            pt.idx = pt_idx + 1

            # loop through surfaces
            for surface in self.surfaces:
                if surface.point_on_surface(point=pt):
                    pt.poly_idxs.append(surface.idx)

        # assign facet indexs, note tags in gmsh start at 1
        for fct_idx, fct in enumerate(self.facets):
            fct.idx = fct_idx + 1

    def compile_polygon(
        self,
        polygon: shapely.Polygon,
        poly_idx: int,
        loop_idx: int,
    ) -> tuple[list[Point], list[Facet], list[Point], int]:
        """Creates points, facets and holes given a ``Polygon``.

        Args:
            polygon: A :class:`~shapely.Polygon` object.
            poly_idx: Polygon index.
            loop_idx: Loop index.

        Returns:
            A list of points, facets, and holes, and the current loop index (``points``,
            ``facets``, ``holes``, ``loop_idx``).
        """
        pt_list: list[Point] = []
        fct_list: list[Facet] = []
        hl_list: list[Point] = []

        # create new surface and append to class list
        surface = Surface(idx=poly_idx)
        self.surfaces.append(surface)

        # construct perimeter points (note shapely doubles up first & last point)
        for coords in list(polygon.exterior.coords[:-1]):
            new_pt = Point(x=coords[0], y=coords[1], tol=self.tol)
            pt_list.append(new_pt)

        # create perimeter facets and add to facet list
        new_facets, curve_loop = self.create_facet_list(
            pt_list=pt_list, loop_idx=loop_idx
        )
        fct_list.extend(new_facets)

        # add curve loop to current surface
        surface.curve_loops.append(curve_loop)

        # construct interior regions
        for interior in polygon.interiors:
            int_pt_list: list[Point] = []
            loop_idx += 1  # update loop index

            # create interior points (note shapely doubles up first & last point)
            for coords in interior.coords[:-1]:
                new_pt = Point(x=coords[0], y=coords[1], tol=self.tol)
                int_pt_list.append(new_pt)

            # add interior points to poly list
            pt_list.extend(int_pt_list)

            # create interior facets and add to facet list
            new_facets, curve_loop = self.create_facet_list(
                pt_list=int_pt_list, loop_idx=loop_idx
            )
            fct_list.extend(new_facets)

            # add curve loop to current surface
            surface.curve_loops.append(curve_loop)

            # create hole point:
            # first convert the list of interior points to a list of tuples
            int_pt_list_tup = [int_pt.to_tuple() for int_pt in int_pt_list]

            # create a polygon of the hole region
            int_poly = shapely.Polygon(int_pt_list_tup)

            # add hole point to the list of hole points
            hl_pt_coords = int_poly.representative_point().coords
            hl_list.append(
                Point(x=hl_pt_coords[0][0], y=hl_pt_coords[0][1], tol=self.tol)
            )

        return pt_list, fct_list, hl_list, loop_idx

    def create_facet_list(
        self,
        pt_list: list[Point],
        loop_idx: int,
    ) -> tuple[list[Facet], CurveLoop]:
        """Creates a closed list of facets from a list of points.

        Args:
            pt_list: List of points.
            loop_idx: Loop index.

        Returns:
            Closed list of facets and curve loop.
        """
        fct_list: list[Facet] = []

        # create new curve loop and append to class list
        curve_loop = CurveLoop(idx=loop_idx)
        self.curve_loops.append(curve_loop)

        # create facets
        for idx, pt in enumerate(pt_list):
            pt1 = pt
            # if we are not at the end of the list
            if idx + 1 != len(pt_list):
                pt2 = pt_list[idx + 1]
            # otherwise loop back to starting point
            else:
                pt2 = pt_list[0]

            # create new facet
            new_facet = Facet(pt1=pt1, pt2=pt2)

            # add facet to facet list and curve loop
            fct_list.append(new_facet)
            curve_loop.facets.append(new_facet)

        return fct_list, curve_loop

    def embed_point(
        self,
        x: float,
        y: float,
        mesh_size: float | None = None,
    ) -> None:
        """Embeds a point into the mesh.

        Args:
            x: ``x`` location of the embedded point.
            y: ``y`` location of the embedded point.
            mesh_size: Optional mesh size at the embedded point. If not provided, takes
                the mesh size of the polygon the point is embedded into. Defaults to
                ``None``.

        Raises:
            ValueError: If the point does not lie within any polygons.

        Warning:
            Ensure that embedded points lie within the meshed region and are not located
            within any holes in the mesh.

            Further, due to floating point precision errors, it is recommended that
            embedded points are also not placed on the edges of polygons.

            Note, not all the above conditions are checked.
        """
        # find polygon that point lies within
        pt = shapely.Point(x, y)

        for idx, poly in enumerate(self.polygons.geoms):
            if shapely.within(pt, poly):
                poly_idx = idx
                break
        else:
            raise ValueError(f"Point ({x}, {y}) does not lie within any polygons.")

        # create point object and assign poly idx
        point = Point(x=x, y=y, tol=self.tol, mesh_size=mesh_size)
        point.poly_idxs = [poly_idx + 1]  # note poly indexes start at 1
        self.embedded_geometry.append(point)

    def embed_line(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
        mesh_size: float | None = None,
    ) -> None:
        """Embeds a point into the mesh.

        Args:
            point1: Point location (``x``, ``y``) of the start of the embedded line.
            point2: Point location (``x``, ``y``) of the end of the embedded line.
            mesh_size: Optional mesh size along the embedded line. If not provided
                takes the mesh size of the polygon the line is embedded into. Defaults
                to ``None``.

        Raises:
            ValueError: If one of the points does not lie within any polygons.
            ValueError: If the points lie within different polygons.

        Warning:
            Ensure that embedded lines lie within the meshed region and are not located
            within any holes in the mesh.

            Due to floating point precision errors, it is recommended that embedded
            lines do not touch the edges of polygons.

            Embedded lines should not cross any other embedded lines or geometry, i.e.
            should not cross from one polygon to another.

            Note, not all the above conditions are checked.
        """
        # find polygon that points lies within
        pts = [shapely.Point(point1[0], point1[1]), shapely.Point(point2[0], point2[1])]
        poly_idxs = []

        for pt in pts:
            for idx, poly in enumerate(self.polygons.geoms):
                if shapely.within(pt, poly):
                    poly_idxs.append(idx)
                    break
            else:
                raise ValueError(f"Point ({pt}) does not lie within any polygons.")

        if poly_idxs[0] != poly_idxs[1]:
            raise ValueError(
                f"Point 1 ({pts[0]}) lies within a different polygon to point 2 "
                f"({pts[1]})"
            )

        # create point and facet objects and assign poly idx (note poly)
        pt1 = Point(x=point1[0], y=point1[1], tol=self.tol, mesh_size=mesh_size)
        pt2 = Point(x=point2[0], y=point2[1], tol=self.tol, mesh_size=mesh_size)
        pt1.poly_idxs = [poly_idxs[0] + 1]  # note poly indexes start at 1
        pt2.poly_idxs = [poly_idxs[1] + 1]  # note poly indexes start at 1
        fct = Facet(pt1=pt1, pt2=pt2)

        # add facet to embedded geometry
        self.embedded_geometry.append(fct)

    def calculate_area(self) -> float:
        """Calculates the area of the geometry.

        Returns:
            Geometry area
        """
        return float(self.polygons.area)

    def calculate_centroid(self) -> tuple[float, float]:
        """Calculates the centroid of the geometry.

        Returns:
            Geometry centroid
        """
        x, y = self.polygons.centroid.coords[0]

        return float(x), float(y)

    def calculate_extents(self) -> tuple[float, float, float, float]:
        """Calculate geometry extents.

        Calculates the minimum and maximum ``x`` and ``y`` values amongst the list of
        points, i.e. the points that describe the bounding box of the ``Geometry``
        instance.

        Returns:
            Minimum and maximum ``x`` and ``y`` values (``x_min``, ``x_max``, ``y_min``,
            ``y_max``)
        """
        min_x, min_y, max_x, max_y = self.polygons.bounds
        return min_x, max_x, min_y, max_y

    def align_to(
        self,
        other: Geometry | tuple[float, float],
        on: str,
        inner: bool = False,
    ) -> Geometry:
        """Aligns the geometry to another ``Geometry`` object or point.

        Returns a new ``Geometry`` object, representing ``self`` translated so that is
        aligned on one of the outer or inner bounding box edges of ``other``.

        Args:
            other: A ``Geometry`` or point (``x``, ``y``) to align to.
            on: Which side of ``other`` to align to, either ``“left”``, ``“right”``,
                ``“bottom”``, or ``“top”``.
            inner: If ``True``, aligns to the inner bounding box edge of ``other``.
                Defaults to ``False``.

        Returns:
            New ``Geometry`` object aligned to ``other``.

        Example:
            TODO.
        """
        # setup mappings for transformations
        align_self_map = {
            "left": 1,
            "right": 0,
            "bottom": 3,
            "top": 2,
        }
        other_as_geom_map = {
            "left": 0,
            "right": 1,
            "bottom": 2,
            "top": 3,
        }
        other_as_point_map = {
            "left": 0,
            "right": 0,
            "bottom": 1,
            "top": 1,
        }

        # get the coordinate to align from
        self_extents = self.calculate_extents()
        self_align_idx = align_self_map[on]
        self_align_coord = self_extents[self_align_idx]

        # get the coordinate to align to
        if isinstance(other, Geometry):
            align_to_idx = other_as_geom_map[on]
            align_to_coord = other.calculate_extents()[align_to_idx]
        else:
            align_to_idx = other_as_point_map[on]
            align_to_coord = other[align_to_idx]

        if inner:
            self_align_coord = self_extents[align_to_idx]

        # calculate offset
        offset = align_to_coord - self_align_coord

        # shift geometry
        if on in ["top", "bottom"]:
            arg = "y"
        else:
            arg = "x"

        kwargs = {arg: offset}

        return self.shift_geometry(**kwargs)

    def align_center(
        self,
        align_to: Geometry | tuple[float, float] | None = None,
    ) -> Geometry:
        """Aligns the centroid of the geometry to another ``Geometry``, point or origin.

        Returns a new ``Geometry`` object, translated such that its centroid is aligned
        to the centroid of another ``Geometry``, a point, or the origin.

        Args:
            align_to: Location to align the centroid to, either another ``Geometry``
                object, a point (``x``, ``y``) or ``None``. Defaults to ``None`` (i.e.
                align to the origin).

        Raises:
            ValueError: If ``align_to`` is not a valid input.

        Returns:
            New ``Geometry`` object aligned to ``align_to``.

        Example:
            TODO.
        """
        cx, cy = self.calculate_centroid()

        # align to geometry centroid
        if align_to is None:
            shift_x, shift_y = round(-cx, self.tol), round(-cy, self.tol)
        # align to centroid of another geometry
        elif isinstance(align_to, Geometry):
            other_cx, other_cy = align_to.calculate_centroid()
            shift_x = round(other_cx - cx, self.tol)
            shift_y = round(other_cy - cy, self.tol)
        # align to a point
        else:
            try:
                point_x, point_y = align_to
                shift_x = round(point_x - cx, self.tol)
                shift_y = round(point_y - cy, self.tol)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"align_to must be either a Geometry object or an (x, y) "
                    f"coordinate, not {align_to}."
                ) from exc

        return self.shift_geometry(x=shift_x, y=shift_y)

    def shift_geometry(
        self,
        x: float = 0.0,
        y: float = 0.0,
    ) -> Geometry:
        """Shifts the geometry by (``x``, ``y``).

        Args:
            x: Distance to shift along the ``x`` axis. Defaults to ``0.0``.
            y: Distance to shift along the ``y`` axis. Defaults to ``0.0``.

        Returns:
            New ``Geometry`` object shifted by (``x``, ``y``).

        Example:
            TODO.
        """
        return Geometry(
            polygons=affinity.translate(geom=self.polygons, xoff=x, yoff=y),
            materials=self.materials,
            embedded_geometry=self.embedded_geometry,
            tol=self.tol,
        )

    def rotate_geometry(
        self,
        angle: float,
        rot_point: tuple[float, float] | str = "center",
        use_radians: bool = False,
    ) -> Geometry:
        """Rotates the geometry by an ``angle`` about a ``rot_point``.

        Args:
            angle: Angle by which to rotate the geometry. A positive angle leads to a
                counter-clockwise rotation.
            rot_point: Point (``x``, ``y``) about which to rotate the geometry. May also
                be ``"center"`` (rotates about center of bounding box) or "centroid"
                (rotates about centroid). Defaults to ``"center"``.
            use_radians: If True, ``angle`` is in radians, if ``False`` angle is in
                degrees. Defaults to ``False``.

        Returns:
            New ``Geometry`` object rotated by ``angle`` about ``rot_point``.

        Example:
            TODO.
        """
        return Geometry(
            polygons=affinity.rotate(
                geom=self.polygons,
                angle=angle,
                origin=rot_point,
                use_radians=use_radians,
            ),
            materials=self.materials,
            embedded_geometry=self.embedded_geometry,
            tol=self.tol,
        )

    def mirror_geometry(
        self,
        axis: str = "x",
        mirror_point: tuple[float, float] | str = "center",
    ) -> Geometry:
        """Mirrors the geometry about a point on either the ``x`` or ``y`` axis.

        Args:
            axis: Mirror axis, may be ``"x"`` or ``"y"``. Defaults to ``"x"``.
            mirror_point: Point (``x``, ``y``) about which to mirror the geometry. May
                also be ``"center"`` (mirrors about center of bounding box) or
                "centroid" (mirrors about centroid). Defaults to ``"center"``.

        Raises:
            ValueError: If ``axis`` is not ``"x"`` or ``"y"``.

        Returns:
             New ``Geometry`` object mirrored about ``mirror_point`` on ``axis``.

        Example:
            TODO.
        """
        if axis == "x":
            xfact = 1.0
            yfact = -1.0
        elif axis == "y":
            xfact = -1.0
            yfact = 1.0
        else:
            raise ValueError(f"axis must be 'x' or 'y', not {axis}.")

        return Geometry(
            polygons=affinity.scale(
                geom=self.polygons, xfact=xfact, yfact=yfact, origin=mirror_point
            ),
            materials=self.materials,
            embedded_geometry=self.embedded_geometry,
            tol=self.tol,
        )

    def __or__(
        self,
        other: Geometry,
    ) -> Geometry:
        """Performs a union operation using the ``|`` operator.

        .. note::
            The material of the first geometry is applied to the entire region of the
            "unioned" geometry. Keeps the smallest value of ``tol`` between the two
            geometries.

        Args:
            other: ``Geometry`` object to union with.

        Raises:
            ValueError: If ``shapely`` is unable to perform the union.

        Returns:
            New ``Geometry`` object unioned with ``other``.

        Example:
            TODO.
        """
        tol = min(self.tol, other.tol)

        try:
            new_polygon = self.filter_non_polygons(
                input_geom=self.polygons | other.polygons
            )

            return Geometry(
                polygons=new_polygon,
                materials=self.materials[0],
                embedded_geometry=self.embedded_geometry + other.embedded_geometry,
                tol=tol,
            )
        except Exception as exc:
            raise ValueError(
                f"Cannot perform union on these two objects: {self} | {other}"
            ) from exc

    def __sub__(
        self,
        other: Geometry,
    ) -> Geometry:
        """Performs a difference operation using the ``-`` operator.

        .. warning::
            If ``self`` or ``other`` contains multiple regions, these regions may be
            combined into one region after the difference operation. It is recommended
            to first perform difference operations on :class:`~shapely.Polygon` objects,
            and later combine into into :class:`shapely.MultiPolygon` objects, see the
            example below. *Check the assignment of materials after a difference
            operation.*

        Args:
            other: ``Geometry`` object to difference with.

        Raises:
            ValueError: If ``shapely`` is unable to perform the difference.

        Returns:
            New ``Geometry`` object differenced with ``other``.

        Example:
            TODO. Use brackets to show order of operations important!
        """
        try:
            new_polygon = self.filter_non_polygons(
                input_geom=self.polygons - other.polygons
            )

            # non-polygon results
            if isinstance(new_polygon, shapely.GeometryCollection):
                raise ValueError(
                    f"Cannot perform difference on these two objects: {self} - {other}"
                )
            # polygon or multipolygon object
            elif isinstance(new_polygon, (shapely.Polygon, shapely.MultiPolygon)):
                return Geometry(
                    polygons=new_polygon, materials=self.materials, tol=self.tol
                )
            else:
                raise ValueError(
                    f"Cannot perform difference on these two objects: {self} - {other}"
                )
        except Exception as exc:
            raise ValueError(
                f"Cannot perform difference on these two objects: {self} - {other}"
            ) from exc

    def __add__(
        self,
        other: Geometry,
    ) -> Geometry:
        """Performs an addition operation using the ``+`` operator.

        .. note::
            The smallest value of ``tol`` is applied to both geometries.

        Args:
            other: ``Geometry`` object to add to.

        Returns:
            New ``Geometry`` object added with ``other``.

        Example:
            TODO.
        """
        poly_list: list[shapely.Polygon] = []
        mat_list: list[Material] = []

        # loop through each list of polygons and combine
        for poly, mat in zip(self.polygons.geoms, self.materials):
            poly_list.append(poly)
            mat_list.append(mat)

        for poly, mat in zip(other.polygons.geoms, other.materials):
            poly_list.append(poly)
            mat_list.append(mat)

        tol = min(self.tol, other.tol)

        return Geometry(
            polygons=shapely.MultiPolygon(polygons=poly_list),
            materials=mat_list,
            embedded_geometry=self.embedded_geometry + other.embedded_geometry,
            tol=tol,
        )

    def __and__(
        self,
        other: Geometry,
    ) -> Geometry:
        """Performs an intersection operation using the ``&`` operator.

        .. note::
            The material of the first geometry is applied to the entire region of the
            "intersected" geometry. Keeps the smallest value of ``tol`` between the two
            geometries.

        Args:
            other: ``Geometry`` object to intersection with.

        Raises:
            ValueError: If ``shapely`` is unable to perform the intersection.

        Returns:
            New ``Geometry`` object intersected with ``other``.

        Example:
            TODO.
        """
        tol = min(self.tol, other.tol)

        try:
            new_polygon = self.filter_non_polygons(
                input_geom=self.polygons & other.polygons
            )

            return Geometry(
                polygons=new_polygon,
                materials=self.materials[0],
                embedded_geometry=self.embedded_geometry + other.embedded_geometry,
                tol=tol,
            )
        except Exception as exc:
            raise ValueError(
                f"Cannot perform intersection on these two Geometry instances: "
                f"{self} & {other}"
            ) from exc

    @staticmethod
    def filter_non_polygons(
        input_geom: shapely.GeometryCollection
        | shapely.LineString
        | shapely.Point
        | shapely.Polygon
        | shapely.MultiPolygon,
    ) -> shapely.Polygon | shapely.MultiPolygon:
        """Filters shapely geometries to return only polygons.

        Returns a ``Polygon`` or a ``MultiPolygon`` representing any such ``Polygon`` or
        ``MultiPolygon`` that may exist in the ``input_geom``. If ``input_geom`` is a
        ``LineString`` or ``Point``, an empty ``Polygon`` is returned.

        Args:
            input_geom: Shapely geometry to filter

        Returns:
            Filtered polygon
        """
        if isinstance(input_geom, (shapely.Polygon, shapely.MultiPolygon)):
            return input_geom
        elif isinstance(input_geom, shapely.GeometryCollection):
            acc = []

            for item in input_geom.geoms:
                if isinstance(item, (shapely.Polygon, shapely.MultiPolygon)):
                    acc.append(item)

            if len(acc) == 0:
                return shapely.Polygon()
            elif len(acc) == 1:
                return acc[0]
            else:
                return shapely.MultiPolygon(polygons=acc)
        elif isinstance(input_geom, (shapely.Point, shapely.LineString)):
            return shapely.Polygon()
        else:
            return shapely.Polygon()

    def create_mesh(
        self,
        mesh_sizes: float | list[float] = 0.0,
        quad_mesh: bool = False,
        mesh_order: int = 1,
        serendipity: bool = False,
        mesh_algorithm: int = 6,
        subdivision_algorithm: int = 0,
        fields: list[Field] | None = None,
    ) -> None:
        """Creates and stores a triangular mesh of the geometry.

        Args:
            mesh_sizes: A list of the characteristic mesh lengths for each ``polygon``
                in the ``Geometry`` object. If a list of length 1 or a ``float`` i
                passed, then this one size will be applied to all ``polygons``. A value
                of ``0`` removes the area constraint. Defaults to ``0.0``.
            quad_mesh: If set to ``True``, recombines the triangular mesh to create
                quadrilaterals. Defaults to ``False``.
            mesh_order: Order of the mesh, ``1`` - linear or ``2`` - quadratic. Defaults
                to ``1``.
            serendipity: If set to ``True``, creates serendipity elements for
                quadrilateral meshes, i.e. creates ``"Quad8"`` elements instead of
                ``"Quad9"`` elements. Defaults to ``False``.
            mesh_algorithm: Gmsh meshing algorithm, see below for more details. Defaults
                to ``6``.
            subdivision_algorithm: Gmsh subdivision algorithm, see below for more
                details. Defaults to ``0``.
            fields: A list of ``Field`` objects, describing mesh refinement fields.

        Raises:
            ValueError: If the length of ``mesh_sizes`` does not equal the number of
                polygons, or is not a float/list of length 1.

        .. admonition:: ``mesh_algorithm``

            TODO - information about meshing algorithm.

        .. admonition:: ``subdivision_algorithm``

            TODO - information about subdivision algorithm.
        """
        # convert mesh_size to an appropriately sized list
        if isinstance(mesh_sizes, (float, int)):
            mesh_sizes = [float(mesh_sizes)] * len(self.surfaces)

        if len(mesh_sizes) == 1:
            mesh_sizes = mesh_sizes * len(self.surfaces)

        # check mesh_sizes length
        if len(mesh_sizes) != len(self.surfaces):
            raise ValueError(
                "Length of 'mesh_sizes' must equal the number of polygons or 1."
            )

        self.mesh.create_mesh(
            points=self.points,
            facets=self.facets,
            curve_loops=self.curve_loops,
            surfaces=self.surfaces,
            materials=self.materials,
            embedded_geometry=self.embedded_geometry,
            mesh_sizes=mesh_sizes,
            quad_mesh=quad_mesh,
            mesh_order=mesh_order,
            serendipity=serendipity,
            mesh_algorithm=mesh_algorithm,
            subdivision_algorithm=subdivision_algorithm,
            fields=[] if fields is None else fields,
        )

    def plot_geometry(
        self,
        load_case: LoadCase | None = None,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """Plots the geometry.

        Optionally also renders the boundary conditions of a load case if provided.

        Args:
            load_case: Plots the boundary conditions within a load case if provided.
                Defaults to ``None``.
            kwargs: See below.

        Keyword Args:
            title (str): Plot title. Defaults to ``"Geometry"``.
            tags(list[str]): A list of tags to plot, can contain any of the following:
                ``"points"``, ``"facets"``. Defaults to ``[]``.
            legend (bool):  If set to ``True``, plots the legend. Defaults to ``True``.
            kwargs (dict[str, Any]): Other keyword arguments are passed to
                :meth:`~planestress.post.plotting.plotting_context`.

        Returns:
            Matplotlib axes object.

        Example:
            TODO.
        """
        # get keyword arguments
        title: str = kwargs.pop("title", "Geometry")
        tags: list[str] = kwargs.pop("tags", [])
        legend: bool = kwargs.pop("legend", True)

        # create plot and setup the plot
        with plotting_context(title=title, **kwargs) as (_, ax):
            assert ax
            label: str | None

            # plot the points and facets
            label = "Points & Facets"
            for fct in self.facets:
                ax.plot(
                    (fct.pt1.x, fct.pt2.x),
                    (fct.pt1.y, fct.pt2.y),
                    "ko-",
                    markersize=2,
                    linewidth=1.5,
                    label=label,
                )
                label = None

            # plot the holes
            label = "Holes"
            for hl in self.holes:
                ax.plot(hl.x, hl.y, "rx", markersize=5, markeredgewidth=1, label=label)
                label = None

            # display the tags
            # plot point tags
            if "points" in tags:
                for pt in self.points:
                    ax.annotate(str(pt.idx + 1), xy=(pt.x, pt.y), color="r")

            # plot facet tags
            if "facets" in tags:
                for fct in self.facets:
                    xy = (fct.pt1.x + fct.pt2.x) / 2, (fct.pt1.y + fct.pt2.y) / 2

                    ax.annotate(str(fct.idx + 1), xy=xy, color="b")

            # plot the load case
            if load_case is not None:
                for boundary_condition in load_case.boundary_conditions:
                    # boundary_condition.plot()
                    print(boundary_condition)  # TODO - plot this!

            # display the legend
            if legend:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        return ax

    def plot_mesh(
        self,
        load_case: LoadCase | None = None,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        r"""Plots the finite element mesh.

        Optionally also renders the boundary conditions of a load case if provided.

        Args:
            load_case: Plots the boundary conditions within a load case if provided.
                Defaults to ``None``.
            kwargs: See below.

        Keyword Args:
            title (str): Plot title. Defaults to ``"Finite Element Mesh"``.
            materials (bool): If set to ``True`` shades the elements with the specified
                material colors. Defaults to ``True``.
            node_indexes (bool): If set to ``True``, plots the indexes of each node.
                Defaults to ``False``.
            element_indexes (bool): If set to ``True``, plots the indexes of each
                element. Defaults to ``False``.
            alpha (float): Transparency of the mesh outlines,
                :math:`0 \leq \alpha \leq 1`. Defaults to ``0.5``.
            kwargs (dict[str, Any]): Other keyword arguments are passed to
                :meth:`~planestress.post.plotting.plotting_context`.

        Raises:
            RuntimeError: If a mesh has not yet been generated.

        Returns:
            Matplotlib axes object.

        Example:
            TODO.
        """
        # get keyword arguments
        title: str = kwargs.pop("title", "Finite Element Mesh")
        materials: bool = kwargs.pop("materials", True)
        node_indexes: bool = kwargs.pop("node_indexes", False)
        element_indexes: bool = kwargs.pop("element_indexes", False)
        alpha: float = kwargs.pop("alpha", 0.5)

        if len(self.mesh.nodes) > 0:
            return self.mesh.plot_mesh(
                load_case=load_case,
                title=title,
                materials=materials,
                node_indexes=node_indexes,
                element_indexes=element_indexes,
                alpha=alpha,
                **kwargs,
            )
        else:
            raise RuntimeError("Generate a mesh with create_mesh() prior to plotting.")


@dataclass(eq=True)
class Point:
    """Class describing a point in 2D space.

    Args:
        x: ``x`` location of the point.
        y: ``y`` location of the point.
        tol: Number of digits to round the point to.

    Attributes:
        idx: Point index.
        poly_idxs: Indexes of polygons that contain the point.
        mesh_size: Mesh size at the point.
    """

    x: float
    y: float
    tol: int
    idx: int = field(init=False)
    poly_idxs: list[int] = field(init=False, default_factory=list)
    mesh_size: float | None = None

    def __post_init__(self) -> None:
        """Point object post init method."""
        self.round()

    def __eq__(
        self,
        other: object,
    ) -> bool:
        """Override __eq__ method to base equality on location only.

        Args:
            other: Other object to check equality against.

        Returns:
            ``True`` if ``Points`` objects are equal.
        """
        if isinstance(other, Point):
            tol = 10.0 ** (-self.tol + 1)
            x_diff = abs(self.x - other.x)
            y_diff = abs(self.y - other.y)
            return x_diff <= tol and y_diff <= tol
        else:
            return False

    def __repr__(self) -> str:
        """Override __repr__ method to account for unbound idx.

        Returns:
            String representation of the object.
        """
        try:
            idx = self.idx
        except AttributeError:
            idx = None

        return f"Point(x={self.x}, y={self.y}, tol={self.tol}, idx={idx})"

    def round(self) -> None:
        """Rounds the point to ``tol`` digits."""
        self.x = round(self.x, self.tol)
        self.y = round(self.y, self.tol)

    def to_tuple(self) -> tuple[float, float]:
        """Converts the point to a tuple.

        Returns:
            ``Point`` in tuple format (``x``, ``y``).
        """
        return self.x, self.y

    def to_shapely_point(self) -> shapely.Point:
        """Converts the point to a ``shapely`` ``Point`` object.

        Returns:
            ``Point`` as a :class:`shapely.Point`.
        """
        return shapely.Point(self.x, self.y)


@dataclass(eq=True)
class Facet:
    """Class describing a facet of a 2D geometry, i.e. an edge.

    Args:
        pt1: First point in the facet.
        pt2: Second point in the facet.
        poly_idx: Polygon index.
        loop_idx: Loop index.

    Attributes:
        idx: Facet index.
    """

    pt1: Point
    pt2: Point
    idx: int = field(init=False)

    def __eq__(
        self,
        other: object,
    ) -> bool:
        """Override __eq__ method to account for points in either order.

        Args:
            other: Other object to check equality against.

        Returns:
            ``True`` if ``Facet`` objects are equal.
        """
        if isinstance(other, Facet):
            return (self.pt1 == other.pt1 and self.pt2 == other.pt2) or (
                self.pt1 == other.pt2 and self.pt2 == other.pt1
            )
        else:
            return False

    def to_tuple(self) -> tuple[float, float]:
        """Converts the facet to a tuple.

        Raises:
            RuntimeError: If a point in the facet hasn't been assigned an index.

        Returns:
            ``Facet`` in tuple format (``pt1_idx``, ``pt2_idx``).
        """
        idx_1 = self.pt1.idx
        idx_2 = self.pt2.idx

        if idx_1 is None:
            raise RuntimeError(f"Point 1: {self.pt1} has not been assigned an index.")

        if idx_2 is None:
            raise RuntimeError(f"Point 2: {self.pt2} has not been assigned an index.")

        return idx_1, idx_2

    def to_shapely_line(self) -> shapely.LineString:
        """Converts the line to a ``shapely`` ``Line`` object.

        Returns:
            ``Facet`` as a :class:`shapely.LineString`.
        """
        return shapely.LineString(
            [self.pt1.to_shapely_point(), self.pt2.to_shapely_point()]
        )

    def zero_length(self) -> bool:
        """Tests whether or not a facet is zero length.

        Returns:
            ``True`` if the facet has zero length (i.e. ``pt1 == pt2``).
        """
        return self.pt1 == self.pt2

    def update_point(
        self,
        old: Point,
        new: Point,
    ) -> None:
        """If the facet contains the point ``old``, replace with ``new``.

        Args:
            old: Old ``Point`` to replace.
            new: ``Point`` to replace ``old`` with.
        """
        if self.pt1 == old:
            self.pt1 = new

        if self.pt2 == old:
            self.pt2 = new


@dataclass
class CurveLoop:
    """Class describing a curve loop of a 2D geometry, i.e. a group of facets.

    Args:
        idx: Curve loop index.

    Attributes:
        facets: List of facets in the curve loop.
    """

    idx: int
    facets: list[Facet] = field(init=False, default_factory=list)

    def update_facet(
        self,
        old: Facet,
        new: Facet,
    ) -> None:
        """If the curve loop contains the facet ``old``, replace with ``new``.

        Args:
            old: Old ``Facet`` to replace.
            new: ``Facet`` to replace ``old`` with.
        """
        # loop through facets
        for idx, fct in enumerate(self.facets):
            if fct == old:
                self.facets[idx] = new
                return


@dataclass
class Surface:
    """Class describing a surface of a 2D geometry, i.e. a group of curve loops.

    The first loop will be the outer shell, while the remaining (if any) loops define
    the interiors (hole edges).

    Args:
        idx: Surface index.

    Attributes:
        curve_loops: List of curve loops on the surface.
    """

    idx: int
    curve_loops: list[CurveLoop] = field(init=False, default_factory=list)

    def point_on_surface(
        self,
        point: Point,
    ) -> bool:
        """Checks to see if a ``Point`` is on this surface.

        Args:
            point: ``Point`` object.

        Returns:
            ``True`` if the point is on this surface.
        """
        # loop through curve loops
        for loop in self.curve_loops:
            # loop through facets
            for facet in loop.facets:
                if point == facet.pt1 or point == facet.pt2:
                    return True

        return False
