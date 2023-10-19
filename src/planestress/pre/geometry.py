"""Classes relating to planestress geometry."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import shapely as shapely
import shapely.affinity as affinity
from triangle import triangulate

import planestress.pre.boundary_condition as bc
from planestress.post.plotting import plotting_context
from planestress.pre.material import DEFAULT_MATERIAL, Material
from planestress.pre.mesh import Mesh


if TYPE_CHECKING:
    import matplotlib.axes

    from planestress.pre.load_case import LoadCase


class Geometry:
    """Class describing a geometric region."""

    def __init__(
        self,
        polygons: shapely.Polygon | shapely.MultiPolygon,
        materials: Material | list[Material] = DEFAULT_MATERIAL,
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
            tol: The points in the geometry get rounded to ``tol`` digits. Defaults to
                ``12``.

        Raises:
            ValueError:
                If the number of ``materials`` does not equal the number of
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
        self.tol = tol

        # allocate points, facets, holes, control_points
        self.points: list[Point] = []
        self.facets: list[Facet] = []
        self.holes: list[Point] = []
        self.control_points: list[Point] = []

        # compile the geometry into points, facets, holes and control_points
        self.compile_geometry()

        # create point STRtree
        self.pts_str_tree = shapely.STRtree(
            [pt.to_shapely_point() for pt in self.points]
        )
        self.fcts_str_tree = shapely.STRtree(
            [fct.to_shapely_line() for fct in self.facets]
        )

        # allocate mesh variables
        self.mesh: Mesh | None = None
        self.point_markers = [0] * len(self.points)
        self.facet_markers = [0] * len(self.facets)

    def compile_geometry(self) -> None:
        """Creates points, facets, holes and control_points from shapely geometry."""
        # loop through each polygon
        for poly in self.polygons.geoms:
            # first create points,facets, holes and control_points for each polygon
            poly_points, poly_facets, poly_holes, poly_cp = self.compile_polygon(
                polygon=poly
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

            # add facets to the global list, skipping duplicate and zero length facets
            for fct in poly_facets:
                if fct not in self.facets and not fct.zero_length():
                    self.facets.append(fct)

            # add holes to list of multipolygon holes
            self.holes.extend(poly_holes)

            # add control points to the global list
            self.control_points.append(poly_cp)

        # assign point indexes
        for idx, pt in enumerate(self.points):
            pt.idx = idx

    def compile_polygon(
        self,
        polygon: shapely.Polygon,
    ) -> tuple[list[Point], list[Facet], list[Point], Point]:
        """Creates points, facets, holes and a control point given a ``Polygon``.

        Args:
            polygon: A :class:`~shapely.Polygon` object.

        Returns:
            A list of points, facets, holes and a control point (``points``, ``facets``,
            ``holes``, ``control_point``).
        """
        pt_list: list[Point] = []
        fct_list: list[Facet] = []
        hl_list: list[Point] = []

        # construct perimeter points (note shapely doubles up first & last point)
        for coords in list(polygon.exterior.coords[:-1]):
            new_pt = Point(x=coords[0], y=coords[1], tol=self.tol)
            pt_list.append(new_pt)

        # create perimeter facets
        fct_list.extend(self.create_facet_list(pt_list=pt_list))

        # construct holes, for each interior (hole) region
        for hl in polygon.interiors:
            int_pt_list: list[Point] = []

            # create hole (note shapely doubles up first & last point)
            for coords in hl.coords[:-1]:
                new_pt = Point(x=coords[0], y=coords[1], tol=self.tol)
                int_pt_list.append(new_pt)

            # add interior points to poly list
            pt_list.extend(int_pt_list)

            # create hole facets
            fct_list.extend(self.create_facet_list(pt_list=int_pt_list))

            # create hole point
            # first convert the list of interior points to a list of tuples
            int_pt_list_tup = [hl_pt.to_tuple() for hl_pt in int_pt_list]

            # create a polygon of the hole region
            hl_poly = shapely.Polygon(int_pt_list_tup)

            # add hole point to the list of hole points
            hl_pt_coords = hl_poly.representative_point().coords
            hl_list.append(
                Point(x=hl_pt_coords[0][0], y=hl_pt_coords[0][1], tol=self.tol)
            )

        # construct control point
        cp_pt_coords = polygon.representative_point().coords
        cp_pt = Point(x=cp_pt_coords[0][0], y=cp_pt_coords[0][1], tol=self.tol)

        return pt_list, fct_list, hl_list, cp_pt

    @staticmethod
    def create_facet_list(pt_list: list[Point]) -> list[Facet]:
        """Creates a closed list of facets from a list of points.

        Args:
            pt_list: List of points.

        Returns:
            Closed list of facets.
        """
        fct_list: list[Facet] = []

        # create facets
        for idx, pt in enumerate(pt_list):
            pt1 = pt
            # if we are not at the end of the list
            if idx + 1 != len(pt_list):
                pt2 = pt_list[idx + 1]
            # otherwise loop back to starting point
            else:
                pt2 = pt_list[0]

            fct_list.append(Facet(pt1=pt1, pt2=pt2))

        return fct_list

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

        # shift section
        if on in ["top", "bottom"]:
            arg = "y"
        else:
            arg = "x"

        kwargs = {arg: offset}

        return self.shift_section(**kwargs)

    def align_centre(
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

        return self.shift_section(x=shift_x, y=shift_y)

    def shift_section(
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
            tol=self.tol,
        )

    def rotate_section(
        self,
        angle: float,
        rot_point: tuple[float, float] | str = "center",
        use_radians: bool = False,
    ) -> Geometry:
        """Rotates the geometry by an ``angle`` about a ``rot_point``.

        Args:
            angle: Angle by which to rotate the section. A positive angle leads to a
                counter-clockwise rotation.
            rot_point: Point (``x``, ``y``) about which to rotate the section. May also
                be ``"center"`` (rotates about centre of bounding box) or "centroid"
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
                origin=rot_point,  # type: ignore
                use_radians=use_radians,
            ),
            materials=self.materials,
            tol=self.tol,
        )

    def mirror_section(
        self,
        axis: str = "x",
        mirror_point: tuple[float, float] | str = "center",
    ) -> Geometry:
        """Mirrors the geometry about a point on either the ``x`` or ``y`` axis.

        Args:
            axis: Mirror axis, may be ``"x"`` or ``"y"``. Defaults to ``"x"``.
            mirror_point: Point (``x``, ``y``) about which to mirror the section. May
                also be ``"center"`` (mirrors about centre of bounding box) or
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
                geom=self.polygons, xfact=xfact, yfact=yfact, origin=mirror_point  # type: ignore
            ),
            materials=self.materials,
            tol=self.tol,
        )

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

            return Geometry(polygons=new_polygon, materials=self.materials[0], tol=tol)
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
                input_geom=self.polygons - other.polygons
            )

            return Geometry(polygons=new_polygon, materials=self.materials[0], tol=tol)
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

    def find_point_index(
        self,
        point: tuple[float, float],
    ) -> int:
        """Returns the index of the point in the geometry closest to ``point``.

        Args:
            point: Point (``x``, ``y``) to find in the geometry.

        Returns:
            Index of closest point in geometry to ``point``.
        """
        pt = shapely.Point(point[0], point[1])
        idx = self.pts_str_tree.nearest(geometry=pt)

        return cast(int, idx)

    def find_facet_index(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
    ) -> int:
        """Returns the index of the facet in the geometry closest to ``facet``.

        Args:
            point1: First point (``x``, ``y``) of the facet to find in the geometry.
            point2: Second point (``x``, ``y``) of the facet to find in the geometry.

        Returns:
            Index of closest facet in geometry to ``facet``.
        """
        mid_point = shapely.Point(
            0.5 * (point1[0] + point2[0]), 0.5 * (point1[1] + point2[1])
        )
        idx = self.fcts_str_tree.nearest(geometry=mid_point)

        return cast(int, idx)

    def add_point_marker(
        self,
        pt_idx: int,
    ) -> int:
        """Performs the tasks required to add a marker to a point.

        Args:
            pt_idx: Index of the point to add a marker to.

        Raises:
            ValueError: If the index is invalid.

        Returns:
            Mesh marker ID.
        """
        # check point index lies in range
        if pt_idx < 0 or pt_idx > len(self.points) - 1:
            raise ValueError(
                f"pt_idx must be an integer between 0 and {len(self.points) - 1}."
            )

        # check to see if a mesh has already been generated
        if self.mesh:
            warnings.warn(
                "Please regenerate the mesh prior to creating a PlaneStress object."
            )

        # check to see if point already has custom point marker
        if self.point_markers[pt_idx] == 0:
            # add point marker to point
            # note custom point marker ids start at 2 and increment by 2
            if max(self.point_markers) == 0:
                marker_id = 2
            else:
                marker_id = max(self.point_markers) + 2

            self.point_markers[pt_idx] = marker_id
        else:
            # get marker id
            marker_id = self.point_markers[pt_idx]

        return marker_id

    def add_facet_marker(
        self,
        fct_idx: int,
    ) -> int:
        """Performs the tasks required to add a marker to a facet.

        Args:
            fct_idx: Index of the facet to add a marker to.

        Raises:
            ValueError: If the index is invalid.

        Returns:
            Mesh marker ID.
        """
        # check facet index lies in range
        if fct_idx < 0 or fct_idx > len(self.facets) - 1:
            raise ValueError(
                f"fct_idx must be an integer between 0 and {len(self.facets) - 1}."
            )

        # check to see if a mesh has already been generated
        if self.mesh:
            warnings.warn(
                "Please regenerate the mesh prior to creating a PlaneStress object."
            )

        # check to see if facet already has custom point marker
        if self.facet_markers[fct_idx] == 0:
            # add facet marker to facet
            # note custom facet marker ids start at 3 and increment by 2
            if max(self.facet_markers) == 0:
                marker_id = 3
            else:
                marker_id = max(self.facet_markers) + 2

            self.facet_markers[fct_idx] = marker_id
        else:
            # get marker id
            marker_id = self.facet_markers[fct_idx]

        return marker_id

    def add_node_support(
        self,
        point: tuple[float, float],
        direction: str,
        value: float = 0.0,
        pt_idx: int | None = None,
    ) -> bc.NodeSupport:
        """Adds a node support to the geometry.

        Args:
            point: Point location (``x``, ``y``) of the node support.
            direction: Direction of the node support, either ``"x"`` or ``"y"``.
            value: Imposed displacement to apply to the node support. Defaults to
                ``0.0``, i.e. a fixed node support.
            pt_idx: If the index of the point is known, this can be provided as an
                alternative to ``point``. Defaults to ``None``.

        Warns:
            If the node support is added after generating a mesh, the mesh will need to
            be regenerated prior to creating a
            :class:`~planestress.analysis.PlaneStress` object.

        Returns:
            Node support boundary condition object.

        Example:
            TODO.
        """
        # get the point index
        if not pt_idx:
            pt_idx = self.find_point_index(point=point)

        # add the point marker to the specified point
        marker_id = self.add_point_marker(pt_idx=pt_idx)

        # create node support boundary condition
        node_support = bc.NodeSupport(
            marker_id=marker_id, direction=direction, value=value
        )

        return node_support

    def add_node_spring(
        self,
        point: tuple[float, float],
        direction: str,
        value: float,
        pt_idx: int | None = None,
    ) -> bc.NodeSpring:
        """Adds a node spring to the geometry.

        Args:
            point: Point location (``x``, ``y``) of the node spring.
            direction: Direction of the node spring, either ``"x"`` or ``"y"``.
            value: Spring stiffness.
            pt_idx: If the index of the point is known, this can be provided as an
                alternative to ``point``. Defaults to ``None``.

        Warns:
            If the node spring is added after generating a mesh, the mesh will need to
            be regenerated prior to creating a
            :class:`~planestress.analysis.PlaneStress` object.

        Returns:
            Node spring boundary condition object.

        Example:
            TODO.
        """
        # get the point index
        if not pt_idx:
            pt_idx = self.find_point_index(point=point)

        # add the point marker to the specified point
        marker_id = self.add_point_marker(pt_idx=pt_idx)

        # create node spring boundary condition
        node_spring = bc.NodeSpring(
            marker_id=marker_id, direction=direction, value=value
        )

        return node_spring

    def add_node_load(
        self,
        point: tuple[float, float],
        direction: str,
        value: float,
        pt_idx: int | None = None,
    ) -> bc.NodeLoad:
        """Adds a node load to the geometry.

        Args:
            point: Point location (``x``, ``y``) of the node load.
            direction: Direction of the node load, either ``"x"`` or ``"y"``.
            value: Node load.
            pt_idx: If the index of the point is known, this can be provided as an
                alternative to ``point``. Defaults to ``None``.

        Warns:
            If the node load is added after generating a mesh, the mesh will need to
            be regenerated prior to creating a
            :class:`~planestress.analysis.PlaneStress` object.

        Returns:
            Node load boundary condition object.

        Example:
            TODO.
        """
        # get the point index
        if not pt_idx:
            pt_idx = self.find_point_index(point=point)
            print(pt_idx)

        # add the point marker to the specified point
        marker_id = self.add_point_marker(pt_idx=pt_idx)

        # create node support boundary condition
        node_load = bc.NodeLoad(marker_id=marker_id, direction=direction, value=value)

        return node_load

    def add_line_support(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
        direction: str,
        value: float = 0.0,
        fct_idx: int | None = None,
    ) -> bc.LineSupport:
        """Adds a line support to the geometry.

        All nodes along the line will have this boundary condition applied.

        Args:
            point1: Point location (``x``, ``y``) of the start of the line support.
            point2: Point location (``x``, ``y``) of the end of the line support.
            direction: Direction of the line support, either ``"x"`` or ``"y"``.
            value: Imposed displacement to apply to the line support. Defaults to
                ``0.0``, i.e. a fixed line support.
            fct_idx: If the index of the facet is known, this can be provided as an
                alternative to ``point1`` and ``point2``. Defaults to ``None``.

        Warns:
            If the line support is added after generating a mesh, the mesh will need to
            be regenerated prior to creating a
            :class:`~planestress.analysis.PlaneStress` object.

        Returns:
            Line support boundary condition object.

        Example:
            TODO.
        """
        # get the facet index
        if not fct_idx:
            fct_idx = self.find_facet_index(point1=point1, point2=point2)

        # add the facet marker to the specified point
        marker_id = self.add_facet_marker(fct_idx=fct_idx)

        # create line support boundary condition
        line_support = bc.LineSupport(
            marker_id=marker_id, direction=direction, value=value
        )

        return line_support

    def add_line_spring(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
        direction: str,
        value: float,
        fct_idx: int | None = None,
    ) -> bc.LineSpring:
        """Adds a line spring to the geometry.

        The spring stiffness is specified per unit length and equivalent nodal springs
        applied to nodes along this line.

        TODO - look into elastic foundation?

        Args:
            point1: Point location (``x``, ``y``) of the start of the line spring.
            point2: Point location (``x``, ``y``) of the end of the line spring.
            direction: Direction of the line spring, either ``"x"`` or ``"y"``.
            value: Spring stiffness per unit length.
            fct_idx: If the index of the facet is known, this can be provided as an
                alternative to ``point1`` and ``point2``. Defaults to ``None``.

        Warns:
            If the line spring is added after generating a mesh, the mesh will need to
            be regenerated prior to creating a
            :class:`~planestress.analysis.PlaneStress` object.

        Returns:
            Line spring boundary condition object.

        Example:
            TODO.
        """
        # get the facet index
        if not fct_idx:
            fct_idx = self.find_facet_index(point1=point1, point2=point2)

        # add the facet marker to the specified point
        marker_id = self.add_facet_marker(fct_idx=fct_idx)

        # create line support boundary condition
        line_spring = bc.LineSpring(
            marker_id=marker_id, direction=direction, value=value
        )

        return line_spring

    def add_line_load(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
        direction: str,
        value: float,
        fct_idx: int | None = None,
    ) -> bc.LineLoad:
        """Adds a line load to the geometry.

        Args:
            point1: Point location (``x``, ``y``) of the start of the line load.
            point2: Point location (``x``, ``y``) of the end of the line load.
            direction: Direction of the node load, either ``"x"`` or ``"y"``.
            value: Line load per unit length.
            fct_idx: If the index of the facet is known, this can be provided as an
                alternative to ``point1`` and ``point2``. Defaults to ``None``.

        Warns:
            If the line load is added after generating a mesh, the mesh will need to
            be regenerated prior to creating a
            :class:`~planestress.analysis.PlaneStress` object.

        Returns:
            Line load boundary condition object.

        Example:
            TODO.
        """
        # get the facet index
        if not fct_idx:
            fct_idx = self.find_facet_index(point1=point1, point2=point2)

        # add the facet marker to the specified point
        marker_id = self.add_facet_marker(fct_idx=fct_idx)

        # create line load boundary condition
        line_load = bc.LineLoad(marker_id=marker_id, direction=direction, value=value)

        return line_load

    def create_mesh(
        self,
        mesh_sizes: float | list[float] = 0.0,
        linear: bool = True,
        min_angle: float = 30.0,
        coarse: bool = False,
    ) -> None:
        """Creates and stores a triangular mesh of the geometry.

        Args:
            mesh_sizes: A list of the maximum mesh element areas for each ``polygon`` in
                the ``Geometry`` object. If a list of length 1 or a ``float`` is passed,
                then this one size will be applied to all ``polygons``. A value of ``0``
                removes the area constraint. Defaults to ``0.0``.
            linear: Order of the triangular mesh, if ``True`` generates linear ``Tri3``
                elements, if ``False`` generates quadratic ``Tri6`` elements. Defaults
                to ``True``.
            min_angle: The meshing algorithm adds vertices to the mesh to ensure that no
                angle is smaller than the minimum angle (in degrees, rounded to 1
                decimal place). Defaults to ``30.0``.
            coarse: If set to ``True``, will create a coarse mesh (no area or quality
                constraints). Defaults to ``False``.

        .. admonition:: A note on ``min_angle``

            Note that small angles between input segments cannot be eliminated. If the
            minimum angle is 20.7° or smaller, the triangulation algorithm is
            theoretically guaranteed to terminate (given sufficient precision). The
            algorithm often doesn't terminate for angles greater than 33°. Some meshes
            may require angles well below 20° to avoid problems associated with
            insufficient floating-point precision.
        """
        if isinstance(mesh_sizes, (float, int)):
            mesh_sizes = [mesh_sizes]

        if len(mesh_sizes) == 1:
            mesh_sizes = mesh_sizes * len(self.control_points)

        tri: dict[str, Any] = {}  # create tri dictionary
        tri["vertices"] = [pt.to_tuple() for pt in self.points]  # set points
        tri["vertex_markers"] = self.point_markers  # set point markers
        tri["segments"] = [fct.to_tuple() for fct in self.facets]  # set facets
        tri["segment_markers"] = self.facet_markers  # set facet markers

        if self.holes:
            tri["holes"] = [pt.to_tuple() for pt in self.holes]  # set holes

        # prepare regions
        regions: list[list[float | int]] = []

        for idx, cp in enumerate(self.control_points):
            rg = [cp.x, cp.y, idx, mesh_sizes[idx]]
            regions.append(rg)

        tri["regions"] = regions  # set regions

        # set mesh order
        m_order = "" if linear else "o2"

        # generate mesh
        if coarse:
            mesh = triangulate(tri, "pA{m_order}")
        else:
            mesh = triangulate(tri, f"pq{min_angle:.1f}Aa{m_order}")

        self.mesh = Mesh(
            nodes=np.array(mesh["vertices"], dtype=np.float64),
            elements=np.array(mesh["triangles"], dtype=np.int32),
            attributes=np.array(
                mesh["triangle_attributes"].T[0], dtype=np.int32
            ).tolist(),
            node_markers=np.array(mesh["vertex_markers"].T[0], dtype=np.int32).tolist(),
            segments=np.array(mesh["segments"], dtype=np.int32),
            segment_markers=np.array(
                mesh["segment_markers"].T[0], dtype=np.int32
            ).tolist(),
            linear=linear,
        )

        # re-order mid-nodes if quadratic
        if not linear:
            self.mesh.elements[:, [3, 4, 5]] = self.mesh.elements[:, [5, 3, 4]]

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

        Keyword Args:
            title (str): Plot title. Defaults to ``"Geometry"``.
            labels(list[str]): A list of index labels to plot, can contain any of the
                following: ``"points"``, ``"facets"``, ``"holes"``,
                ``"control_points"``. Defaults to ``["control_points"]``.
            plot_cps (bool): If set to ``True``, plots the control points. Defaults to
                ``True``.
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
        labels: list[str] = kwargs.pop("label", ["control_points"])
        plot_cps: bool = kwargs.pop("plot_cps", True)
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

            # plot the control points
            if plot_cps:
                label = "Control Points"
                for cpts in self.control_points:
                    ax.plot(cpts.x, cpts.y, "bo", markersize=5, label=label)
                    label = None

            # display the labels
            # plot control_point labels
            if "control_points" in labels:
                for idx, pt in enumerate(self.control_points):
                    ax.annotate(str(idx), xy=(pt.x, pt.y), color="b")

            # plot point labels
            if "points" in labels:
                for idx, pt in enumerate(self.points):
                    ax.annotate(str(idx), xy=(pt.x, pt.y), color="r")

            # plot facet labels
            if "facets" in labels:
                for idx, fct in enumerate(self.facets):
                    xy = (fct.pt1.x + fct.pt2.x) / 2, (fct.pt1.y + fct.pt2.y) / 2

                    ax.annotate(str(idx), xy=xy, color="b")

            # plot hole labels
            if "holes" in labels:
                for idx, pt in enumerate(self.holes):
                    ax.annotate(str(idx), xy=(pt.x, pt.y), color="r")

            # plot the load case
            if load_case is not None:
                for boundary_condition in load_case.boundary_conditions:
                    # boundary_condition.plot()
                    print(boundary_condition.marker_id)  # TODO - plot this!

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

         Keyword Args:
            title (str): Plot title. Defaults to ``"Finite Element Mesh"``.
            materials (bool): If set to ``True`` shades the elements with the specified
                material colors. Defaults to ``True``.
            nodes (bool): If set to ``True`` plots the nodes of the mesh. Defaults to
                ``False``.
            node_indexes (bool): If set to ``True``, plots the indexes of each node.
                Defaults to ``False``.
            element_indexes (bool): If set to ``True``, plots the indexes of each
                element. Defaults to ``False``.
            alpha (float): Transparency of the mesh outlines,
                :math:`0 \leq \alpha \leq 1`. Defaults to ``0.5``.
            mask (list[bool] | None): Mask array to mask out triangles, must be same
                length as number of elements in mesh. Defaults to ``None``.
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
        nodes: bool = kwargs.pop("nodes", False)
        node_indexes: bool = kwargs.pop("node_indexes", False)
        element_indexes: bool = kwargs.pop("element_indexes", False)
        alpha: float = kwargs.pop("alpha", 0.5)
        mask: list[bool] | None = kwargs.pop("mask", None)

        if self.mesh is not None:
            return self.mesh.plot_mesh(
                load_case=load_case,
                material_list=self.materials,
                title=title,
                materials=materials,
                nodes=nodes,
                node_indexes=node_indexes,
                element_indexes=element_indexes,
                alpha=alpha,
                mask=mask,
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
        idx: Point index. Defaults to ``None``.
    """

    x: float
    y: float
    tol: int
    idx: int | None = None

    def __post_init__(self) -> None:
        """Point object post init method."""
        self.round()

    def __eq__(
        self,
        other: Point,
    ) -> bool:
        """Override __eq__ method to neglect index.

        Args:
            other: Other ``Point`` to check equality against.

        Returns:
            ``True`` if ``Points`` objects are equal.
        """
        return self.x == other.x and self.y == other.y

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
        pt1: First point in the facet
        pt2: Second point in the facet
    """

    pt1: Point
    pt2: Point

    def __eq__(
        self,
        other: Facet,
    ) -> bool:
        """Override __eq__ method to account for points in either order.

        Args:
            other: Other ``Facet`` to check equality against.

        Returns:
            ``True`` if ``Facet`` objects are equal.
        """
        return (self.pt1 == other.pt1 and self.pt2 == other.pt2) or (
            self.pt1 == other.pt2 and self.pt2 == other.pt1
        )

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
