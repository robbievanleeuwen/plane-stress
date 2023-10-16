"""Classes relating to planestress geometry."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from shapely import GeometryCollection, LineString, MultiPolygon, Polygon, affinity
from triangle import triangulate

import planestress.pre.boundary_condition as bc
from planestress.post.plotting import plotting_context
from planestress.pre.material import DEFAULT_MATERIAL, Material
from planestress.pre.mesh import Mesh


if TYPE_CHECKING:
    import matplotlib.axes


class Geometry:
    """Class describing a geometric region."""

    def __init__(
        self,
        polygons: Polygon | MultiPolygon,
        materials: Material | list[Material] = DEFAULT_MATERIAL,
        tol: int = 12,
    ) -> None:
        """Inits the Geometry class.

        Note ensure length of materials = number of polygons
        """
        # convert polygon to multipolygon
        if isinstance(polygons, Polygon):
            polygons = MultiPolygon(polygons=[polygons])

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

            # add facets to the global list
            self.facets.extend(poly_facets)

            # add holes to list of multipolygon holes
            self.holes.extend(poly_holes)

            # add control points to the global list
            self.control_points.append(poly_cp)

        # assign point indices
        for idx, pt in enumerate(self.points):
            pt.idx = idx

    def compile_polygon(
        self,
        polygon: Polygon,
    ) -> tuple[list[Point], list[Facet], list[Point], Point]:
        """Create a list of points, facets and holes + control point given a Polygon."""
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
            hl_poly = Polygon(int_pt_list_tup)

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
        """Creates a closed list of facets from a list of points."""
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
        """Aligns the geometry to another ``Geometry`` or point."""
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
        """Aligns the geometry to a centre point."""
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
        """Shifts the geometry by (``x``, ``y``)."""
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
        """Rotates the geometry."""
        return Geometry(
            polygons=affinity.rotate(
                geom=self.polygons,
                angle=angle,
                origin=rot_point,
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
        """Compound & geom."""
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
        """Performs a difference operation using the ``|`` operator.

        Note - applies material of first geometry to entire new geom.
        """
        try:
            new_polygon = self.filter_non_polygons(
                input_geom=self.polygons | other.polygons
            )

            return Geometry(
                polygons=new_polygon, materials=self.materials[0], tol=self.tol
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

        Note about combining resulting geometry into one section! Check in materials!
        """
        try:
            new_polygon = self.filter_non_polygons(
                input_geom=self.polygons - other.polygons
            )

            # non-polygon results
            if isinstance(new_polygon, GeometryCollection):
                raise ValueError(
                    f"Cannot perform difference on these two objects: {self} - {other}"
                )
            # polygon or multipolygon object
            elif isinstance(new_polygon, (Polygon, MultiPolygon)):
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

        Keeps the largest tol.
        """
        poly_list: list[Polygon] = []
        mat_list: list[Material] = []

        # loop through each list of polygons and combine
        for poly, mat in zip(self.polygons.geoms, self.materials):
            poly_list.append(poly)
            mat_list.append(mat)

        for poly, mat in zip(other.polygons.geoms, other.materials):
            poly_list.append(poly)
            mat_list.append(mat)

        tol = max(self.tol, other.tol)

        return Geometry(
            polygons=MultiPolygon(polygons=poly_list), materials=mat_list, tol=tol
        )

    def __and__(
        self,
        other: Geometry,
    ) -> Geometry:
        """Performs an intersection operation using the ``&`` operator.

        Note - applies material of first geometry to entire new geom.
        """
        try:
            new_polygon = self.filter_non_polygons(
                input_geom=self.polygons - other.polygons
            )

            return Geometry(
                polygons=new_polygon, materials=self.materials[0], tol=self.tol
            )
        except Exception as exc:
            raise ValueError(
                f"Cannot perform intersection on these two Geometry instances: "
                f"{self} & {other}"
            ) from exc

    @staticmethod
    def filter_non_polygons(
        input_geom: GeometryCollection | LineString | Point | Polygon | MultiPolygon,
    ) -> Polygon | MultiPolygon:
        """Filters shapely geometries to return only polygons.

        Returns a ``Polygon`` or a ``MultiPolygon`` representing any such ``Polygon`` or
        ``MultiPolygon`` that may exist in the ``input_geom``. If ``input_geom`` is a
        ``LineString`` or ``Point``, an empty ``Polygon`` is returned.

        Args:
            input_geom: Shapely geometry to filter

        Returns:
            Filtered polygon
        """
        if isinstance(input_geom, (Polygon, MultiPolygon)):
            return input_geom
        elif isinstance(input_geom, GeometryCollection):
            acc = []

            for item in input_geom.geoms:
                if isinstance(item, (Polygon, MultiPolygon)):
                    acc.append(item)

            if len(acc) == 0:
                return Polygon()
            elif len(acc) == 1:
                return acc[0]
            else:
                return MultiPolygon(polygons=acc)
        elif isinstance(input_geom, (Point, LineString)):
            return Polygon()
        else:
            return Polygon()

    def add_node_support(
        self,
        point: tuple[float, float],
        direction: str,
        value: float = 0.0,
    ) -> bc.NodeSupport:
        """Adds a node support to the geometry."""
        # check to see if a mesh has already been generated
        self.check_boundary_condition_mesh()

        # create point object
        pt = Point(point[0], point[1], tol=self.tol)

        # find point in list of points
        try:
            idx = self.points.index(pt)
        except ValueError as exc:
            raise ValueError("Cannot find the point: {pt} in the geometry.") from exc

        # check to see if point already has custom point marker
        if self.point_markers[idx] == 0:
            # add point marker to point, note custom point marker ids start at 2
            marker_id = max(1, max(self.point_markers)) + 1
            self.point_markers[idx] = marker_id
        else:
            # get marker id
            marker_id = self.point_markers[idx]

        # create node support boundary condition
        ns_bc = bc.NodeSupport(marker_id=marker_id, direction=direction, value=value)

        return ns_bc

    def check_boundary_condition_mesh(self) -> None:
        """Checks to see if a mesh exist before creating a new boundary condition.

        If a mesh exists, warns the user to regenerate the mesh prior to conducting an
        analysis.
        """
        if self.mesh:
            warnings.warn(
                "Please regenerate the mesh prior to creating a PlaneStress object."
            )

    def create_mesh(
        self,
        mesh_sizes: float | list[float] = 0.0,
        linear: bool = True,
        min_angle: float = 30.0,
        coarse: bool = False,
    ) -> None:
        """Creates a triangular mesh of the geometry.

        Args:
            mesh_sizes: A float describing the maximum mesh element area to be used in
                the finite-element mesh for each polygon the ``Geometry`` object. If a
                list of length 1 or a ``float`` is passed, then the one size will be
                applied to all polygons. A value of ``0`` removes the area constraint.
            linear: Order of triangular mesh, if ``True`` gives linear elements and if
                ``False`` gives quadratic elements
            min_angle: The meshing algorithm adds vertices to the mesh to ensure that no
                angle smaller than the minimum angle (in degrees, rounded to 1 decimal
                place). Note that small angles between input segments cannot be
                eliminated. If the minimum angle is 20.7 deg or smaller, the
                triangulation algorithm is theoretically guaranteed to terminate (given
                sufficient precision). The algorithm often doesn't terminate for angles
                greater than 33 deg. Some meshes may require angles well below 20 deg to
                avoid problems associated with insufficient floating-point precision.
            coarse: If set to True, will create a coarse mesh (no area or quality
                constraints)
        """
        if isinstance(mesh_sizes, (float, int)):
            mesh_sizes = [mesh_sizes]

        if len(mesh_sizes) == 1:
            mesh_sizes = mesh_sizes * len(self.control_points)

        tri: dict[str, Any] = {}  # create tri dictionary
        tri["vertices"] = [pt.to_tuple() for pt in self.points]  # set points
        tri["vertex_markers"] = self.point_markers  # set point markers
        tri["segments"] = [fct.to_tuple() for fct in self.facets]  # set facets

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
            linear=linear,
        )

        # re-order mid-nodes if quadratic
        if not linear:
            self.mesh.elements[:, [3, 4, 5]] = self.mesh.elements[:, [5, 3, 4]]

    def plot_geometry(
        self,
        labels: tuple[str] = ("control_points",),
        title: str = "Geometry",
        cp: bool = True,
        legend: bool = True,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """Plots the geometry."""
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
            if cp:
                label = "Control Points"
                for cpts in self.control_points:
                    ax.plot(cpts.x, cpts.y, "bo", markersize=5, label=label)
                    label = None

            # display the labels
            for label in labels:
                # plot control_point labels
                if label == "control_points":
                    for idx, pt in enumerate(self.control_points):
                        ax.annotate(str(idx), xy=(pt.x, pt.y), color="b")

                # plot point labels
                if label == "points":
                    for idx, pt in enumerate(self.points):
                        ax.annotate(str(idx), xy=(pt.x, pt.y), color="r")

                # plot facet labels
                if label == "facets":
                    for idx, fct in enumerate(self.facets):
                        xy = (fct.pt1.x + fct.pt2.x) / 2, (fct.pt1.y + fct.pt2.y) / 2

                        ax.annotate(str(idx), xy=xy, color="b")

                # plot hole labels
                if label == "holes":
                    for idx, pt in enumerate(self.holes):
                        ax.annotate(str(idx), xy=(pt.x, pt.y), color="r")

            # display the legend
            if legend:
                ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        return ax

    def plot_mesh(
        self,
        nodes: bool = False,
        nd_num: bool = False,
        el_num: bool = False,
        materials: bool = False,
        alpha: float = 0.5,
        mask: list[bool] | None = None,
        title: str = "Finite Element Mesh",
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        """Plots the finite element mesh."""
        if self.mesh is not None:
            return self.mesh.plot_mesh(
                material_list=self.materials,
                nodes=nodes,
                nd_num=nd_num,
                el_num=el_num,
                materials=materials,
                alpha=alpha,
                mask=mask,
                title=title,
                **kwargs,
            )
        else:
            raise RuntimeError("Generate a mesh with create_mesh() prior to plotting.")


@dataclass(eq=True)
class Point:
    """Class describing a point in 2D space."""

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
        """Override __eq__ method to neglect index."""
        return self.x == other.x and self.y == other.y

    def round(self) -> None:
        """Rounds the point to ``tol`` digits."""
        self.x = round(self.x, self.tol)
        self.y = round(self.y, self.tol)

    def to_tuple(self) -> tuple[float, float]:
        """Converts the point to a tuple."""
        return self.x, self.y


@dataclass(eq=True)
class Facet:
    """Class describing a facet of a 2D geometry, i.e. an edge."""

    pt1: Point
    pt2: Point

    def to_tuple(self) -> tuple[float, float]:
        """Converts the facet to a tuple."""
        idx_1 = self.pt1.idx
        idx_2 = self.pt2.idx

        if idx_1 is None:
            raise RuntimeError(f"Point 1: {self.pt1} has not been assigned an index.")

        if idx_2 is None:
            raise RuntimeError(f"Point 2: {self.pt2} has not been assigned an index.")

        return idx_1, idx_2

    def update_point(
        self,
        old: Point,
        new: Point,
    ) -> None:
        """If the facet contains the point ``old``, replace with ``new``."""
        if self.pt1 == old:
            self.pt1 = new

        if self.pt2 == old:
            self.pt2 = new
