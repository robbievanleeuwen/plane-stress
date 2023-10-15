"""Classes relating to planestress geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from shapely import GeometryCollection, LineString, MultiPolygon, Polygon
from triangle import triangulate

from planestress.post.post import plotting_context
from planestress.pre.material import Material
from planestress.pre.mesh import Mesh


if TYPE_CHECKING:
    import matplotlib.axes


class Geometry:
    """Class describing a geometric region."""

    def __init__(
        self,
        polygons: Polygon | MultiPolygon,
        materials: Material | list[Material],
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

        # save input data
        self.polygons = polygons
        self.materials = materials
        self.tol = tol

        # allocate points, facets, holes, control_points
        self.points: list[Point] = []
        self.facets: list[Facet] = []
        self.holes: list[Point] = []
        self.control_points: list[Point] = []

        # allocate mesh
        self.mesh: Mesh | None = None

        # compile the geometry into points, facets, holes and control_points
        self.compile_geometry()

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
            new_pt = Point(x=coords[0], y=coords[1])
            new_pt.round(tol=self.tol)
            pt_list.append(new_pt)

        # create perimeter facets
        fct_list.extend(self.create_facet_list(pt_list=pt_list))

        # construct holes, for each interior (hole) region
        for hl in polygon.interiors:
            int_pt_list: list[Point] = []

            # create hole (note shapely doubles up first & last point)
            for coords in hl.coords[:-1]:
                new_pt = Point(x=coords[0], y=coords[1])
                new_pt.round(tol=self.tol)
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
            hl_list.append(Point(x=hl_pt_coords[0][0], y=hl_pt_coords[0][1]))

        # construct control point
        cp_pt_coords = polygon.representative_point().coords
        cp_pt = Point(x=cp_pt_coords[0][0], y=cp_pt_coords[0][1])

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

    def align_to(self) -> None:
        """a"""
        raise NotImplementedError

    def shift_section(self) -> None:
        """compound & geom"""
        raise NotImplementedError

    def rotate_section(self) -> None:
        """compound & geom"""
        raise NotImplementedError

    def mirror_section(self) -> None:
        """compound & geom"""
        raise NotImplementedError

    def calcalate_extents(self) -> tuple[float, float, float, float]:
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

    def __or__(self) -> None:
        """a"""
        raise NotImplementedError

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
                msg = "Cannot perform 'difference' on these two objects: "
                msg += f"{self} - {other}"
                raise ValueError(msg)
            # polygon or multipolygon object
            elif isinstance(new_polygon, Polygon | MultiPolygon):
                return Geometry(
                    polygons=new_polygon, materials=self.materials, tol=self.tol
                )
            else:
                msg = "Cannot perform 'difference' on these two objects: "
                msg += f"{self} - {other}"
                raise ValueError(msg)
        except Exception as e:
            msg = "Cannot perform 'difference' on these two objects: "
            msg += f"{self} - {other}"
            raise ValueError(msg) from e

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

    def __and__(self) -> None:
        """a"""
        raise NotImplementedError

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
                if isinstance(item, Polygon | MultiPolygon):
                    acc.append(item)

            if len(acc) == 0:
                return Polygon()
            elif len(acc) == 1:
                return acc[0]
            else:
                return MultiPolygon(polygons=acc)
        elif isinstance(input_geom, Point | LineString):
            return Polygon()
        else:
            return Polygon()

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
            order: Order of triangular mesh, if ``True`` gives linear elements and if
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
            attributes=np.array(mesh["triangle_attributes"].T[0], dtype=np.int32),
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
    idx: int | None = None

    def round(
        self,
        tol: int,
    ) -> None:
        """Rounds the point to ``tol`` digits."""

        self.x = round(self.x, tol)
        self.y = round(self.y, tol)

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
