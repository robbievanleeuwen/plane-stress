"""Class describing a planestress mesh."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import gmsh
import numpy as np
import numpy.typing as npt
import shapely as shapely
from matplotlib import collections

import planestress.pre.geometry as ps_geom
from planestress.analysis.finite_elements.finite_element import (
    LinearLine,
    QuadraticLine,
)
from planestress.analysis.finite_elements.quad4 import Quad4
from planestress.analysis.finite_elements.quad8 import Quad8
from planestress.analysis.finite_elements.quad9 import Quad9
from planestress.analysis.finite_elements.tri3 import Tri3
from planestress.analysis.finite_elements.tri6 import Tri6
from planestress.post.plotting import plotting_context


if TYPE_CHECKING:
    import matplotlib.axes

    from planestress.analysis.finite_elements.finite_element import (
        FiniteElement,
        LineElement,
    )
    from planestress.pre.geometry import CurveLoop, Facet, Point, Surface
    from planestress.pre.load_case import LoadCase
    from planestress.pre.material import Material


@dataclass
class Mesh:
    """Class for a plane-stress mesh.

    Attributes:
        nodes: List of nodes describing the mesh, e.g. ``[[x1, y1], [x2, y2], ... ]``.
        elements: List of finite element objects in the mesh.
        line_elements: List of line element objects in the mesh.
        triangulation: List of indexes defining the triangles in the mesh (quads &
            higher order elements converted to triangles) for plotting purposes.
        materials: List of material objects for each region in the mesh.
        tagged_nodes: List of nodes tagged in the mesh.
        tagged_lines: List of lines tagged in the mesh.
        nodes_str_tree: A :class:`shapely.STRtree` of the nodes in the mesh.
        tagged_nodes_str_tree: A :class:`shapely.STRtree` of the tagged nodes in the
            mesh.
        tagged_lines_str_tree: A :class:`shapely.STRtree` of the tagged lines in the
            mesh.
        bbox: Bounding box of the model geometry
            ``(xmin, ymin, zmin, xmax, ymax, zmax).``
    """

    nodes: npt.NDArray[np.float64] = field(
        init=False, default_factory=lambda: np.array([])
    )
    elements: list[FiniteElement] = field(init=False, default_factory=list)
    line_elements: list[LineElement] = field(init=False, default_factory=list)
    triangulation: list[tuple[int, int, int]] = field(init=False, default_factory=list)
    materials: list[Material] = field(init=False, default_factory=list)
    tagged_nodes: list[TaggedNode] = field(init=False, default_factory=list)
    tagged_lines: list[TaggedLine] = field(init=False, default_factory=list)
    nodes_str_tree: shapely.STRtree = field(init=False)
    tagged_nodes_str_tree: shapely.STRtree = field(init=False)
    tagged_lines_str_tree: shapely.STRtree = field(init=False)
    bbox: tuple[float, float, float, float, float, float] = field(init=False)

    def create_mesh(
        self,
        points: list[Point],
        facets: list[Facet],
        curve_loops: list[CurveLoop],
        surfaces: list[Surface],
        materials: list[Material],
        embedded_geometry: list[Point | Facet],
        mesh_sizes: list[float],
        quad_mesh: bool,
        mesh_order: int,
        serendipity: bool,
        mesh_algorithm: int,
        subdivision_algorithm: int,
        fields: list[Field],
        verbosity: int = 0,
    ) -> None:
        """Creates a mesh from geometry using gmsh.

        Args:
            points: List of ``Point`` objects.
            facets: List of ``Facet`` objects.
            curve_loops: List of ``CurveLoop`` objects.
            surfaces: List of ``Surface`` objects.
            materials: A list of ``Material`` objects for each region in the mesh.
            embedded_geometry: List of embedded points and lines.
            mesh_sizes: A list of the characteristic mesh lengths for each region in the
                mesh.
            quad_mesh: If set to ``True``, recombines the triangular mesh to create
                quadrilaterals.
            mesh_order: Order of the mesh, ``1`` - linear or ``2`` - quadratic.
            serendipity: If set to ``True``, creates serendipity elements for
                quadrilateral meshes.
            mesh_algorithm: Gmsh mesh algorithm, see
                https://gmsh.info/doc/texinfo/gmsh.html#index-Mesh_002eAlgorithm
            subdivision_algorithm: Gmsh subdivision algorithm, see
                https://gmsh.info/doc/texinfo/gmsh.html#index-Mesh_002eSubdivisionAlgorithm
            fields: A list of ``Field`` objects, describing mesh refinement fields.
            verbosity: Gmsh verbosity level, see
                https://gmsh.info/doc/texinfo/gmsh.html#index-General_002eVerbosity.
                Defaults to ``0``.
        """
        # save materials
        self.materials = materials

        # init gmsh
        gmsh.initialize()

        # set verbosity
        gmsh.option.set_number("General.Verbosity", verbosity)

        # init model
        gmsh.model.add("plane-stress")

        # set mesh algorithm
        gmsh.option.set_number("Mesh.Algorithm", mesh_algorithm)

        # set mesh recombine
        if quad_mesh:
            gmsh.option.set_number("Mesh.RecombineAll", 1)
        else:
            gmsh.option.set_number("Mesh.RecombineAll", 0)

        # set serendipity
        if serendipity:
            gmsh.option.set_number("Mesh.SecondOrderIncomplete", 1)
        else:
            gmsh.option.set_number("Mesh.SecondOrderIncomplete", 0)

        # set subdivision algorithm
        gmsh.option.set_number("Mesh.SubdivisionAlgorithm", subdivision_algorithm)

        # build gmsh geometry
        # add points to gmsh geometry
        for point in points:
            # determine mesh size (note surface idxs start at 1)
            mesh_size_list = [mesh_sizes[idx - 1] for idx in point.poly_idxs]
            mesh_size = min(mesh_size_list)  # take the minimum mesh size
            gmsh.model.geo.add_point(
                x=point.x, y=point.y, z=0.0, meshSize=mesh_size, tag=point.idx
            )

        # add facets (lines) to gmsh geometry
        for facet in facets:
            gmsh.model.geo.add_line(
                startTag=facet.pt1.idx, endTag=facet.pt2.idx, tag=facet.idx
            )

        # add curve loops (line sequences) to gmsh geometry
        for curve_loop in curve_loops:
            curve_tags = [facet.idx for facet in curve_loop.facets]
            gmsh.model.geo.add_curve_loop(
                curveTags=curve_tags, tag=curve_loop.idx, reorient=True
            )

        # add surfaces to gmsh geometry
        for surface in surfaces:
            wire_tags = [curve_loop.idx for curve_loop in surface.curve_loops]
            gmsh.model.geo.add_plane_surface(wireTags=wire_tags, tag=surface.idx)

        # synchronize gmsh CAD entities
        gmsh.model.geo.synchronize()

        # embed points and lines
        for geo in embedded_geometry:
            if isinstance(geo, ps_geom.Point):
                # get mesh size if not specified
                if geo.mesh_size is None:
                    mesh_size = mesh_sizes[geo.poly_idxs[0] - 1]
                else:
                    mesh_size = geo.mesh_size

                # add point to geometry
                pt = gmsh.model.geo.add_point(
                    x=geo.x, y=geo.y, z=0.0, meshSize=mesh_size
                )

                # synchronize model
                gmsh.model.geo.synchronize()

                # embed point
                gmsh.model.mesh.embed(dim=0, tags=[pt], inDim=2, inTag=geo.poly_idxs[0])
            if isinstance(geo, ps_geom.Facet):
                # get mesh size if not specified (both points will be identical)
                if geo.pt1.mesh_size is None:
                    mesh_size = mesh_sizes[geo.pt1.poly_idxs[0] - 1]
                else:
                    mesh_size = geo.pt1.mesh_size

                # add points and line to geometry
                pt1 = gmsh.model.geo.add_point(
                    x=geo.pt1.x, y=geo.pt1.y, z=0.0, meshSize=mesh_size
                )
                pt2 = gmsh.model.geo.add_point(
                    x=geo.pt2.x, y=geo.pt2.y, z=0.0, meshSize=mesh_size
                )
                ln = gmsh.model.geo.add_line(startTag=pt1, endTag=pt2)

                # synchronize model
                gmsh.model.geo.synchronize()

                # embed points and line
                gmsh.model.mesh.embed(
                    dim=0, tags=[pt1, pt2], inDim=2, inTag=geo.pt1.poly_idxs[0]
                )
                gmsh.model.mesh.embed(
                    dim=1, tags=[ln], inDim=2, inTag=geo.pt1.poly_idxs[0]
                )

        # check surface orientation:
        # list describing if surface is correctly oriented
        surface_orientated: list[bool] = []

        for _, tag in gmsh.model.get_entities(dim=2):
            normal = gmsh.model.get_normal(tag, (0, 0))

            # if surface is incorrectly oriented, re-orient
            if normal[2] < 0:
                surface_orientated.append(False)
            else:
                surface_orientated.append(True)

        # calculate bounding box
        self.bbox = gmsh.model.get_bounding_box(dim=-1, tag=-1)

        # apply fields
        field_tags = []

        for fld in fields:
            field_tag = fld.apply_field()
            field_tags.append(field_tag)

        # set background mesh
        if len(field_tags) > 0:
            min_tag = gmsh.model.mesh.field.add(fieldType="Min")
            gmsh.model.mesh.field.set_numbers(
                tag=min_tag, option="FieldsList", values=field_tags
            )

            gmsh.model.mesh.field.set_as_background_mesh(tag=min_tag)

        # generate 2D mesh
        gmsh.model.mesh.generate(2)

        # set mesh order
        gmsh.model.mesh.set_order(order=mesh_order)

        # view model - for debugging
        # gmsh.fltk.run()

        # save mesh to self
        self.save_mesh(materials=materials, surface_orientated=surface_orientated)

        # clean-up
        gmsh.finalize()

        # create triangulation for plotting purpose
        self.create_triangulation()

    def save_mesh(
        self,
        materials: list[Material],
        surface_orientated: list[bool],
    ) -> None:
        """Saves the generated gmsh to the ``Mesh`` object.

        Args:
            materials: List of material objects.
            surface_orientated: List describing if surface is correctly oriented.

        Raises:
            ValueError: If there is an unsupported gmsh element type in the mesh.
        """
        # reset mesh
        self.nodes = np.array([])
        self.elements = []
        self.line_elements = []
        self.tagged_nodes = []
        self.tagged_lines = []

        # save all nodes
        node_tags, node_coords, _ = gmsh.model.mesh.get_nodes()
        node_coords = np.reshape(node_coords, (len(node_tags), 3))
        self.nodes = np.array(node_coords[:, :2], dtype=np.float64)

        # create STRtree of nodes
        self.nodes_str_tree = shapely.STRtree(
            geoms=[shapely.Point(node[0], node[1]) for node in self.nodes]
        )

        # save all finite elements
        el_idx: int = 0
        el_obj: type

        # loop through all surface entities
        for _, surf_tag in gmsh.model.get_entities(dim=2):
            mat = materials[int(surf_tag) - 1]  # get material for current surface

            # get elements for current surface
            (
                el_types,
                el_tags_by_type,
                el_node_tags_by_type,
            ) = gmsh.model.mesh.get_elements(dim=2, tag=surf_tag)

            # for each element type
            for el_type, el_tags, el_node_tags_list in zip(
                el_types, el_tags_by_type, el_node_tags_by_type
            ):
                # get number of elements
                num_elements = int(len(el_tags))

                # tri3 elements
                if el_type == 2:
                    # assign element object
                    el_obj = Tri3
                    num_nodes = 3
                # quad4 elements
                elif el_type == 3:
                    # assign element object
                    el_obj = Quad4
                    num_nodes = 4
                # tri6 elements
                elif el_type == 9:
                    # assign element object
                    el_obj = Tri6
                    num_nodes = 6
                # quad9 elements
                elif el_type == 10:
                    # assign element object
                    el_obj = Quad9
                    num_nodes = 9
                # quad8 elements
                elif el_type == 16:
                    # assign element object
                    el_obj = Quad8
                    num_nodes = 8
                else:
                    raise ValueError(f"Unsupported gmsh element type: type {el_type}.")

                # reshape node tags list
                el_node_tags_list = np.reshape(
                    el_node_tags_list, (num_elements, num_nodes)
                )

                # loop through each element
                for el_tag, el_node_tags in zip(el_tags, el_node_tags_list):
                    # convert gmsh tag to node index
                    node_idxs = np.array(el_node_tags - 1, dtype=np.int32)

                    # reverse node indexes if surface not oriented
                    surface_idx = int(surf_tag) - 1
                    orientation = surface_orientated[surface_idx]

                    coords = self.nodes[node_idxs, :].transpose()

                    # add element to list of elements
                    self.elements.append(
                        el_obj(
                            el_idx=el_idx,
                            el_tag=el_tag,
                            coords=coords,
                            node_idxs=node_idxs.tolist(),
                            material=mat,
                            orientation=orientation,
                        )
                    )
                    el_idx += 1

        # save all line elements
        line_idx: int = 0
        line_obj: type
        (
            line_types,
            line_tags_by_type,
            line_node_tags_by_type,
        ) = gmsh.model.mesh.get_elements(dim=1)

        # for each line type
        for line_type, line_tags, line_node_tags_list in zip(
            line_types, line_tags_by_type, line_node_tags_by_type
        ):
            # linear line elements
            if line_type == 1:
                # reshape node tags list
                num_lines = int(len(line_tags))
                line_node_tags_list = np.reshape(line_node_tags_list, (num_lines, 2))

                # assign element object
                line_obj = LinearLine
            # quadratic line elements
            elif line_type == 8:
                # reshape node tags list
                num_lines = int(len(line_tags))
                line_node_tags_list = np.reshape(line_node_tags_list, (num_lines, 3))

                # assign element object
                line_obj = QuadraticLine
            else:
                raise ValueError(f"Unsupported gmsh line type: type {line_type}.")

            # loop through each line element
            for line_tag, line_node_tags in zip(line_tags, line_node_tags_list):
                # convert gmsh tag to node index
                node_idxs = np.array(line_node_tags - 1, dtype=np.int32)
                coords = self.nodes[node_idxs, :].transpose()

                # add element to list of elements
                self.line_elements.append(
                    line_obj(
                        line_idx=line_idx,
                        line_tag=line_tag,
                        coords=coords,
                        node_idxs=node_idxs.tolist(),
                    )
                )
                line_idx += 1

        # save node entities
        for _, tag in gmsh.model.get_entities(dim=0):
            # get current node entitiy
            node_tag, node_coords, _ = gmsh.model.mesh.getNodes(dim=0, tag=tag)

            # get index of node in self.nodes
            node_idx = self.get_node_idx_by_coordinates(
                x=node_coords[0], y=node_coords[1]
            )

            # add to list of tagged nodes
            self.tagged_nodes.append(
                TaggedNode(
                    node_idx=node_idx,
                    tag=node_tag[0],
                    x=node_coords[0],
                    y=node_coords[1],
                )
            )

        # create STRtree of tagged nodes
        self.tagged_nodes_str_tree = shapely.STRtree(
            geoms=[shapely.Point(node.x, node.y) for node in self.tagged_nodes]
        )

        # save line entities
        for _, tag in gmsh.model.get_entities(dim=1):
            # get node tags that define line
            _, node_tags = gmsh.model.get_adjacencies(dim=1, tag=tag)

            # build list of tagged nodes
            tagged_nodes = []
            for node_tag in node_tags:
                tagged_nodes.append(self.get_tagged_node(tag=node_tag))

            # get element tags of line elements along the line
            _, line_tags_by_type, _ = gmsh.model.mesh.get_elements(dim=1, tag=tag)

            # build list of line elements
            line_list = []
            for line_tags in line_tags_by_type:
                for line_tag in line_tags:
                    line_list.append(self.get_line_element_by_tag(tag=line_tag))

            # add to list of tagged lines
            self.tagged_lines.append(
                TaggedLine(tag=tag, tagged_nodes=tagged_nodes, elements=line_list)
            )

        # create STRtree of tagged lines
        self.tagged_lines_str_tree = shapely.STRtree(
            geoms=[line.to_shapely_line() for line in self.tagged_lines]
        )

    def create_triangulation(self) -> None:
        """Creates a list of triangle indexes that are used for plotting purposes.

        Elements that are not three-noded triangles need to be further subdivided into
        triangles to allow for the use of triangular plotting functions in
        post-processing.
        """
        # reset triangles
        self.triangulation = []

        for element in self.elements:
            self.triangulation.extend(element.get_triangulation())

    def num_nodes(self) -> int:
        """Returns the number of nodes in the mesh.

        Returns:
            Number of nodes in the mesh.
        """
        return len(self.nodes)

    def check_nearest_tol(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
    ) -> bool:
        """Checks if the point 1 is relatively close to point 2.

        The acceptable tolerance is taken to be 1% of the minimum dimension of the
        bounding box.

        Args:
            point1: Location of point 1 (``x``, ``y``).
            point2: Location of point 2 (``x``, ``y``).

        Returns:
            ``True`` if point 1 is relatively close to point 2.
        """
        x = self.bbox[3] - self.bbox[0]
        y = self.bbox[4] - self.bbox[1]
        tol = 0.01 * min(x, y)

        if abs(point1[0] - point2[0]) > tol or abs(point1[1] - point2[1]) > tol:
            return False
        else:
            return True

    def get_node_idx_by_coordinates(
        self,
        x: float,
        y: float,
    ) -> int:
        """Returns the node index at or nearest to the point (``x``, ``y``).

        Args:
            x: ``x`` location of the node to find.
            y: ``y`` location of the node to find.

        Raises:
            ValueError: If the point is not close to a node.

        Returns:
            Index of the node closest to (``x``, ``y``).
        """
        # get node index
        idx = self.nodes_str_tree.nearest(geometry=shapely.Point(x, y))

        # check we are close to the desired node
        node = self.nodes[idx]

        if not self.check_nearest_tol(point1=(node[0], node[1]), point2=(x, y)):
            raise ValueError(
                f"The point ({x}, {y}) is not close to a node in the mesh. The nearest "
                f"node is located at {node}."
            )

        return cast(int, idx)

    def get_tagged_node(
        self,
        tag: int,
    ) -> TaggedNode:
        """Returns a ``TaggedNode`` given a node tag.

        Args:
            tag: Node tag.

        Raises:
            ValueError: If there is no ``TaggedNode`` with a tag equal to ``tag``.

        Returns:
            Tagged node identified by ``tag``.
        """
        for tg in self.tagged_nodes:
            if tg.tag == tag:
                return tg
        else:
            raise ValueError(f"Cannot find TaggedNode with tag {tag}.")

    def get_tagged_node_by_coordinates(
        self,
        x: float,
        y: float,
    ) -> TaggedNode:
        """Returns a ``TaggedNode`` at or nearest to the point (``x``, ``y``).

        Args:
            x: ``x`` location of the tagged node to find.
            y: ``y`` location of the tagged node to find.

        Raises:
            ValueError: If the point is not close to a tagged node.

        Returns:
            Tagged node closest to (``x``, ``y``).
        """
        idx = self.tagged_nodes_str_tree.nearest(geometry=shapely.Point(x, y))

        # check we are close to the desired tagged node
        node = self.tagged_nodes[cast(int, idx)]

        if not self.check_nearest_tol(point1=(node.x, node.y), point2=(x, y)):
            raise ValueError(
                f"The point ({x}, {y}) is not close to a tagged node in the mesh. The "
                f"nearest tagged node is located at {node}."
            )

        return self.tagged_nodes[cast(int, idx)]

    def get_tagged_line(
        self,
        tag: int,
    ) -> TaggedLine:
        """Returns a ``TaggedLine`` given a line tag.

        Args:
            tag: Line tag.

        Raises:
            ValueError: If there is no ``TaggedLine`` with a tag equal to ``tag``.

        Returns:
            Tagged line identified by ``tag``.
        """
        for tg in self.tagged_lines:
            if tg.tag == tag:
                return tg
        else:
            raise ValueError(f"Cannot find TaggedLine with tag {tag}.")

    def get_tagged_line_by_coordinates(
        self,
        point1: tuple[float, float],
        point2: tuple[float, float],
    ) -> TaggedLine:
        """Returns a ``TaggedLine`` closest to the line defined by two points.

        Raises:
            ValueError: If the line is not close to a tagged line.

        Args:
            point1: First point (``x``, ``y``) of the tagged line to find.
            point2: Second point (``x``, ``y``) of the tagged line to find.

        Returns:
            Tagged line closest to the line defined by two points.
        """
        mid_point = shapely.Point(
            0.5 * (point1[0] + point2[0]), 0.5 * (point1[1] + point2[1])
        )
        idx = self.tagged_lines_str_tree.nearest(geometry=mid_point)

        # check we are close to the desired line
        line = self.tagged_lines[cast(int, idx)]
        ln_mid = 0.5 * (line.tagged_nodes[0].x + line.tagged_nodes[1].x), 0.5 * (
            line.tagged_nodes[0].y + line.tagged_nodes[1].y
        )

        if not self.check_nearest_tol(point1=ln_mid, point2=(mid_point.x, mid_point.y)):
            raise ValueError(
                f"The line with mid-point at ({mid_point}) is not close to a mid-point "
                f"of a tagged line in the mesh. The nearest tagged line has a "
                f"mid-point that is located at {ln_mid}."
            )

        return self.tagged_lines[cast(int, idx)]

    def get_line_element_by_tag(
        self,
        tag: int,
    ) -> LineElement:
        """Returns a ``LineElement`` given an element tag.

        Args:
            tag: Line element tag.

        Raises:
            ValueError: If there is no ``LineElement`` with a tag equal to ``tag``.

        Returns:
            Line element identified by ``tag``.
        """
        for line in self.line_elements:
            if line.line_tag == tag:
                return line
        else:
            raise ValueError(f"Cannot find LineElement with tag {tag}.")

    def get_finite_element_by_tag(
        self,
        tag: int,
    ) -> FiniteElement:
        """Returns a ``FiniteElement`` given an element tag.

        Args:
            tag: Finite element tag.

        Raises:
            ValueError: If there is no ``FiniteElement`` with a tag equal to ``tag``.

        Returns:
            Finite element identified by ``tag``.
        """
        for el in self.elements:
            if el.el_tag == tag:
                return el
        else:
            raise ValueError(f"Cannot find FiniteElement with tag {tag}.")

    def plot_mesh(
        self,
        load_case: LoadCase | None,
        title: str,
        materials: bool,
        node_indexes: bool,
        element_indexes: bool,
        alpha: float,
        ux: npt.NDArray[np.float64] | None = None,
        uy: npt.NDArray[np.float64] | None = None,
        **kwargs: Any,
    ) -> matplotlib.axes.Axes:
        r"""Plots the finite element mesh.

        Optionally also renders the boundary conditions of a load case if provided. Also
        plots a deformed mesh if ``ux`` and/or ``uy`` is provided. In this case, the
        undeformed mesh is also plotted with ``alpha=0.2``, ``materials`` is set to
        ``False`` and ``load_case`` is set to ``None``.

        Args:
            load_case: Plots the boundary conditions within a load case if provided.
                Defaults to ``None``.
            title: Plot title.
            materials: If set to ``True`` shades the elements with the specified
                material colors.
            node_indexes: If set to ``True``, plots the indexes of each node.
            element_indexes: If set to ``True``, plots the indexes of each element.
            alpha: Transparency of the mesh outlines, :math:`0 \leq \alpha \leq 1`.
            ux: Deformation component in the ``x`` direction. Defaults to ``None``.
            uy: Deformation component in the ``y`` direction. Defaults to ``None``.
            kwargs (dict[str, Any]): Other keyword arguments are passed to
                :meth:`~planestress.post.plotting.plotting_context`.

        Returns:
            Matplotlib axes object.
        """
        # create plot and setup the plot
        with plotting_context(title=title, **kwargs) as (_, ax):
            assert ax

            # get number of unique materials
            unique_materials = list(set(self.materials))
            num_materials = len(unique_materials)

            # generate an array of polygon vertices and colors for each unique material
            verts: list[list[npt.NDArray[np.float64]]] = [
                [] for _ in range(num_materials)
            ]
            colors: list[list[str | float]] = [[] for _ in range(num_materials)]

            for element in self.elements:
                idx = unique_materials.index(element.material)  # get material index

                # get vertices - take care to create new array so as not to change vals
                idxs, coords = element.get_polygon_coordinates()
                coords = np.array(coords).transpose()

                # add displacements
                if ux is not None:
                    coords[:, 0] += ux[idxs]

                if uy is not None:
                    coords[:, 1] += uy[idxs]

                verts[idx].append(coords)

                # add colors
                colors[idx].append(element.material.color)

            # generate collection of polygons for each material
            for idx in range(num_materials):
                # get face color
                if materials:
                    fcs = colors[idx]
                else:
                    fcs = [1.0, 1.0, 1.0, 0.0]

                col = collections.PolyCollection(
                    verts[idx],
                    edgecolors=[0.0, 0.0, 0.0, alpha],
                    facecolors=fcs,
                    linewidth=0.5,
                    label=self.materials[idx].name,
                )
                ax.add_collection(collection=col)

            # if deformed shape, plot the original mesh
            if ux is not None or uy is not None:
                verts_orig = []

                for element in self.elements:
                    # add vertices
                    _, coords = element.get_polygon_coordinates()
                    verts_orig.append(np.array(coords).transpose())

                col = collections.PolyCollection(
                    verts_orig,
                    edgecolors=[1.0, 0.0, 0.0, 0.1],
                    facecolors=[1.0, 1.0, 1.0, 0.0],
                    linewidth=0.5,
                    linestyle="dashed",
                )
                ax.add_collection(collection=col)

            ax.autoscale_view()

            # if materials, display the legend
            if materials:
                ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                )

            # node numbers
            if node_indexes:
                for idx, pt in enumerate(self.nodes):
                    ax.annotate(
                        str(idx), xy=(pt[0], pt[1]), color="r", ha="center", va="center"
                    )

            # element numbers
            if element_indexes:
                for el in self.elements:
                    pt = np.average(el.coords, axis=1)
                    ax.annotate(
                        str(el.el_idx),
                        xy=(pt[0], pt[1]),
                        color="b",
                        ha="center",
                        va="center",
                    )

            # plot the load case
            if load_case is not None:
                for boundary_condition in load_case.boundary_conditions:
                    # boundary_condition.plot()
                    print(boundary_condition)  # TODO - plot this!

        return ax


class Field:
    """Abstract class for a mesh refinement field."""

    def apply_field(self) -> int:
        """Applies the field and returns the field tag.

        Raises:
            NotImplementedError: If this method hasn't been implemented.
        """
        raise NotImplementedError


class DistanceField(Field):
    """Class for a distance mesh refinement field."""

    def __init__(
        self,
        min_size: float,
        max_size: float,
        min_distance: float,
        max_distance: float,
        point_tags: list[int] | None = None,
        line_tags: list[int] | None = None,
        sampling: int = 20,
    ) -> None:
        """Inits the DistanceField class.

        Args:
            min_size: _description_
            max_size: _description_
            min_distance: _description_
            max_distance: _description_
            point_tags: _description_. Defaults to ``None``.
            line_tags: _description_. Defaults to ``None``.
            sampling: _description_. Defaults to ``20``.
        """
        self.min_size = min_size
        self.max_size = max_size
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.point_tags = [] if point_tags is None else point_tags
        self.line_tags = [] if line_tags is None else line_tags
        self.sampling = sampling

    def apply_field(self) -> int:
        """Applies the distance field and returns the field tag.

        Returns:
            Field tag.
        """
        # add distance field
        dist_tag = gmsh.model.mesh.field.add(fieldType="Distance")
        gmsh.model.mesh.field.set_numbers(
            tag=dist_tag, option="PointsList", values=self.point_tags
        )
        gmsh.model.mesh.field.set_numbers(
            tag=dist_tag, option="CurvesList", values=self.line_tags
        )
        gmsh.model.mesh.field.set_number(
            tag=dist_tag, option="Sampling", value=self.sampling
        )

        # add threshold field
        field_tag = gmsh.model.mesh.field.add(fieldType="Threshold")
        gmsh.model.mesh.field.set_number(
            tag=field_tag, option="InField", value=dist_tag
        )
        gmsh.model.mesh.field.set_number(
            tag=field_tag, option="SizeMin", value=self.min_size
        )
        gmsh.model.mesh.field.set_number(
            tag=field_tag, option="SizeMax", value=self.max_size
        )
        gmsh.model.mesh.field.set_number(
            tag=field_tag, option="DistMin", value=self.min_distance
        )
        gmsh.model.mesh.field.set_number(
            tag=field_tag, option="DistMax", value=self.max_distance
        )

        return cast(int, field_tag)


class BoxField(Field):
    """Class for a box mesh refinement field."""

    def __init__(
        self,
        extents: tuple[float, float, float, float],
        min_size: float,
        max_size: float,
        thickness: float,
    ) -> None:
        """Inits the BoxField class.

        Args:
            extents: _description_
            min_size: _description_
            max_size: _description_
            thickness: _description_
        """
        self.extents = extents
        self.min_size = min_size
        self.max_size = max_size
        self.thickness = thickness

    def apply_field(self) -> int:
        """Applies the box field and returns the field tag.

        Returns:
            Field tag.
        """
        # add box field
        box_tag = gmsh.model.mesh.field.add(fieldType="Box")
        gmsh.model.mesh.field.set_number(tag=box_tag, option="VIn", value=self.min_size)
        gmsh.model.mesh.field.set_number(
            tag=box_tag, option="VOut", value=self.max_size
        )
        gmsh.model.mesh.field.set_number(
            tag=box_tag, option="XMin", value=self.extents[0]
        )
        gmsh.model.mesh.field.set_number(
            tag=box_tag, option="XMax", value=self.extents[1]
        )
        gmsh.model.mesh.field.set_number(
            tag=box_tag, option="YMin", value=self.extents[2]
        )
        gmsh.model.mesh.field.set_number(
            tag=box_tag, option="YMax", value=self.extents[3]
        )
        gmsh.model.mesh.field.set_number(
            tag=box_tag, option="Thickness", value=self.thickness
        )

        return cast(int, box_tag)


@dataclass
class TaggedEntity:
    """Class describing a tagged entity in the mesh.

    Args:
        tag: Gmsh tag.
    """

    tag: int


@dataclass
class TaggedNode(TaggedEntity):
    """Class describing a tagged node.

    Args:
        tag: Gmsh node tag.
        node_idx: Index of node in mesh.
        x: ``x`` location of the node.
        y: ``y`` location of the node.
    """

    tag: int
    node_idx: int
    x: float
    y: float

    def to_shapely_point(self) -> shapely.Point:
        """Converts the tagged node to a ``shapely`` ``Point`` object.

        Returns:
            ``TaggedNode`` as a :class:`shapely.Point`.
        """
        return shapely.Point(self.x, self.y)


@dataclass
class TaggedLine(TaggedEntity):
    """Class describing a tagged line.

    Args:
        tag: Gmsh line tag.
        tagged_nodes: List ``TaggedNode`` objects at each end of line.
        elements: List of ``FiniteElement`` objects along the line.
    """

    tag: int
    tagged_nodes: list[TaggedNode]
    elements: list[LineElement]

    def to_shapely_line(self) -> shapely.LineString:
        """Converts the tagged line to a ``shapely`` ``Line`` object.

        Returns:
            ``TaggedLine`` as a :class:`shapely.LineString`.
        """
        return shapely.LineString(
            [
                self.tagged_nodes[0].to_shapely_point(),
                self.tagged_nodes[1].to_shapely_point(),
            ]
        )
