"""Class describing a planestress mesh."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import gmsh
import numpy as np
import numpy.typing as npt
import shapely as shapely
from matplotlib import collections

import planestress.analysis.finite_element as fe
from planestress.post.plotting import plotting_context


if TYPE_CHECKING:
    import matplotlib.axes

    from planestress.pre.geometry import CurveLoop, Facet, Point, Surface
    from planestress.pre.load_case import LoadCase
    from planestress.pre.material import Material


@dataclass
class Mesh:
    """Class for a plane-stress mesh.

    Attributes:
        nodes: List of nodes describing the mesh, e.g. ``[[x1, y1], [x2, y2], ... ]``.
        elements: List of finite element objects in the mesh.
        tagged_nodes: List of nodes tagged in the mesh.
        tagged_lines: List of lines tagged in the mesh.
        str_tree: A :class:`shapely.STRtree` of the nodes in the mesh.
    """

    nodes: npt.NDArray[np.float64] = field(
        init=False, default_factory=lambda: np.array([])
    )
    elements: list[fe.FiniteElement] = field(init=False, default_factory=list)
    triangulation: list[tuple[int, int, int]] = field(init=False, default_factory=list)
    materials: list[Material] = field(init=False, default_factory=list)
    line_elements: list[fe.LineElement] = field(init=False, default_factory=list)
    tagged_nodes: list[TaggedNode] = field(init=False, default_factory=list)
    tagged_lines: list[TaggedLine] = field(init=False, default_factory=list)
    nodes_str_tree: shapely.STRtree = field(init=False)
    tagged_nodes_str_tree: shapely.STRtree = field(init=False)
    tagged_lines_str_tree: shapely.STRtree = field(init=False)

    def create_mesh(
        self,
        points: list[Point],
        facets: list[Facet],
        curve_loops: list[CurveLoop],
        surfaces: list[Surface],
        mesh_sizes: float | list[float],
        materials: list[Material],
        verbosity: int = 0,
    ) -> None:
        """Creates a mesh using gmsh."""
        # convert mesh_size to an appropriately sized list
        if isinstance(mesh_sizes, (float, int)):
            mesh_sizes = [float(mesh_sizes)] * len(surfaces)

        if len(mesh_sizes) == 1:
            mesh_sizes = mesh_sizes * len(surfaces)

        # check mesh_sizes length
        if len(mesh_sizes) != len(surfaces):
            raise ValueError(
                "Length of 'mesh_sizes' must equal the number of polygons or 1."
            )

        # save materials
        self.materials = materials

        # init gmsh
        gmsh.initialize()

        # set verbosity
        gmsh.option.set_number("General.Verbosity", verbosity)

        # init model
        gmsh.model.add("plane-stress")

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

        # TODO - ADD MESHING OPTIONS!
        # linear/quadratic
        # tri/quad
        # mesh refinement options

        # generate 2D mesh
        gmsh.model.mesh.generate(2)

        # view model - for debugging
        # gmsh.fltk.run()

        # save mesh to self
        self.save_mesh(materials=materials)

        # clean-up
        gmsh.finalize()

        # create triangulation for plotting purpose
        self.create_triangulation()

    def save_mesh(self, materials) -> None:
        """Saves the generated gmsh to the ``Mesh`` object."""
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
        for _, tag in gmsh.model.get_entities(dim=2):
            mat = materials[int(tag) - 1]  # get material for current entity

            # get elements for current entity
            (
                el_types,
                el_tags_by_type,
                el_node_tags_by_type,
            ) = gmsh.model.mesh.get_elements(dim=2, tag=tag)

            # for each element type
            for el_type, el_tags, el_node_tags_list in zip(
                el_types, el_tags_by_type, el_node_tags_by_type
            ):
                # tri3 elements
                if el_type == 2:
                    # reshape node tags list
                    num_elements = int(len(el_tags))
                    el_node_tags_list = np.reshape(el_node_tags_list, (num_elements, 3))
                    # assign element object
                    el_obj = fe.Tri3
                # tri6 elements
                elif el_type == 9:
                    raise NotImplementedError
                else:
                    raise ValueError(f"Unsupported gmsh element type: type {el_type}.")

                # loop through each element
                for el_tag, el_node_tags in zip(el_tags, el_node_tags_list):
                    # convert gmsh tag to node index
                    node_idxs = np.array(el_node_tags - 1, dtype=np.int32)
                    coords = self.nodes[node_idxs, :].transpose()

                    # add element to list of elements
                    self.elements.append(
                        el_obj(
                            el_idx=el_idx,
                            el_tag=el_tag,
                            coords=coords,
                            node_idxs=node_idxs.tolist(),
                            material=mat,
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
                line_obj = fe.LinearLine
            # quadratic line elements
            elif line_type == 8:
                raise NotImplementedError
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
        """Creates a list of triangle indices that are used for plotting purposes.

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

    def get_node_idx_by_coordinates(
        self,
        x: float,
        y: float,
    ) -> int:
        """Returns the node index at or nearest to the point (``x``, ``y``).

        Args:
            x: ``x`` location of the node to find.
            y: ``y`` location of the node to find.

        Returns:
            Index of the node closest to (``x``, ``y``).
        """
        idx = self.nodes_str_tree.nearest(geometry=shapely.Point(x, y))

        return cast(int, idx)

    def get_tagged_node(
        self,
        tag: int,
    ) -> TaggedNode:
        """Returns a ``TaggedNode`` given a node tag."""
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

        Returns:
            Tagged node closest to (``x``, ``y``).
        """
        idx = self.tagged_nodes_str_tree.nearest(geometry=shapely.Point(x, y))

        return self.tagged_nodes[cast(int, idx)]

    def get_tagged_line(
        self,
        tag: int,
    ) -> TaggedLine:
        """Returns a ``TaggedLine`` given a line tag."""
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

        return self.tagged_lines[cast(int, idx)]

    def get_line_element_by_tag(
        self,
        tag: int,
    ) -> fe.LineElement:
        """Returns a ``LineElement`` given an element tag."""
        for line in self.line_elements:
            if line.line_tag == tag:
                return line
        else:
            raise ValueError(f"Cannot find FiniteElement with tag {tag}.")

    def get_finite_element_by_tag(
        self,
        tag: int,
    ) -> fe.FiniteElement:
        """Returns a ``FiniteElement`` given an element tag."""
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
            material_list: List of materials that correspond to the mesh attributes.
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

            # get number of materials
            num_materials = len(self.materials)

            # generate an array of polygon vertices and colors for each material
            verts = [[] for _ in range(num_materials)]
            colors = [[] for _ in range(num_materials)]

            for element in self.elements:
                idx = self.materials.index(element.material)  # get material index

                # get vertices - take care to create new array so as not to change vals
                coords = np.array(np.transpose(element.coords))

                # add displacements
                if ux is not None:
                    coords[:, 0] += ux[element.node_idxs]

                if uy is not None:
                    coords[:, 1] += uy[element.node_idxs]

                verts[idx].append(coords)

                # add colors
                colors[idx].append(element.material.color)

            # generate collection of polygons for each material
            for idx in range(num_materials):
                # get face color
                if materials:
                    fcs = colors[idx]
                else:
                    fcs = (1.0, 1.0, 1.0, 0.0)

                col = collections.PolyCollection(
                    verts[idx],
                    edgecolors=(0.0, 0.0, 0.0, alpha),
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
                    verts_orig.append(np.transpose(element.coords))

                col = collections.PolyCollection(
                    verts_orig,
                    edgecolors=(1.0, 0.0, 0.0, 0.1),
                    facecolors=(1.0, 1.0, 1.0, 0.0),
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
    elements: list[fe.FiniteElement]

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
