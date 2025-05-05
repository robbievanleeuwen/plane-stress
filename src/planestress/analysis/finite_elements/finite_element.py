"""Finite element abstract class for a plane-stress analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import planestress.analysis.utils as utils

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from planestress.post.results import ElementResults
    from planestress.pre.material import Material


class FiniteElement:
    """Abstract base class for a plane-stress finite element."""

    def __init__(
        self,
        el_idx: int,
        el_tag: int,
        coords: npt.NDArray[np.float64],
        node_idxs: list[int],
        material: Material,
        orientation: bool,
        num_nodes: int,
        int_points: int,
    ) -> None:
        """Inits the FiniteElement class.

        Args:
            el_idx: Element index.
            el_tag: Element mesh tag.
            coords: A :class:`numpy.ndarray` of coordinates defining the element, e.g.
                ``[[x1, x2, x3], [y1, y2, y3]]``.
            node_idxs: List of node indexes defining the element, e.g.
                ``[idx1, idx2, idx3]``.
            material: Material of the element.
            orientation: If ``True`` the element is oriented correctly, if ``False`` the
                element's nodes will need reordering.
            num_nodes: Number of nodes in the finite element.
            int_points: Number of integration points used for the finite element.
        """
        self.el_idx = el_idx
        self.el_tag = el_tag
        self.coords = coords
        self.node_idxs = node_idxs
        self.material = material
        self.orientation = orientation
        self.num_nodes = num_nodes
        self.int_points = int_points

    def __repr__(self) -> str:
        """Override __repr__ method.

        Returns:
            String representation of the object.
        """
        return (
            f"{self.__class__.__name__} - id: {self.el_idx}, tag: {self.el_tag}, "
            f"material: {self.material.name}."
        )

    def element_row_indexes(self) -> list[int]:
        """Returns the row indexes fore the global stiffness matrix.

        Returns:
            Row indexes.
        """
        dofs = utils.dof_map(node_idxs=tuple(self.node_idxs))

        return [dof for dof in dofs for _ in range(len(dofs))]

    def element_col_indexes(self) -> list[int]:
        """Returns the column indexes fore the global stiffness matrix.

        Returns:
            Column indexes.
        """
        dofs = utils.dof_map(node_idxs=tuple(self.node_idxs))

        return dofs * len(dofs)

    def element_stiffness_matrix(self) -> npt.NDArray[np.float64]:
        """Assembles the stiffness matrix for the element.

        Raises:
            NotImplementedError: If this method hasn't been implemented
        """
        raise NotImplementedError

    def element_load_vector(
        self,
        acceleration_field: tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        """Assembles the load vector for the element.

        Args:
            acceleration_field: Acceleration field (``a_x``, ``a_y``).

        Raises:
            NotImplementedError: If this method hasn't been implemented
        """
        raise NotImplementedError

    def calculate_element_stresses(
        self,
        u: npt.NDArray[np.float64],
    ) -> ElementResults:
        r"""Calculates various results for the finite element given nodal displacements.

        Calculates the following:

        - Stress components at the nodes (:math`\sigma_{xx}`, :math`\sigma_{yy}`,
          :math`\sigma_{xy}`).
        - TODO

        Args:
            u: Displacement vector for the element.

        Raises:
            NotImplementedError: If this method hasn't been implemented
        """
        raise NotImplementedError

    def get_polygon_coordinates(self) -> tuple[list[int], npt.NDArray[np.float64]]:
        """Returns a list of coordinates and indexes that define the element exterior.

        Raises:
            NotImplementedError: If this method hasn't been implemented for an element.
        """
        raise NotImplementedError

    def get_triangulation(self) -> list[tuple[int, int, int]]:
        """Returns a list of triangle indexes for the finite element.

        Raises:
            NotImplementedError: If this method hasn't been implemented for an element.
        """
        raise NotImplementedError

    def extrapolate_gauss_points_to_nodes(self) -> npt.NDArray[np.float64]:
        """Returns the extrapolation matrix for a Quad9 element.

        Returns:
            Extrapolation matrix.
        """
        raise NotImplementedError
