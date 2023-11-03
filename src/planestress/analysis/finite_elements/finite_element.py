"""Finite element abstract class for a plane-stress analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

import planestress.analysis.utils as utils


if TYPE_CHECKING:
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


class LineElement:
    """Abstract base class for a line element."""

    def __init__(
        self,
        line_idx: int,
        line_tag: int,
        coords: npt.NDArray[np.float64],
        node_idxs: list[int],
        num_nodes: int,
        int_points: int,
    ) -> None:
        """Inits the LineElement class.

        Args:
            line_idx: Line element index.
            line_tag: Mesh line element tag.
            coords: A :class:`numpy.ndarray` of coordinates defining the element, e.g.
                ``[[x1, x2], [y1, y2]]``.
            node_idxs: List of node indexes defining the element, e.g.
                ``[idx1, idx2]``.
            num_nodes: Number of nodes in the line element.
            int_points: Number of integration points used for the finite element.
        """
        self.line_idx = line_idx
        self.line_tag = line_tag
        self.coords = coords
        self.node_idxs = node_idxs
        self.num_nodes = num_nodes
        self.int_points = int_points

    def __repr__(self) -> str:
        """Override __repr__ method.

        Returns:
            String representation of the object.
        """
        return (
            f"{self.__class__.__name__} - id: {self.line_idx}, "
            f"tag: {self.line_tag}."
        )

    def shape_functions_jacobian(
        self,
        iso_coord: float,
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Evaluates the shape functions and jacobian of the element.

        Args:
            iso_coord: Location of the point in isoparametric coordinates.

        Raises:
            NotImplementedError: If this method hasn't been implemented for an element.
        """
        raise NotImplementedError

    def element_load_vector(
        self,
        direction: str,
        value: float,
    ) -> npt.NDArray[np.float64]:
        """Assembles the load vector for the line element.

        Args:
            direction: Direction of the line load, ``"x"``, ``"y"`` or ``"xy"``.
            value: Value of the line load.

        Returns:
            Line element load vector.
        """
        # allocate element load vector
        f_el = np.zeros(2 * self.num_nodes)

        # create applied force vector
        if direction == "x":
            b = np.array([1.0, 0.0])
        elif direction == "y":
            b = np.array([0.0, 1.0])
        else:
            b = np.array([1.0, 1.0])

        b *= value

        # get gauss points
        gauss_points = utils.gauss_points_line(n_points=self.int_points)

        # loop through each gauss point
        for gauss_point in gauss_points:
            # get shape functions and length
            n, j = self.shape_functions_jacobian(iso_coord=gauss_point[1])

            # form shape function matrix
            n_mat = np.zeros((len(n) * 2, 2))
            n_mat[::2, 0] = n
            n_mat[1::2, 1] = n

            # calculate load vector for current integration point
            f_el += n_mat @ b * gauss_point[0] * j

        return f_el


class LinearLine(LineElement):
    """Class for a two-noded linear line element."""

    def __init__(
        self,
        line_idx: int,
        line_tag: int,
        coords: npt.NDArray[np.float64],
        node_idxs: list[int],
    ) -> None:
        """Inits the LinearLine class.

        Args:
            line_idx: Line element index.
            line_tag: Mesh line element tag.
            coords: A :class:`numpy.ndarray` of coordinates defining the element, e.g.
                ``[[x1, x2], [y1, y2]]``.
            node_idxs: List of node indexes defining the element, e.g.
                ``[idx1, idx2]``.
        """
        super().__init__(
            line_idx=line_idx,
            line_tag=line_tag,
            coords=coords,
            node_idxs=node_idxs,
            num_nodes=2,
            int_points=1,
        )

    def shape_functions_jacobian(
        self,
        iso_coord: float,
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Evaluates the shape functions and jacobian of the element.

        Args:
            iso_coord: Location of the point in isoparametric coordinates.

        Returns:
            Shape functions and jacobian.
        """
        xi = iso_coord  # isoparametric coordinate
        n = np.array([0.5 - 0.5 * xi, 0.5 + 0.5 * xi])  # shape functions
        b_iso = np.array([-0.5, 0.5])  # derivative of shape functions
        j = b_iso @ self.coords.transpose()
        jacobian = np.sqrt(np.sum(j**2))

        return n, jacobian


class QuadraticLine(LineElement):
    """Class for a three-noded quadratic line element."""

    def __init__(
        self,
        line_idx: int,
        line_tag: int,
        coords: npt.NDArray[np.float64],
        node_idxs: list[int],
    ) -> None:
        """Inits the QuadraticLine class.

        ``idx1`` -- ``idx3`` -- ``idx2``

        Args:
            line_idx: Line element index.
            line_tag: Mesh line element tag.
            coords: A :class:`numpy.ndarray` of coordinates defining the element, e.g.
                ``[[x1, x2, x3], [y1, y2,   y3]]``.
            node_idxs: List of node indexes defining the element, e.g.
                ``[idx1, idx2, idx3]``.
        """
        super().__init__(
            line_idx=line_idx,
            line_tag=line_tag,
            coords=coords,
            node_idxs=node_idxs,
            num_nodes=3,
            int_points=2,
        )

    def shape_functions_jacobian(
        self,
        iso_coord: float,
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Evaluates the shape functions and jacobian of the element.

        Args:
            iso_coord: Location of the point in isoparametric coordinates.

        Returns:
            Shape functions and jacobian.
        """
        xi = iso_coord  # isoparametric coordinate

        # shape functions
        n = np.array([-0.5 * xi * (1 - xi), 0.5 * xi * (1 + xi), 1 - xi**2])

        # derivative of shape functions
        b_iso = np.array([xi - 0.5, xi + 0.5, -2 * xi])
        j = b_iso @ self.coords.transpose()
        jacobian = np.sqrt(np.sum(j**2))

        return n, jacobian
