"""Line elements for a plane-stress analysis."""

from __future__ import annotations

from functools import cache

import numpy as np
import numpy.typing as npt
from numba import njit

import planestress.analysis.utils as utils


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
        return f"{self.__class__.__name__} - id: {self.line_idx}, tag: {self.line_tag}."

    @staticmethod
    @cache
    @njit(cache=True, nogil=True)
    def shape_functions_jacobian(
        iso_coord: float,
        coords: tuple[float, ...],
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Evaluates the shape functions and jacobian of the element.

        Args:
            iso_coord: Location of the point in isoparametric coordinates.
            coords: Flattened coordinates array.

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

        # check value
        if value == 0:
            return f_el

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
            n, j = self.shape_functions_jacobian(
                iso_coord=gauss_point[1], coords=tuple(self.coords.ravel())
            )

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

    @staticmethod
    @cache
    @njit(cache=True, nogil=True)
    def shape_functions_jacobian(
        iso_coord: float,
        coords: tuple[float, ...],
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Evaluates the shape functions and jacobian of the element.

        Args:
            iso_coord: Location of the point in isoparametric coordinates.
            coords: Flattened coordinates array.

        Returns:
            Shape functions and jacobian.
        """
        xi = iso_coord  # isoparametric coordinate
        n = np.array([0.5 - 0.5 * xi, 0.5 + 0.5 * xi])  # shape functions
        b_iso = np.array([-0.5, 0.5])  # derivative of shape functions
        coords_array = np.array(coords).reshape((2, 2))
        j = b_iso @ coords_array.transpose()
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
                ``[[x1, x2, x3], [y1, y2, y3]]``.
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

    @staticmethod
    @cache
    @njit(cache=True, nogil=True)
    def shape_functions_jacobian(
        iso_coord: float,
        coords: tuple[float, ...],
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Evaluates the shape functions and jacobian of the element.

        Args:
            iso_coord: Location of the point in isoparametric coordinates.
            coords: Flattened coordinates array.

        Returns:
            Shape functions and jacobian.
        """
        xi = iso_coord  # isoparametric coordinate

        # shape functions
        n = np.array([-0.5 * xi * (1 - xi), 0.5 * xi * (1 + xi), 1 - xi**2])

        # derivative of shape functions
        b_iso = np.array([xi - 0.5, xi + 0.5, -2 * xi])
        coords_array = np.array(coords).reshape((2, 3))
        j = b_iso @ coords_array.transpose()
        jacobian = np.sqrt(np.sum(j**2))

        return n, jacobian
