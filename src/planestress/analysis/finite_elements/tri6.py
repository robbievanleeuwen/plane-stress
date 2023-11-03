"""Tri6 element for a plane-stress analysis."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from numba import njit

import planestress.analysis.utils as utils
from planestress.analysis.finite_elements.finite_element import FiniteElement
from planestress.post.results import ElementResults


if TYPE_CHECKING:
    from planestress.pre.material import Material

TRI_ARRAY = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


class Tri6(FiniteElement):
    """Class for a six-noded quadratic triangular element."""

    def __init__(
        self,
        el_idx: int,
        el_tag: int,
        coords: npt.NDArray[np.float64],
        node_idxs: list[int],
        material: Material,
        orientation: bool,
    ) -> None:
        """Inits the Tri6 class.

        Args:
            el_idx: Element index.
            el_tag: Element mesh tag.
            coords: A ``2 x 6`` :class:`numpy.ndarray` of coordinates defining the
                element, i.e. ``[[x1, ..., x6], [y1, ..., y6]]``.
            node_idxs: A list of node indexes defining the element, i.e.
                ``[idx1, ..., idx6]``.
            material: Material of the element.
            orientation: If ``True`` the element is oriented correctly, if ``False`` the
                element's nodes will need reordering.
        """
        # reorient node indexes and coords if required
        if not orientation:
            node_idxs[1], node_idxs[2] = node_idxs[2], node_idxs[1]
            node_idxs[3], node_idxs[5] = node_idxs[5], node_idxs[3]
            coords[:, [1, 2, 3, 5]] = coords[:, [2, 1, 5, 3]]

        super().__init__(
            el_idx=el_idx,
            el_tag=el_tag,
            coords=coords,
            node_idxs=node_idxs,
            material=material,
            num_nodes=6,
            int_points=3,
            orientation=orientation,
        )

    @staticmethod
    @cache
    def shape_functions(
        iso_coords: tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Returns the shape functions at a point for a Tri6 element.

        Args:
            iso_coords: Location of the point in isoparametric coordinates.

        Returns:
            The values of the shape functions ``[N1, N2, N3, N4, N5, N6]``.
        """
        # location of isoparametric cooordinates
        zeta1, zeta2, zeta3 = iso_coords

        # generate the shape functions for a Tri6 element
        return np.array(
            [
                zeta1 * (2.0 * zeta1 - 1.0),
                zeta2 * (2.0 * zeta2 - 1.0),
                zeta3 * (2.0 * zeta3 - 1.0),
                4.0 * zeta1 * zeta2,
                4.0 * zeta2 * zeta3,
                4.0 * zeta1 * zeta3,
            ],
        )

    @staticmethod
    @cache
    @njit(cache=True, nogil=True)  # type: ignore
    def b_matrix_jacobian(
        iso_coords: tuple[float, float, float],
        coords: tuple[float, ...],
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Calculates the B matrix and jacobian at an isoparametric point for a Tri6.

        Args:
            iso_coords: Location of the point in isoparametric coordinates.
            coords: Flattened list of coordinates.

        Raises:
            RuntimeError: If the jacobian is less than zero.

        Returns:
            Derivatives of the shape function (B matrix) and value of the jacobian,
            (``b_mat``, ``j``).
        """
        # reshape coords
        coords_array = np.array(coords).reshape((2, 6))

        # get b_iso
        zeta1, zeta2, zeta3 = iso_coords
        b_iso = np.array(
            [
                # d/d(zeta1)
                [4.0 * zeta1 - 1.0, 0.0, 0.0, 4.0 * zeta2, 0.0, 4.0 * zeta3],
                # d/d(zeta2)
                [0.0, 4.0 * zeta2 - 1.0, 0.0, 4.0 * zeta1, 4.0 * zeta3, 0.0],
                # d/d(zeta3)
                [0.0, 0.0, 4.0 * zeta3 - 1.0, 0.0, 4.0 * zeta2, 4.0 * zeta1],
            ],
        )

        # form Jacobian matrix
        j = np.ones((3, 3))
        j[:, 1:] = b_iso @ coords_array.transpose()

        # calculate the jacobian
        jacobian = 0.5 * np.linalg.det(j)

        # if the area of the element is not zero
        if jacobian != 0:
            b_mat = TRI_ARRAY @ np.linalg.solve(j, b_iso)
        else:
            b_mat = np.zeros((2, 6), dtype=np.float64)  # empty b matrix

        # check sign of jacobian
        if jacobian < 0:
            raise RuntimeError("Jacobian of element is less than zero.")

        # form plane stress b matrix
        b_mat_ps = np.zeros((3, 12))

        # fill first two rows with first two rows of b matrix
        # first row - every second entry starting with first
        # second row - every second entry starting with second
        for i in range(2):
            b_mat_ps[i, i::2] = b_mat[i, :]

        # last row:
        # fill every second entry (starting with the first) from second row of b matrix
        b_mat_ps[2, ::2] = b_mat[1, :]
        # fill every second entry (starting with the second) from first row of b matrix
        b_mat_ps[2, 1::2] = b_mat[0, :]

        return b_mat_ps, jacobian

    def element_stiffness_matrix(self) -> npt.NDArray[np.float64]:
        """Assembles the stiffness matrix for a Tri6 element.

        Returns:
            Element stiffness matrix.
        """
        # allocate element stiffness matrix
        k_el = np.zeros((2 * self.num_nodes, 2 * self.num_nodes))

        # get d_matrix
        d_mat = self.material.get_d_matrix()

        # get gauss points
        gauss_points = utils.gauss_points_triangle(n_points=self.int_points)

        # loop through each gauss point
        for gauss_point in gauss_points:
            # extract weight and isoparametric coordinates
            weight = gauss_point[0]
            iso_coords = gauss_point[1:]

            # get b matrix and jacobian
            b_mat, j = self.b_matrix_jacobian(
                iso_coords=iso_coords,
                coords=tuple(self.coords.ravel()),
            )

            # calculate stiffness matrix for current integration point
            k_el += (
                b_mat.transpose() @ d_mat @ b_mat * weight * j * self.material.thickness
            )

        return k_el.ravel()

    def element_load_vector(
        self,
        acceleration_field: tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        """Assembles the load vector for a Tri6 element.

        Args:
            acceleration_field: Acceleration field (``a_x``, ``a_y``).

        Returns:
            Element load vector.
        """
        # allocate element load vector
        f_el = np.zeros(2 * self.num_nodes)

        # if acceleration is zero, return empty vector
        if acceleration_field[0] == 0.0 and acceleration_field[1] == 0.0:
            return f_el

        # calculate body force field
        b = np.array(acceleration_field) * self.material.density

        # get gauss points
        gauss_points = utils.gauss_points_triangle(n_points=self.int_points)

        # loop through each gauss point
        for gauss_point in gauss_points:
            # extract weight and isoparametric coordinates
            weight = gauss_point[0]
            iso_coords = gauss_point[1:]

            # get shape functions and jacobian
            n = self.shape_functions(iso_coords=iso_coords)
            _, j = self.b_matrix_jacobian(
                iso_coords=iso_coords,
                coords=tuple(self.coords.ravel()),
            )

            # form shape function matrix
            n_mat = np.zeros((len(n) * 2, 2))
            n_mat[::2, 0] = n
            n_mat[1::2, 1] = n

            # calculate load vector for current integration point
            f_el += n_mat @ b * weight * j * self.material.thickness

        return f_el

    def calculate_element_stresses(
        self,
        u: npt.NDArray[np.float64],
    ) -> ElementResults:
        r"""Calculates various results for a Tri6 element given nodal displacements.

        Calculates the following:

        - Stress components at the nodes (:math`\sigma_{xx}`, :math`\sigma_{yy}`,
          :math`\sigma_{xy}`).
        - TODO

        Args:
            u: Displacement vector for the element.

        Returns:
            Element results object.
        """
        # get d_matrix
        d_mat = self.material.get_d_matrix()

        # calculate stresses at gauss points, then extrapolate to nodes:
        # initialise gauss points stress results
        sigs_gps = np.zeros((self.int_points, 3))

        # get locations of gauss points in isoparametric coordinates
        gauss_points = utils.gauss_points_triangle(n_points=self.int_points)

        # loop through each point to calculate the stress
        for idx, iso_coords in enumerate(gauss_points):
            # get b matrix
            b_mat, _ = self.b_matrix_jacobian(
                iso_coords=iso_coords[1:],
                coords=tuple(self.coords.ravel()),
            )

            # calculate stress
            sigs_gps[idx, :] = d_mat @ b_mat @ u

        # extrapolate to nodes
        sigs = self.extrapolate_gauss_points_to_nodes() @ sigs_gps

        return ElementResults(
            el_idx=self.el_idx,
            node_idxs=self.node_idxs,
            sigs=sigs,
        )

    @cache
    def extrapolate_gauss_points_to_nodes(self) -> npt.NDArray[np.float64]:
        """Returns the extrapolation matrix for a Tri6 element.

        Returns:
            Extrapolation matrix.
        """
        return np.array(
            [
                [5.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0],
                [-1.0 / 3.0, 5.0 / 3.0, -1.0 / 3.0],
                [-1.0 / 3.0, -1.0 / 3.0, 5.0 / 3.0],
                [2.0 / 3.0, 2.0 / 3.0, -1.0 / 3.0],
                [-1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
                [2.0 / 3.0, -1.0 / 3.0, 2.0 / 3.0],
            ]
        )

    def get_polygon_coordinates(self) -> tuple[list[int], npt.NDArray[np.float64]]:
        """Returns a list of coordinates and indexes that define the element exterior.

        Returns:
            List of node indexes and exterior coordinates
        """
        return self.node_idxs[0:3], self.coords[0:3]

    def get_triangulation(self) -> list[tuple[int, int, int]]:
        """Returns a list of triangle indexes for a Tri6 element.

        Returns:
            List of triangle indexes.
        """
        return [
            (self.node_idxs[0], self.node_idxs[3], self.node_idxs[5]),
            (self.node_idxs[3], self.node_idxs[1], self.node_idxs[4]),
            (self.node_idxs[3], self.node_idxs[4], self.node_idxs[5]),
            (self.node_idxs[5], self.node_idxs[4], self.node_idxs[2]),
        ]
