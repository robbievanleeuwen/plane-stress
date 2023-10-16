"""Finite element classes for a plane-stress analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt


if TYPE_CHECKING:
    from planestress.pre.material import Material

TRI_ARRAY = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


class FiniteElement:
    """Abstract base class for a triangular plane-stress finite element."""

    def __init__(
        self,
        el_idx: int,
        coords: npt.NDArray[np.float64],
        node_idxs: list[int],
        material: Material,
    ) -> None:
        """Inits the FiniteElement class."""
        self.el_idx = el_idx
        self.coords = coords
        self.node_idxs = node_idxs
        self.material = material
        self.num_nodes: int = 0

    def __str__(self) -> str:
        """Override string method."""
        return (
            f"{self.__class__.__name__} - id: {self.el_idx}, material: "
            f"{self.material.name}"
        )

    def shape_functions(
        self,
        gauss_point: tuple[float, float, float],
    ) -> npt.NDArray[np.float64]:
        """Returns the shape functions for the finite element."""
        raise NotImplementedError

    def shape_functions_derivatives(
        self,
        gauss_point: tuple[float, float, float],
    ) -> npt.NDArray[np.float64]:
        """Returns the derivatives of the shape functions for the finite element."""
        raise NotImplementedError

    def global_gauss_point(
        self,
        gauss_point: tuple[float, float, float],
    ) -> tuple[float, float]:
        """Returns the global coordinates of the gauss point."""
        nx, ny = self.coords @ self.shape_functions(gauss_point=gauss_point)

        return nx, ny

    def b_matrix_jacobian(
        self,
        gauss_point: tuple[float, float, float],
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Calculates properties related to the finite element.

        See Felippa Ch. 24
        """
        b_iso = self.shape_functions_derivatives(gauss_point=gauss_point)

        # form Jacobian matrix
        j = np.ones((3, 3))
        j[:, 1:] = b_iso @ self.coords.transpose()

        # calculate the jacobian
        jacobian = 0.5 * np.linalg.det(j)

        # if the area of the element is not zero
        if jacobian != 0:
            b_mat = TRI_ARRAY @ np.linalg.solve(j, b_iso)
        else:
            b_mat = np.zeros((2, self.num_nodes))  # empty b matrix

        # form plane stress b matrix
        b_mat_ps = np.zeros((3, 2 * self.num_nodes))

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

    def element_stiffness_matrix(
        self,
        n_points: int,
    ) -> npt.NDArray[np.float64]:
        """Assembles the stiffness matrix for the element."""
        # allocate element stiffness matrix
        k_el = np.zeros((2 * self.num_nodes, 2 * self.num_nodes))

        # get d_matrix
        d_mat = self.material.get_d_matrix()

        # get Gauss points
        gauss_points = self.gauss_points(n_points=n_points)

        # loop through each gauss point
        for gauss_point in gauss_points:
            b_mat, j = self.b_matrix_jacobian(gauss_point=gauss_point[1:])

            # calculate stiffness matrix for current integration point
            k_el += (
                b_mat.transpose()
                @ d_mat
                @ b_mat
                * gauss_point[0]
                * j
                * self.material.thickness
            )

        return k_el

    def element_load_vector(
        self,
        n_points: int,
    ) -> npt.NDArray[np.float64]:
        """Assembles the load matrix for the element."""
        # allocate element load vector
        k_el = np.zeros(2 * self.num_nodes)

        # TODO - implement!

        return k_el

    @staticmethod
    def gauss_points(n_points: int) -> npt.NDArray[np.float64]:
        """Gaussian weights and locations for ``n_point`` Gaussian integration."""
        # one point gaussian integration
        if n_points == 1:
            return np.array([[1.0, 1.0 / 3, 1.0 / 3, 1.0 / 3]])

        # three point gaussian integration
        if n_points == 3:
            return np.array(
                [
                    [1.0 / 3, 2.0 / 3, 1.0 / 6, 1.0 / 6],
                    [1.0 / 3, 1.0 / 6, 2.0 / 3, 1.0 / 6],
                    [1.0 / 3, 1.0 / 6, 1.0 / 6, 2.0 / 3],
                ]
            )

        # four-point integration
        if n_points == 4:
            return np.array(
                [
                    [-27.0 / 48.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    [25.0 / 48.0, 0.6, 0.2, 0.2],
                    [25.0 / 48.0, 0.2, 0.6, 0.2],
                    [25.0 / 48.0, 0.2, 0.2, 0.6],
                ]
            )

        # six point gaussian integration
        if n_points == 6:
            g1 = 1.0 / 18 * (8 - np.sqrt(10) + np.sqrt(38 - 44 * np.sqrt(2.0 / 5)))
            g2 = 1.0 / 18 * (8 - np.sqrt(10) - np.sqrt(38 - 44 * np.sqrt(2.0 / 5)))
            w1 = (620 + np.sqrt(213125 - 53320 * np.sqrt(10))) / 3720
            w2 = (620 - np.sqrt(213125 - 53320 * np.sqrt(10))) / 3720

            return np.array(
                [
                    [w2, 1 - 2 * g2, g2, g2],
                    [w2, g2, 1 - 2 * g2, g2],
                    [w2, g2, g2, 1 - 2 * g2],
                    [w1, g1, g1, 1 - 2 * g1],
                    [w1, 1 - 2 * g1, g1, g1],
                    [w1, g1, 1 - 2 * g1, g1],
                ]
            )

        raise ValueError("n must be 1, 3, 4 or 6.")


class Tri3(FiniteElement):
    """Class for a three-noded linear triangular element.

    Coords: 2 x 3
    """

    def __init__(
        self,
        el_idx: int,
        coords: npt.NDArray[np.float64],
        node_idxs: list[int],
        material: Material,
    ) -> None:
        """Inits the Tri3 class."""
        super().__init__(
            el_idx=el_idx, coords=coords, node_idxs=node_idxs, material=material
        )
        self.num_nodes: int = 3

    def shape_functions(
        self,
        gauss_point: tuple[float, float, float],
    ) -> npt.NDArray[np.float64]:
        """Returns the shape functions for a Tri3 element."""
        # location of isoparametric co-ordinates for each Gauss point
        eta, xi, zeta = gauss_point

        return np.array([eta, xi, zeta])

    def shape_functions_derivatives(
        self,
        gauss_point: tuple[float, float, float],
    ) -> npt.NDArray[np.float64]:
        """Returns the derivatives of the shape functions for a Tri3 element."""
        # derivatives of the shape functions wrt the isoparametric co-ordinates
        return np.array(
            [
                [1, 0, 0],  # d/d(eta)
                [0, 1, 0],  # d/d(xi)
                [0, 0, 1],  # d/d(zeta)
            ]
        )


class Tri6(FiniteElement):
    """Class for a six-noded quadratic triangular element.

    Coords: 2 x 6
    """

    def __init__(
        self,
        el_idx: int,
        coords: npt.NDArray[np.float64],
        node_idxs: list[int],
        material: Material,
    ) -> None:
        """Inits the Tri6 class."""
        super().__init__(
            el_idx=el_idx, coords=coords, node_idxs=node_idxs, material=material
        )
        self.num_nodes: int = 6

    def shape_functions(
        self,
        gauss_point: tuple[float, float, float],
    ) -> npt.NDArray[np.float64]:
        """Returns the shape functions for a Tri3 element."""
        # location of isoparametric co-ordinates for each Gauss point
        eta, xi, zeta = gauss_point

        return np.array(
            [
                eta * (2 * eta - 1),
                xi * (2 * xi - 1),
                zeta * (2 * zeta - 1),
                4 * eta * xi,
                4 * xi * zeta,
                4 * eta * zeta,
            ],
        )

    def shape_functions_derivatives(
        self,
        gauss_point: tuple[float, float, float],
    ) -> npt.NDArray[np.float64]:
        """Returns the derivatives of the shape functions for a Tri3 element."""
        # location of isoparametric co-ordinates for each Gauss point
        eta, xi, zeta = gauss_point

        # derivatives of the shape functions wrt the isoparametric co-ordinates
        return np.array(
            [
                [4 * eta - 1, 0, 0, 4 * xi, 0, 4 * zeta],  # d/d(eta)
                [0, 4 * xi - 1, 0, 4 * eta, 4 * zeta, 0],  # d/d(xi)
                [0, 0, 4 * zeta - 1, 0, 4 * xi, 4 * eta],  # d/d(zeta)
            ],
        )
