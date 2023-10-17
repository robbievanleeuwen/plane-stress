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
        num_nodes: int,
    ) -> None:
        """Inits the FiniteElement class."""
        self.el_idx = el_idx
        self.coords = coords
        self.node_idxs = node_idxs
        self.material = material
        self.num_nodes = num_nodes

    def __str__(self) -> str:
        """Override string method."""
        return (
            f"{self.__class__.__name__} - id: {self.el_idx}, material: "
            f"{self.material.name}"
        )

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

    @staticmethod
    def shape_functions(
        iso_coords: tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Returns the shape functions for the finite element."""
        raise NotImplementedError

    @staticmethod
    def shape_functions_derivatives(
        iso_coords: tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Returns the derivatives of the shape functions for the finite element."""
        raise NotImplementedError

    def iso_to_global(
        self,
        iso_coords: tuple[float, float, float],
    ) -> tuple[float, float]:
        """Returns the global coordinates of the isoparametric coordinates."""
        x, y = self.coords @ self.shape_functions(iso_coords=iso_coords)

        return x, y

    @staticmethod
    def nodal_isoparametric_coordinates() -> npt.NDArray[np.float64]:
        """Returns the values of the isoparametric coordinates at the nodes."""
        raise NotImplementedError

    def b_matrix_jacobian(
        self,
        iso_coords: tuple[float, float, float],
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Calculates the B matrix and jacobian for a plane-stress element.

        See Felippa Ch. 24
        """
        # get the b matrix wrt. the isoparametric coordinates
        b_iso = self.shape_functions_derivatives(iso_coords=iso_coords)

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
            b_mat, j = self.b_matrix_jacobian(iso_coords=gauss_point[1:])

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

    def get_element_results(
        self,
        u: npt.NDArray[np.float64],
    ) -> ElementResults:
        """Calculates results for the finite element.

        Note for superparametric triangles we don't need to extrapolate from Gauss
        points (Felippa).
        """
        # initialise stress results
        sigs = np.zeros((self.num_nodes, 3))

        # get d_matrix
        d_mat = self.material.get_d_matrix()

        # get isoparametric coordinates at nodes
        iso_coords = self.nodal_isoparametric_coordinates()

        # loop through each node
        for idx, coords in enumerate(iso_coords):
            b_mat, _ = self.b_matrix_jacobian(iso_coords=coords)

            # calculate stress at node
            sigs[idx, :] = d_mat @ b_mat @ u

        return ElementResults(
            el_idx=self.el_idx,
            coords=self.coords,
            node_idxs=self.node_idxs,
            material=self.material,
            num_nodes=self.num_nodes,
            sigs=sigs,
        )


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
            el_idx=el_idx,
            coords=coords,
            node_idxs=node_idxs,
            material=material,
            num_nodes=3,
        )

    @staticmethod
    def shape_functions(
        iso_coords: tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Returns the shape functions for a Tri3 element."""
        # location of isoparametric coordinates
        eta, xi, zeta = iso_coords

        # for a Tri3, the shape functions are the isoparametric coordinates
        return np.array([eta, xi, zeta])

    @staticmethod
    def shape_functions_derivatives(
        iso_coords: tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Returns the derivatives of the shape functions for a Tri3 element."""
        # derivatives of the shape functions wrt the isoparametric coordinates
        return np.array(
            [
                [1.0, 0.0, 0.0],  # d/d(eta)
                [0.0, 1.0, 0.0],  # d/d(xi)
                [0.0, 0.0, 1.0],  # d/d(zeta)
            ]
        )

    @staticmethod
    def nodal_isoparametric_coordinates() -> npt.NDArray[np.float64]:
        """Returns the values of the isoparametric coordinates at the nodes."""
        return np.array(
            [
                [1.0, 0.0, 0.0],  # node 1
                [0.0, 1.0, 0.0],  # node 2
                [0.0, 0.0, 1.0],  # node 3
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
            el_idx=el_idx,
            coords=coords,
            node_idxs=node_idxs,
            material=material,
            num_nodes=6,
        )

    @staticmethod
    def shape_functions(
        iso_coords: tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Returns the shape functions for a Tri6 element."""
        # location of isoparametric cooordinates
        eta, xi, zeta = iso_coords

        # generate the shape functions for a Tri6 element
        return np.array(
            [
                eta * (2.0 * eta - 1.0),
                xi * (2.0 * xi - 1.0),
                zeta * (2.0 * zeta - 1.0),
                4.0 * eta * xi,
                4.0 * xi * zeta,
                4.0 * eta * zeta,
            ],
        )

    @staticmethod
    def shape_functions_derivatives(
        iso_coords: tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Returns the derivatives of the shape functions for a Tri6 element."""
        # location of isoparametric coordinates
        eta, xi, zeta = iso_coords

        # derivatives of the shape functions wrt the isoparametric cooordinates
        return np.array(
            [
                [4.0 * eta - 1.0, 0.0, 0.0, 4.0 * xi, 0.0, 4.0 * zeta],  # d/d(eta)
                [0.0, 4.0 * xi - 1.0, 0.0, 4.0 * eta, 4.0 * zeta, 0.0],  # d/d(xi)
                [0.0, 0.0, 4.0 * zeta - 1.0, 0.0, 4.0 * xi, 4.0 * eta],  # d/d(zeta)
            ],
        )

    @staticmethod
    def nodal_isoparametric_coordinates() -> npt.NDArray[np.float64]:
        """Returns the values of the isoparametric coordinates at the nodes."""
        return np.array(
            [
                [1.0, 0.0, 0.0],  # node 1
                [0.0, 1.0, 0.0],  # node 2
                [0.0, 0.0, 1.0],  # node 3
                [0.5, 0.5, 0.0],  # node 4
                [0.0, 0.5, 0.5],  # node 5
                [0.5, 0.0, 0.5],  # node 6
            ]
        )


class ElementResults(FiniteElement):
    """Class for storing the results of a finite element."""

    def __init__(
        self,
        el_idx: int,
        coords: npt.NDArray[np.float64],
        node_idxs: list[int],
        material: Material,
        num_nodes: int,
        sigs: npt.NDArray[np.float64],
    ) -> None:
        """Inits the ElementResults class."""
        super().__init__(
            el_idx=el_idx,
            coords=coords,
            node_idxs=node_idxs,
            material=material,
            num_nodes=num_nodes,
        )
        self.sigs = sigs
