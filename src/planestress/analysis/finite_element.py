"""Finite element classes for a plane-stress analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

import planestress.analysis.utils as utils
from planestress.post.results import ElementResults


if TYPE_CHECKING:
    from planestress.pre.material import Material

TRI_ARRAY = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


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

    @staticmethod
    def shape_functions(
        iso_coords: tuple[float, float, float],
    ) -> npt.NDArray[np.float64]:
        """Returns the shape functions at a point.

        Args:
            iso_coords: Location of the point in isoparametric coordinates.

        Raises:
            NotImplementedError: If this method hasn't been implemented for an element.
        """
        raise NotImplementedError

    @staticmethod
    def shape_functions_derivatives(
        iso_coords: tuple[float, float, float],
    ) -> npt.NDArray[np.float64]:
        """Returns the derivatives of the shape functions at a point.

        Args:
            iso_coords: Location of the point in isoparametric coordinates.

        Raises:
            NotImplementedError: If this method hasn't been implemented for an element.
        """
        raise NotImplementedError

    @staticmethod
    def nodal_isoparametric_coordinates() -> npt.NDArray[np.float64]:
        """Returns the values of the isoparametric coordinates at the nodes.

        Raises:
            NotImplementedError: If this method hasn't been implemented for an element.
        """
        raise NotImplementedError

    def b_matrix_jacobian(
        self,
        iso_coords: tuple[float, float, float],
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Calculates the B matrix and jacobian at an isoparametric point.

        Args:
            iso_coords: Location of the point in isoparametric coordinates.

        Raises:
            NotImplementedError: If this method hasn't been implemented for an element.
        """
        raise NotImplementedError

    def extrapolate_gauss_points_to_nodes(self) -> npt.NDArray[np.float64]:
        """Returns the extrapolation matrix for the element.

        Raises:
            NotImplementedError: If this method hasn't been implemented for an element.
        """
        raise NotImplementedError

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

        Returns:
            Flattened element stiffness matrix.
        """
        # allocate element stiffness matrix
        k_el = np.zeros((2 * self.num_nodes, 2 * self.num_nodes))

        # get d_matrix
        d_mat = self.material.get_d_matrix()

        # get gauss points
        if isinstance(self, TriangularElement):
            gp_func = utils.gauss_points_triangle
        else:
            gp_func = utils.gauss_points_quad

        gauss_points = gp_func(n_points=self.int_points)

        # loop through each gauss point
        for gauss_point in gauss_points:
            # extract weight and isoparametric coordinates
            weight = gauss_point[0]
            iso_coords = gauss_point[1:]

            # get b matrix and jacobian
            b_mat, j = self.b_matrix_jacobian(iso_coords=iso_coords)

            # calculate stiffness matrix for current integration point
            k_el += (
                b_mat.transpose() @ d_mat @ b_mat * weight * j * self.material.thickness
            )

        return k_el.ravel()

    def element_load_vector(
        self,
        acceleration_field: tuple[float, float],
    ) -> npt.NDArray[np.float64]:
        """Assembles the load vector for the element.

        Args:
            acceleration_field: Acceleration field (``a_x``, ``a_y``).

        Returns:
            Element load vector.
        """
        # allocate element load vector
        f_el = np.zeros(2 * self.num_nodes)

        # calculate body force field
        b = np.array(acceleration_field) * self.material.density

        # get gauss points
        if isinstance(self, TriangularElement):
            gp_func = utils.gauss_points_triangle
        else:
            gp_func = utils.gauss_points_quad

        gauss_points = gp_func(n_points=self.int_points)

        # loop through each gauss point
        for gauss_point in gauss_points:
            # extract weight and isoparametric coordinates
            weight = gauss_point[0]
            iso_coords = gauss_point[1:]

            # get shape functions and jacobian
            n = self.shape_functions(iso_coords=iso_coords)
            _, j = self.b_matrix_jacobian(iso_coords=iso_coords)

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
        r"""Calculates various results for the finite element given nodal displacements.

        Calculates the following:

        - Stress components at the nodes (:math`\sigma_{xx}`, :math`\sigma_{yy}`,
          :math`\sigma_{xy}`).
        - TODO

        Args:
            u: Displacement vector for the element.

        Returns:
            ``ElementResults`` object.
        """
        # get d_matrix
        d_mat = self.material.get_d_matrix()

        # for triangular elements, calculate stresses directly at the nodes
        # for quadrilateral elements, calculate stresses at gauss points, then
        # extrapolate to nodes
        if isinstance(self, TriangularElement):
            # initialise nodal points stress results
            sigs_points = np.zeros((self.num_nodes, 3))

            # get locations of nodes in isoparametric coordinates
            points = self.nodal_isoparametric_coordinates()
        else:
            # initialise gauss points stress results
            sigs_points = np.zeros((self.int_points**2, 3))

            # get locations of gauss points in isoparametric coordinates
            points = utils.gauss_points_quad(n_points=self.int_points)[:, 1:]

        # loop through each point to calculate the stress
        for idx, iso_coords in enumerate(points):
            # get b matrix
            b_mat, _ = self.b_matrix_jacobian(iso_coords=iso_coords)

            # calculate stress
            sigs_points[idx, :] = d_mat @ b_mat @ u

        # if quadrilaterals, extrapolate to nodes
        if isinstance(self, QuadrilateralElement):
            sigs = self.extrapolate_gauss_points_to_nodes() @ sigs_points
        else:
            sigs = sigs_points

        return ElementResults(
            el_idx=self.el_idx,
            node_idxs=self.node_idxs,
            sigs=sigs,
        )

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


class TriangularElement(FiniteElement):
    """Abstract base class for a triangular plane-stress finite element."""

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
        """Inits the TriangularElement class.

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
        super().__init__(
            el_idx=el_idx,
            el_tag=el_tag,
            coords=coords,
            node_idxs=node_idxs,
            material=material,
            orientation=orientation,
            num_nodes=num_nodes,
            int_points=int_points,
        )

    def b_matrix_jacobian(
        self,
        iso_coords: tuple[float, float, float],
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Calculates the B matrix and jacobian at an isoparametric point (Tri).

        Args:
            iso_coords: Location of the point in isoparametric coordinates.

        Raises:
            RuntimeError: If the jacobian is less than zero.

        Returns:
            Derivatives of the shape function (B matrix) and value of the jacobian,
            (``b_mat``, ``j``).
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

        # check sign of jacobian
        if jacobian < 0:
            raise RuntimeError(
                f"Jacobian of element {self.el_idx} is less than zero ({jacobian:.2f})."
            )

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

    def get_polygon_coordinates(self) -> tuple[list[int], npt.NDArray[np.float64]]:
        """Returns a list of coordinates and indexes that define the element exterior.

        Returns:
            List of node indexes and exterior coordinates
        """
        return self.node_idxs[0:3], self.coords[:, 0:3]


class Tri3(TriangularElement):
    """Class for a three-noded linear triangular element."""

    def __init__(
        self,
        el_idx: int,
        el_tag: int,
        coords: npt.NDArray[np.float64],
        node_idxs: list[int],
        material: Material,
        orientation: bool,
    ) -> None:
        """Inits the Tri3 class.

        Args:
            el_idx: Element index.
            el_tag: Element mesh tag.
            coords: A ``2 x 3`` :class:`numpy.ndarray` of coordinates defining the
                element, i.e. ``[[x1, x2, x3], [y1, y2, y3]]``.
            node_idxs: A list of node indexes defining the element, i.e.
                ``[idx1, idx2, idx3]``.
            material: Material of the element.
            orientation: If ``True`` the element is oriented correctly, if ``False`` the
                element's nodes will need reordering.
        """
        # reorient node indexes and coords if required
        if not orientation:
            node_idxs[1], node_idxs[2] = node_idxs[2], node_idxs[1]
            coords[:, [1, 2]] = coords[:, [2, 1]]

        super().__init__(
            el_idx=el_idx,
            el_tag=el_tag,
            coords=coords,
            node_idxs=node_idxs,
            material=material,
            orientation=orientation,
            num_nodes=3,
            int_points=1,
        )

    @staticmethod
    def shape_functions(
        iso_coords: tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Returns the shape functions at a point for a Tri3 element.

        Args:
            iso_coords: Location of the point in isoparametric coordinates.

        Returns:
            The values of the shape functions ``[N1, N2, N3]``.
        """
        # location of isoparametric coordinates
        zeta1, zeta2, zeta3 = iso_coords

        # for a Tri3, the shape functions are the isoparametric coordinates
        return np.array([zeta1, zeta2, zeta3])

    @staticmethod
    def shape_functions_derivatives(
        iso_coords: tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Returns the derivatives of the shape functions at a point for a Tri3 element.

        Args:
            iso_coords: Location of the point in isoparametric coordinates.

        Returns:
            The partial derivatives of the shape functions.
        """
        # derivatives of the shape functions wrt the isoparametric coordinates
        return np.array(
            [
                [1.0, 0.0, 0.0],  # d/d(zeta1)
                [0.0, 1.0, 0.0],  # d/d(zeta2)
                [0.0, 0.0, 1.0],  # d/d(zeta3)
            ]
        )

    @staticmethod
    def nodal_isoparametric_coordinates() -> npt.NDArray[np.float64]:
        """Returns the values of the isoparametric coordinates at the nodes.

        Returns:
            Values of the isoparametric coordinates at the nodes.
        """
        return np.array(
            [
                [1.0, 0.0, 0.0],  # node 1
                [0.0, 1.0, 0.0],  # node 2
                [0.0, 0.0, 1.0],  # node 3
            ]
        )

    def get_triangulation(self) -> list[tuple[int, int, int]]:
        """Returns a list of triangle indexes for a Tri3 element.

        Returns:
            List of triangle indexes.
        """
        return [(self.node_idxs[0], self.node_idxs[1], self.node_idxs[2])]


class Tri6(TriangularElement):
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
    def shape_functions_derivatives(
        iso_coords: tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Returns the derivatives of the shape functions at a point for a Tri6 element.

        Args:
            iso_coords: Location of the point in isoparametric coordinates.

        Returns:
            The partial derivatives of the shape functions.
        """
        # location of isoparametric coordinates
        zeta1, zeta2, zeta3 = iso_coords

        # derivatives of the shape functions wrt the isoparametric cooordinates
        return np.array(
            [
                # d/d(zeta1)
                [4.0 * zeta1 - 1.0, 0.0, 0.0, 4.0 * zeta2, 0.0, 4.0 * zeta3],
                # d/d(zeta2)
                [0.0, 4.0 * zeta2 - 1.0, 0.0, 4.0 * zeta1, 4.0 * zeta3, 0.0],
                # d/d(zeta3)
                [0.0, 0.0, 4.0 * zeta3 - 1.0, 0.0, 4.0 * zeta2, 4.0 * zeta1],
            ],
        )

    @staticmethod
    def nodal_isoparametric_coordinates() -> npt.NDArray[np.float64]:
        """Returns the values of the isoparametric coordinates at the nodes.

        Returns:
            Values of the isoparametric coordinates at the nodes.
        """
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


class QuadrilateralElement(FiniteElement):
    """Abstract base class for a quadrilateral plane-stress finite element."""

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
        """Inits the QuadrilateralElement class.

        Args:
            el_idx: Element index.
            el_tag: Element mesh tag.
            coords: A :class:`numpy.ndarray` of coordinates defining the element, e.g.
                ``[[x1, x2, x3, x4], [y1, y2, y3, y4]]``.
            node_idxs: List of node indexes defining the element, e.g.
                ``[idx1, idx2, idx3, idx4]``.
            material: Material of the element.
            orientation: If ``True`` the element is oriented correctly, if ``False`` the
                element's nodes will need reordering.
            num_nodes: Number of nodes in the finite element.
            int_points: Number of integration points used for the finite element.
        """
        super().__init__(
            el_idx=el_idx,
            el_tag=el_tag,
            coords=coords,
            node_idxs=node_idxs,
            material=material,
            orientation=orientation,
            num_nodes=num_nodes,
            int_points=int_points,
        )

    def b_matrix_jacobian(
        self,
        iso_coords: tuple[float, float, float],
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Calculates the B matrix and jacobian at an isoparametric point (Quad).

        Args:
            iso_coords: Location of the point in isoparametric coordinates (note last
                value is ignored).

        Raises:
            RuntimeError: If the jacobian is less than zero.

        Returns:
            Derivatives of the shape function (B matrix) and value of the jacobian,
            (``b_mat``, ``j``).
        """
        # get the b matrix wrt. the isoparametric coordinates
        b_iso = self.shape_functions_derivatives(iso_coords=iso_coords)

        # form Jacobian matrix
        j = b_iso @ self.coords.transpose()

        # calculate the jacobian
        jacobian = np.linalg.det(j)

        # if the area of the element is not zero
        if jacobian != 0:
            b_mat = np.linalg.solve(j, b_iso)
        else:
            b_mat = np.zeros((2, self.num_nodes))  # empty b matrix

        # check sign of jacobian
        if jacobian < 0:
            raise RuntimeError(
                f"Jacobian of element {self.el_idx} is less than zero ({jacobian:.2f})."
            )

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

    def get_polygon_coordinates(self) -> tuple[list[int], npt.NDArray[np.float64]]:
        """Returns a list of coordinates and indexes that define the element exterior.

        Returns:
            List of node indexes and exterior coordinates
        """
        return self.node_idxs[0:4], self.coords[:, 0:4]


class Quad4(QuadrilateralElement):
    """Class for a four-noded linear quadrilateral element."""

    def __init__(
        self,
        el_idx: int,
        el_tag: int,
        coords: npt.NDArray[np.float64],
        node_idxs: list[int],
        material: Material,
        orientation: bool,
    ) -> None:
        """Inits the Quad4 class.

        Args:
            el_idx: Element index.
            el_tag: Element mesh tag.
            coords: A ``2 x 4`` :class:`numpy.ndarray` of coordinates defining the
                element, i.e. ``[[x1, x2, x3, x4], [y1, y2, y3, y4]]``.
            node_idxs: A list of node indexes defining the element, i.e.
                ``[idx1, idx2, idx3, idx4]``.
            material: Material of the element.
            orientation: If ``True`` the element is oriented correctly, if ``False`` the
                element's nodes will need reordering.
        """
        # reorient node indexes and coords if required
        if not orientation:
            node_idxs[1], node_idxs[3] = node_idxs[3], node_idxs[1]
            coords[:, [1, 3]] = coords[:, [3, 1]]

        super().__init__(
            el_idx=el_idx,
            el_tag=el_tag,
            coords=coords,
            node_idxs=node_idxs,
            material=material,
            orientation=orientation,
            num_nodes=4,
            int_points=2,
        )

    @staticmethod
    def shape_functions(
        iso_coords: tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Returns the shape functions at a point for a Quad4 element.

        Args:
            iso_coords: Location of the point in isoparametric coordinates (note last
                value is ignored).

        Returns:
            The values of the shape functions ``[N1, N2, N3, N4]``.
        """
        # location of isoparametric coordinates
        xi, eta, _ = iso_coords

        # generate the shape functions for a Quad4 element
        return np.array(
            [
                0.25 * (1.0 - xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 + eta),
                0.25 * (1.0 - xi) * (1.0 + eta),
            ]
        )

    @staticmethod
    def shape_functions_derivatives(
        iso_coords: tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Returns the derivatives of the shape functions at a pt for a Quad4 element.

        Args:
            iso_coords: Location of the point in isoparametric coordinates (note last
                value is ignored).

        Returns:
            The partial derivatives of the shape functions.
        """
        # location of isoparametric coordinates
        xi, eta, _ = iso_coords

        # derivatives of the shape functions wrt the isoparametric coordinates
        return np.array(
            [
                # d/d(xi)
                [
                    0.25 * (eta - 1.0),
                    0.25 * (1.0 - eta),
                    0.25 * (1.0 + eta),
                    -0.25 * (1.0 + eta),
                ],
                # d/d(eta)
                [
                    0.25 * (xi - 1.0),
                    -0.25 * (1.0 + xi),
                    0.25 * (1.0 + xi),
                    0.25 * (1.0 - xi),
                ],
            ]
        )

    @staticmethod
    def nodal_isoparametric_coordinates() -> npt.NDArray[np.float64]:
        """Returns the values of the isoparametric coordinates at the nodes.

        Returns:
            Values of the isoparametric coordinates at the nodes.
        """
        return np.array(
            [
                [-1.0, -1.0],  # node 1
                [1.0, -1.0],  # node 2
                [1.0, 1.0],  # node 3
                [-1.0, 1.0],  # node 4
            ]
        )

    def extrapolate_gauss_points_to_nodes(self) -> npt.NDArray[np.float64]:
        """Returns the extrapolation matrix for a Quad4 element.

        Returns:
            Extrapolation matrix.
        """
        return np.array(
            [
                [1.0 + 0.5 * np.sqrt(3), -0.5, 1.0 - 0.5 * np.sqrt(3), -0.5],
                [-0.5, 1.0 + 0.5 * np.sqrt(3), -0.5, 1.0 - 0.5 * np.sqrt(3)],
                [1.0 - 0.5 * np.sqrt(3), -0.5, 1.0 + 0.5 * np.sqrt(3), -0.5],
                [-0.5, 1.0 - 0.5 * np.sqrt(3), -0.5, 1.0 + 0.5 * np.sqrt(3)],
            ]
        )

    def get_triangulation(self) -> list[tuple[int, int, int]]:
        """Returns a list of triangle indexes for a Quad4 element.

        Returns:
            List of triangle indexes.
        """
        return [
            (self.node_idxs[0], self.node_idxs[1], self.node_idxs[2]),
            (self.node_idxs[0], self.node_idxs[2], self.node_idxs[3]),
        ]


class Quad8(QuadrilateralElement):
    """Class for an eight-noded quadratic quadrilateral element."""

    def __init__(
        self,
        el_idx: int,
        el_tag: int,
        coords: npt.NDArray[np.float64],
        node_idxs: list[int],
        material: Material,
        orientation: bool,
    ) -> None:
        """Inits the Quad8 class.

        Args:
            el_idx: Element index.
            el_tag: Element mesh tag.
            coords: A ``2 x 9`` :class:`numpy.ndarray` of coordinates defining the
                element, i.e. ``[[x1, x2, ..., x8], [y1, y2, ..., y8]]``.
            node_idxs: A list of node indexes defining the element, i.e.
                ``[idx1, idx2, ..., idx8]``.
            material: Material of the element.
            orientation: If ``True`` the element is oriented correctly, if ``False`` the
                element's nodes will need reordering.
        """
        # reorient node indexes and coords if required
        if not orientation:
            node_idxs[1], node_idxs[3] = node_idxs[3], node_idxs[1]
            node_idxs[4], node_idxs[7] = node_idxs[7], node_idxs[4]
            node_idxs[5], node_idxs[6] = node_idxs[6], node_idxs[5]
            coords[:, [1, 3, 4, 7, 5, 6]] = coords[:, [3, 1, 7, 4, 6, 5]]

        super().__init__(
            el_idx=el_idx,
            el_tag=el_tag,
            coords=coords,
            node_idxs=node_idxs,
            material=material,
            orientation=orientation,
            num_nodes=8,
            int_points=3,
        )

    @staticmethod
    def shape_functions(
        iso_coords: tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Returns the shape functions at a point for a Quad8 element.

        Args:
            iso_coords: Location of the point in isoparametric coordinates (note last
                value is ignored).

        Returns:
            The values of the shape functions ``[N1, N2, ..., N8]``.
        """
        # location of isoparametric coordinates
        xi, eta, _ = iso_coords

        # generate the shape functions for a Quad8 element
        return np.array(
            [
                -0.25 * (1.0 - xi) * (1.0 - eta) * (1.0 + xi + eta),
                -0.25 * (1.0 + xi) * (1.0 - eta) * (1.0 - xi + eta),
                -0.25 * (1.0 + xi) * (1.0 + eta) * (1.0 - xi - eta),
                -0.25 * (1.0 - xi) * (1.0 + eta) * (1.0 + xi - eta),
                0.5 * (1.0 - xi**2) * (1.0 - eta),
                0.5 * (1.0 + xi) * (1.0 - eta**2),
                0.5 * (1.0 - xi**2) * (1.0 + eta),
                0.5 * (1.0 - xi) * (1.0 - eta**2),
            ]
        )

    @staticmethod
    def shape_functions_derivatives(
        iso_coords: tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Returns the derivatives of the shape functions at a pt for a Quad8 element.

        Args:
            iso_coords: Location of the point in isoparametric coordinates (note last
                value is ignored).

        Returns:
            The partial derivatives of the shape functions.
        """
        # location of isoparametric coordinates
        xi, eta, _ = iso_coords

        # derivatives of the shape functions wrt the isoparametric coordinates
        return np.array(
            [
                # d/d(xi)
                [
                    eta * (0.25 - 0.5 * xi) - 0.25 * eta**2 + 0.5 * xi,
                    0.25 * eta**2 + eta * (-0.5 * xi - 0.25) + 0.5 * xi,
                    0.25 * eta**2 + eta * (0.5 * xi + 0.25) + 0.5 * xi,
                    -0.25 * eta**2 + eta * (0.5 * xi - 0.25) + 0.5 * xi,
                    (eta - 1.0) * xi,
                    0.5 - 0.5 * eta**2,
                    -1.0 * (eta + 1.0) * xi,
                    0.5 * eta**2 - 0.5,
                ],
                # d/d(eta)
                [
                    eta * (0.5 - 0.5 * xi) - 0.25 * xi**2 + 0.25 * xi,
                    eta * (0.5 * xi + 0.5) - 0.25 * xi**2 - 0.25 * xi,
                    eta * (0.5 * xi + 0.5) + 0.25 * xi**2 + 0.25 * xi,
                    eta * (0.5 - 0.5 * xi) + 0.25 * xi**2 - 0.25 * xi,
                    0.5 * xi**2 - 0.5,
                    -eta * (xi + 1.0),
                    0.5 - 0.5 * xi**2,
                    eta * (xi - 1.0),
                ],
            ]
        )

    @staticmethod
    def nodal_isoparametric_coordinates() -> npt.NDArray[np.float64]:
        """Returns the values of the isoparametric coordinates at the nodes.

        Returns:
            Values of the isoparametric coordinates at the nodes.
        """
        return np.array(
            [
                [-1.0, -1.0],  # node 1
                [1.0, -1.0],  # node 2
                [1.0, 1.0],  # node 3
                [-1.0, 1.0],  # node 4
                [0.0, -1.0],  # node 5
                [1.0, 0.0],  # node 6
                [0.0, 1.0],  # node 7
                [-1.0, 0.0],  # node 8
            ]
        )

    def extrapolate_gauss_points_to_nodes(self) -> npt.NDArray[np.float64]:
        """Returns the extrapolation matrix for a Quad8 element.

        Returns:
            Extrapolation matrix.
        """
        # get isoparametric coordinates of gauss element at acutal element nodes
        gauss_iso_coords = (
            np.array(self.nodal_isoparametric_coordinates()) * np.sqrt(15.0) / 3.0
        )

        # initialise extrapolation matrix
        ex_mat = np.zeros((self.num_nodes, self.int_points**2))

        # build extrapolation matrix
        for idx, gic in enumerate(gauss_iso_coords):
            iso_coords = gic[0], gic[1], 0.0  # create iso_coords tuple

            # evaluate shape function at guassian element iso coords
            # note shape functions of the gaussian element are for a Quad9 element
            ex_mat[idx, :] = Quad9.shape_functions(iso_coords=iso_coords)

        return ex_mat

    def get_triangulation(self) -> list[tuple[int, int, int]]:
        """Returns a list of triangle indexes for a Quad8 element.

        Returns:
            List of triangle indexes.
        """
        return [
            (self.node_idxs[0], self.node_idxs[4], self.node_idxs[7]),
            (self.node_idxs[4], self.node_idxs[1], self.node_idxs[5]),
            (self.node_idxs[4], self.node_idxs[5], self.node_idxs[7]),
            (self.node_idxs[5], self.node_idxs[2], self.node_idxs[6]),
            (self.node_idxs[6], self.node_idxs[3], self.node_idxs[7]),
            (self.node_idxs[7], self.node_idxs[5], self.node_idxs[6]),
        ]


class Quad9(QuadrilateralElement):
    """Class for a nine-noded quadratic quadrilateral element."""

    def __init__(
        self,
        el_idx: int,
        el_tag: int,
        coords: npt.NDArray[np.float64],
        node_idxs: list[int],
        material: Material,
        orientation: bool,
    ) -> None:
        """Inits the Quad9 class.

        Args:
            el_idx: Element index.
            el_tag: Element mesh tag.
            coords: A ``2 x 9`` :class:`numpy.ndarray` of coordinates defining the
                element, i.e. ``[[x1, x2, ..., x9], [y1, y2, ..., y9]]``.
            node_idxs: A list of node indexes defining the element, i.e.
                ``[idx1, idx2, ..., idx9]``.
            material: Material of the element.
            orientation: If ``True`` the element is oriented correctly, if ``False`` the
                element's nodes will need reordering.
        """
        # reorient node indexes and coords if required
        if not orientation:
            node_idxs[1], node_idxs[3] = node_idxs[3], node_idxs[1]
            node_idxs[4], node_idxs[7] = node_idxs[7], node_idxs[4]
            node_idxs[5], node_idxs[6] = node_idxs[6], node_idxs[5]
            coords[:, [1, 3, 4, 7, 5, 6]] = coords[:, [3, 1, 7, 4, 6, 5]]

        super().__init__(
            el_idx=el_idx,
            el_tag=el_tag,
            coords=coords,
            node_idxs=node_idxs,
            material=material,
            orientation=orientation,
            num_nodes=9,
            int_points=3,
        )

    @staticmethod
    def shape_functions(
        iso_coords: tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Returns the shape functions at a point for a Quad9 element.

        Args:
            iso_coords: Location of the point in isoparametric coordinates (note last
                value is ignored).

        Returns:
            The values of the shape functions ``[N1, N2, ..., N9]``.
        """
        # location of isoparametric coordinates
        xi, eta, _ = iso_coords

        # generate the shape functions for a Quad9 element
        return np.array(
            [
                0.25 * (1.0 - xi) * (1.0 - eta) * xi * eta,
                -0.25 * (1.0 + xi) * (1.0 - eta) * xi * eta,
                0.25 * (1.0 + xi) * (1.0 + eta) * xi * eta,
                -0.25 * (1.0 - xi) * (1.0 + eta) * xi * eta,
                -0.5 * (1.0 - xi**2) * (1.0 - eta) * eta,
                0.5 * (1.0 + xi) * (1.0 - eta**2) * xi,
                0.5 * (1.0 - xi**2) * (1.0 + eta) * eta,
                -0.5 * (1.0 - xi) * (1.0 - eta**2) * xi,
                (1.0 - xi**2) * (1.0 - eta**2),
            ]
        )

    @staticmethod
    def shape_functions_derivatives(
        iso_coords: tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Returns the derivatives of the shape functions at a pt for a Quad9 element.

        Args:
            iso_coords: Location of the point in isoparametric coordinates (note last
                value is ignored).

        Returns:
            The partial derivatives of the shape functions.
        """
        # location of isoparametric coordinates
        xi, eta, _ = iso_coords

        # derivatives of the shape functions wrt the isoparametric coordinates
        return np.array(
            [
                # d/d(xi)
                [
                    eta * (xi * (0.5 * eta - 0.5) - 0.25 * eta + 0.25),
                    eta * (eta * (0.5 * xi + 0.25) - 0.5 * xi - 0.25),
                    0.5 * eta * (eta + 1.0) * (xi + 0.5),
                    0.5 * eta * (eta + 1.0) * (xi - 0.5),
                    -1.0 * (eta - 1) * eta * xi,
                    eta**2 * (-xi - 0.5) + xi + 0.5,
                    -eta * (eta + 1.0) * xi,
                    eta**2 * (0.5 - xi) + xi - 0.5,
                    2.0 * (eta**2 - 1.0) * xi,
                ],
                # d/d(eta)
                [
                    xi * (eta * (0.5 * xi - 0.5) - 0.25 * xi + 0.25),
                    0.5 * (eta - 0.5) * xi * (xi + 1.0),
                    0.5 * (eta + 0.5) * xi * (xi + 1.0),
                    xi * (eta * (0.5 * xi - 0.5) + 0.25 * xi - 0.25),
                    eta * (1.0 - xi**2) + 0.5 * xi**2 - 0.5,
                    -eta * xi * (xi + 1.0),
                    eta * (1.0 - xi**2) - 0.5 * xi**2 + 0.5,
                    -eta * (xi - 1.0) * xi,
                    2.0 * eta * (xi**2 - 1.0),
                ],
            ]
        )

    @staticmethod
    def nodal_isoparametric_coordinates() -> npt.NDArray[np.float64]:
        """Returns the values of the isoparametric coordinates at the nodes.

        Returns:
            Values of the isoparametric coordinates at the nodes.
        """
        return np.array(
            [
                [-1.0, -1.0],  # node 1
                [1.0, -1.0],  # node 2
                [1.0, 1.0],  # node 3
                [-1.0, 1.0],  # node 4
                [0.0, -1.0],  # node 5
                [1.0, 0.0],  # node 6
                [0.0, 1.0],  # node 7
                [-1.0, 0.0],  # node 8
                [0.0, 0.0],  # node 9
            ]
        )

    def extrapolate_gauss_points_to_nodes(self) -> npt.NDArray[np.float64]:
        """Returns the extrapolation matrix for a Quad9 element.

        Returns:
            Extrapolation matrix.
        """
        # get isoparametric coordinates of gauss element at acutal element nodes
        gauss_iso_coords = (
            np.array(self.nodal_isoparametric_coordinates()) * np.sqrt(15.0) / 3.0
        )

        # initialise extrapolation matrix
        ex_mat = np.zeros((self.num_nodes, self.num_nodes))

        # build extrapolation matrix
        for idx, gic in enumerate(gauss_iso_coords):
            iso_coords = gic[0], gic[1], 0.0  # create iso_coords tuple

            # evaluate shape function at guassian element iso coords
            ex_mat[idx, :] = self.shape_functions(iso_coords=iso_coords)

        return ex_mat

    def get_triangulation(self) -> list[tuple[int, int, int]]:
        """Returns a list of triangle indexes for a Quad9 element.

        Returns:
            List of triangle indexes.
        """
        return [
            (self.node_idxs[0], self.node_idxs[4], self.node_idxs[8]),
            (self.node_idxs[4], self.node_idxs[1], self.node_idxs[8]),
            (self.node_idxs[1], self.node_idxs[5], self.node_idxs[8]),
            (self.node_idxs[5], self.node_idxs[2], self.node_idxs[8]),
            (self.node_idxs[2], self.node_idxs[6], self.node_idxs[8]),
            (self.node_idxs[6], self.node_idxs[3], self.node_idxs[8]),
            (self.node_idxs[3], self.node_idxs[7], self.node_idxs[8]),
            (self.node_idxs[7], self.node_idxs[0], self.node_idxs[8]),
        ]


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
        iso_coord: list[float],
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
            n, j = self.shape_functions_jacobian(iso_coord=gauss_point[1:])

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
        iso_coord: list[float],
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Evaluates the shape functions and jacobian of the element.

        Args:
            iso_coord: Location of the point in isoparametric coordinates.

        Returns:
            Shape functions and jacobian.
        """
        xi = iso_coord[0]  # isoparametric coordinate
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
        iso_coord: list[float],
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Evaluates the shape functions and jacobian of the element.

        Args:
            iso_coord: Location of the point in isoparametric coordinates.

        Returns:
            Shape functions and jacobian.
        """
        xi = iso_coord[0]  # isoparametric coordinate

        # shape functions
        n = np.array([-0.5 * xi * (1 - xi), 0.5 * xi * (1 + xi), 1 - xi**2])

        # derivative of shape functions
        b_iso = np.array([xi - 0.5, xi + 0.5, -2 * xi])
        j = b_iso @ self.coords.transpose()
        jacobian = np.sqrt(np.sum(j**2))

        return n, jacobian
