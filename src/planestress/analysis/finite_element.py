"""Finite element classes for a plane-stress analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt


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

    def gauss_points(
        self,
        n_points: int,
    ) -> npt.NDArray[np.float64]:
        """Gaussian weights and locations for ``n_point`` Gaussian integration.

        Args:
            n_points: Number of gauss points.

        Raises:
            NotImplementedError: If this method hasn't been implemented for an element.
        """
        raise NotImplementedError

    @staticmethod
    def shape_functions(
        iso_coords: tuple[float, float, float]
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
        iso_coords: tuple[float, float, float]
    ) -> npt.NDArray[np.float64]:
        """Returns the derivatives of the shape functions at a point.

        Args:
            iso_coords: Location of the point in isoparametric coordinates.

        Raises:
            NotImplementedError: If this method hasn't been implemented for an element.
        """
        raise NotImplementedError

    def iso_to_global(
        self,
        iso_coords: tuple[float, float, float],
    ) -> tuple[float, float]:
        """Converts a point in isoparametric coordinates to global coordinates.

        Args:
            iso_coords: Location of the point in isoparametric coordinates.

        Returns:
            Location of the point in global coordinates (``x``, ``y``).
        """
        x, y = self.coords @ self.shape_functions(iso_coords=iso_coords)

        return x, y

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

    def element_stiffness_matrix(self) -> npt.NDArray[np.float64]:
        """Assembles the stiffness matrix for the element.

        Returns:
            Element stiffness matrix.
        """
        # allocate element stiffness matrix
        k_el = np.zeros((2 * self.num_nodes, 2 * self.num_nodes))

        # get d_matrix
        d_mat = self.material.get_d_matrix()

        # get Gauss points
        gauss_points = self.gauss_points(n_points=self.int_points)

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

        # get Gauss points
        gauss_points = self.gauss_points(n_points=self.int_points)

        # loop through each gauss point
        for gauss_point in gauss_points:
            # get shape functions and jacobian
            n = self.shape_functions(iso_coords=gauss_point[1:])
            _, j = self.b_matrix_jacobian(iso_coords=gauss_point[1:])

            # form shape function matrix
            n_mat = np.zeros((len(n) * 2, 2))
            n_mat[::2, 0] = n
            n_mat[1::2, 1] = n

            # calculate load vector for current integration point
            f_el += n_mat @ b * gauss_point[0] * j * self.material.thickness

        return f_el

    def get_element_results(
        self,
        u: npt.NDArray[np.float64],
    ) -> ElementResults:
        """Calculates various results for the finite element given nodal displacements.

        Calculates the following:

        - Stresses at nodes
        - TODO

        Args:
            u: Displacement vector for the element.

        Returns:
            ``ElementResults`` object.
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
            el_tag=self.el_tag,
            coords=self.coords,
            node_idxs=self.node_idxs,
            material=self.material,
            orientation=self.orientation,
            num_nodes=self.num_nodes,
            int_points=self.int_points,
            sigs=sigs,
        )

    def get_triangulation(self) -> list[tuple[int, int, int]]:
        """Returns a list of triangle indices for the finite element.

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

    def gauss_points(
        self,
        n_points: int,
    ) -> npt.NDArray[np.float64]:
        """Gaussian weights and locations for ``n_point`` Gaussian integration.

        Args:
            n_points: Number of gauss points.

        Raises:
            ValueError: If ``n_points`` is not 1, 3, 4 or 6.

        Returns:
            Gaussian weights and locations. For each gauss point -
            ``[weight, eta, xi, zeta]``.
        """
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

        raise ValueError(
            f"'n_points' must be 1, 3, 4 or 6 for a {self.__class__.__name__} element."
        )

    def b_matrix_jacobian(
        self,
        iso_coords: tuple[float, float, float],
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Calculates the B matrix and jacobian at an isoparametric point.

        Args:
            iso_coords: Location of the point in isoparametric coordinates.

        Returns:
            Derivatives of the shape function (B matrix) and value of the jacobian,
            (``b_mat``, ``j``).
        """
        # TODO - is this general for rectangular as well? probably not... iso_coords
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
        eta, xi, zeta = iso_coords

        # for a Tri3, the shape functions are the isoparametric coordinates
        return np.array([eta, xi, zeta])

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
                [1.0, 0.0, 0.0],  # d/d(eta)
                [0.0, 1.0, 0.0],  # d/d(xi)
                [0.0, 0.0, 1.0],  # d/d(zeta)
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
        """Returns a list of triangle indices for a Tri3 element.

        Returns:
            List of triangle indices.
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
        # reorient node indexes and coords if required - TODO
        if not orientation:
            raise NotImplementedError

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
        """Returns the derivatives of the shape functions at a point for a Tri6 element.

        Args:
            iso_coords: Location of the point in isoparametric coordinates.

        Returns:
            The partial derivatives of the shape functions.
        """
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


class RectangularElement:
    """Abstract base class for a rectangular plane-stress finite element."""

    pass


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

    def gauss_points(
        self,
        n_points: int,
    ) -> npt.NDArray[np.float64]:
        """Gaussian weights and locations for ``n_point`` Gaussian integration.

        Args:
            n_points: Number of gauss points.

        Raises:
            ValueError: If ``n_points`` is not 1 or 2.

        Returns:
            Gaussian weights and location. For each gauss point - ``[weight, eta]``.
        """
        # one point gaussian integration
        if n_points == 1:
            return np.array([[2.0, 0.0]])

        # two point gaussian integration
        if n_points == 2:
            return np.array(
                [
                    [1.0, -1 / np.sqrt(3)],
                    [1.0, 1 / np.sqrt(3)],
                ]
            )

        raise ValueError(
            f"'n_points' must be 1, or 2 for a {self.__class__.__name__} element."
        )

    def shape_functions_length(
        self,
        iso_coord: list[float],
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Evaluates the shape functions and length of the element.

        TODO - change this to jacobian.

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
            b = np.array([1, 0])
        elif direction == "y":
            b = np.array([0, 1])
        else:
            b = np.array([1, 1])

        b *= value

        # get Gauss points
        gauss_points = self.gauss_points(n_points=self.int_points)

        # loop through each gauss point
        for gauss_point in gauss_points:
            # get shape functions and length
            n, l = self.shape_functions_length(iso_coord=gauss_point[1:])

            # form shape function matrix
            n_mat = np.zeros((len(n) * 2, 2))
            n_mat[::2, 0] = n
            n_mat[1::2, 1] = n

            # calculate load vector for current integration point
            f_el += n_mat @ b * gauss_point[0] * 0.5 * l

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

    def shape_functions_length(
        self,
        iso_coord: list[float],
    ) -> tuple[npt.NDArray[np.float64], float]:
        """Evaluates the shape functions and length of the element.

        TODO - change this to jacobian.

        Args:
            iso_coord: Location of the point in isoparametric coordinates.

        Returns:
            Length of the element.
        """
        eta = iso_coord[0]
        n = np.array([0.5 - 0.5 * eta, 0.5 + 0.5 * eta])
        length = np.sqrt(
            (self.coords[0, 1] - self.coords[0, 0]) ** 2
            + (self.coords[1, 1] - self.coords[1, 0]) ** 2
        )
        return n, length


class ElementResults(FiniteElement):
    """Class for storing the results of a finite element."""

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
        sigs: npt.NDArray[np.float64],
    ) -> None:
        """Inits the ElementResults class.

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
            sigs: Nodal stresses, e.g.
                ``[[sigxx_1, sigyy_1, sigxy_1], ..., [sigxx_3, sigyy_3, sigxy_3]]``.
        """
        super().__init__(
            el_idx=el_idx,
            el_tag=el_tag,
            coords=coords,
            node_idxs=node_idxs,
            material=material,
            num_nodes=num_nodes,
            int_points=int_points,
            orientation=orientation,
        )
        self.sigs = sigs
