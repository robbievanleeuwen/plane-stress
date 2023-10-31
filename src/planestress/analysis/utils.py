"""planestress utility functions."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def dof_map(node_idxs: list[int]) -> list[int]:
    """Maps a list of node indexes to a list of degrees of freedom.

    Args:
        node_idxs: Node indexes to map.

    Returns:
        Global degrees of freedom for each node index in ``node_idxs``.
    """
    dofs = []

    for node_idx in node_idxs:
        dofs.extend([node_idx * 2, node_idx * 2 + 1])

    return dofs


def gauss_points_line(n_points: int) -> npt.NDArray[np.float64]:
    """Gaussian weights and locations for 1D line Gaussian integration.

    Args:
        n_points: Number of gauss points.

    Raises:
        ValueError: If ``n_points`` is not 1, 2 or 3.

    Returns:
        Gaussian weights and location. For each gauss point - ``[weight, xi]``.
    """
    # one point gaussian integration
    if n_points == 1:
        return np.array([[2.0, 0.0]])

    # two point gaussian integration
    if n_points == 2:
        return np.array(
            [
                [1.0, -1.0 / np.sqrt(3)],
                [1.0, 1.0 / np.sqrt(3)],
            ]
        )

    # three point gaussian integration
    if n_points == 3:
        return np.array(
            [
                [5.0 / 9.0, -np.sqrt(3.0 / 5.0)],
                [8.0 / 9.0, 0.0],
                [5.0 / 9.0, np.sqrt(3.0 / 5.0)],
            ]
        )

    raise ValueError("'n_points' must be 1, 2 or 3.")


def gauss_points_triangle(n_points: int) -> npt.NDArray[np.float64]:
    """Gaussian weights and locations for triangular Gaussian integration.

    Args:
        n_points: Number of gauss points.

    Raises:
        ValueError: If ``n_points`` is not 1 or 3.

    Returns:
        Gaussian weights and locations. For each gauss point -
        ``[weight, zeta1, zeta2, zeta3]``.
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

    raise ValueError("'n_points' must be 1 or 3.")


def gauss_points_quad(n_points: int) -> npt.NDArray[np.float64]:
    """Gaussian weights and locations for 2D quadrangle Gaussian integration.

    Note last value in each row is ignored (placeholder).

    Args:
        n_points: Number of gauss points in each direction.

    Raises:
        ValueError: If ``n_points`` is not 1, 2 or 3.

    Returns:
        Gaussian weights and location. For each gauss point -
        ``[weight, xi, eta, 0.0]``.
    """
    # one point gaussian integration
    if n_points == 1:
        return np.array([[4.0, 0.0, 0.0, 0.0]])

    # two point gaussian integration
    if n_points == 2:
        return np.array(
            [
                [1.0, -1.0 / np.sqrt(3), -1.0 / np.sqrt(3), 0.0],
                [1.0, 1.0 / np.sqrt(3), -1.0 / np.sqrt(3), 0.0],
                [1.0, 1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 0.0],
                [1.0, -1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 0.0],
            ]
        )

    # three point gaussian integration - TODO: ordering consistency!!
    if n_points == 3:
        return np.array(
            [
                [25.0 / 81.0, -np.sqrt(3.0 / 5.0), -np.sqrt(3.0 / 5.0), 0.0],
                [40.0 / 81.0, -np.sqrt(3.0 / 5.0), 0.0, 0.0],
                [25.0 / 81.0, -np.sqrt(3.0 / 5.0), np.sqrt(3.0 / 5.0), 0.0],
                [40.0 / 81.0, 0.0, -np.sqrt(3.0 / 5.0), 0.0],
                [64.0 / 81.0, 0.0, 0.0, 0.0],
                [40.0 / 81.0, 0.0, np.sqrt(3.0 / 5.0), 0.0],
                [25.0 / 81.0, np.sqrt(3.0 / 5.0), -np.sqrt(3.0 / 5.0), 0.0],
                [40.0 / 81.0, np.sqrt(3.0 / 5.0), 0.0, 0.0],
                [25.0 / 81.0, np.sqrt(3.0 / 5.0), np.sqrt(3.0 / 5.0), 0.0],
            ]
        )

    raise ValueError("'n_points' must be 1, 2 or 3.")
