"""Methods used for solving linear systems and displaying info on tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import spsolve


if TYPE_CHECKING:
    from scipy.sparse import lil_array


def solve_direct(
    k: npt.NDArray[np.float64],
    f: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solves a linear system using the direct solver method.

    Args:
        k: ``N x N`` matrix of the linear system.
        f: ``N x 1`` right hand side of the linear system.

    Returns:
        The solution vector to the linear system of equations.
    """
    return np.linalg.solve(a=k, b=f)


def solve_direct_sparse(
    k: lil_array,
    f: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solves a sparse linear system using the direct solver method.

    Args:
        k: ``N x N`` sparse matrix of the linear system.
        f: ``N x 1`` right hand side of the linear system.

    Returns:
        The solution vector to the sparse linear system of equations.
    """
    k_csc = k.tocsc()
    k_csc.eliminate_zeros()

    return spsolve(A=k_csc, b=f)  # type: ignore
