"""Methods used for solving linear systems and displaying info on tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

try:
    import pypardiso

    pardiso_solve = pypardiso.spsolve
except ImportError:
    pardiso_solve = None


if TYPE_CHECKING:
    from scipy.sparse import lil_array


def solve_direct(
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

    return spsolve(A=k_csc, b=f)


def solve_pardiso(
    k: lil_array,
    f: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Solves a sparse linear system using the pardiso solver.

    Args:
        k: ``N x N`` sparse matrix of the linear system.
        f: ``N x 1`` right hand side of the linear system.

    Raises:
        RuntimeError: If ``pypardiso`` is not installed.

    Returns:
        The solution vector to the sparse linear system of equations.
    """
    if pardiso_solve is not None:
        k_csc = csc_matrix(k)
        k_csc.eliminate_zeros()

        return pardiso_solve(A=k_csc, b=f)
    else:
        raise RuntimeError(
            "pypardiso not installed, install using the pardiso option, 'pip install "
            "planestress[pardiso]'."
        )
