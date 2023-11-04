"""Class for a planestress analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp_sparse

import planestress.analysis.solver as solver
from planestress.analysis.utils import dof_map
from planestress.post.results import Results


if TYPE_CHECKING:
    from planestress.pre.geometry import Geometry
    from planestress.pre.load_case import LoadCase
    from planestress.pre.mesh import Mesh


class PlaneStress:
    """Class for a plane-stress analysis.

    Attributes:
        geometry: ``Geometry`` object containing a meshed geometry.
        load_cases: List of load cases to analyse.
        mesh: ``Mesh`` object.
    """

    def __init__(
        self,
        geometry: Geometry,
        load_cases: list[LoadCase],
    ) -> None:
        """Inits the PlaneStress class.

        Args:
            geometry: ``Geometry`` object containing a meshed geometry.
            load_cases: List of load cases to analyse.

        Raises:
            RuntimeError: If there is no mesh in the ``Geometry`` object.
        """
        self.geometry = geometry
        self.load_cases = load_cases

        # check mesh has been created
        if len(self.geometry.mesh.nodes) < 1:
            raise RuntimeError(
                "No mesh detected, run Geometry.create_mesh() before creating a "
                "PlaneStress object."
            )

        self.mesh: Mesh = self.geometry.mesh

        # reset boundary conditions mesh tags
        for load_case in self.load_cases:
            load_case.reset_mesh_tags()

        # assign mesh tags to boundary conditions
        for load_case in self.load_cases:
            load_case.assign_mesh_tags(mesh=self.mesh)

    def solve(
        self,
        solver_type: str = "direct",
    ) -> list[Results]:
        """Solves each load case.

        Args:
            solver_type: Solver type, either ``"direct"`` (SciPy SuperLU sparse solver)
                or ``"pardiso"`` (Intel oneAPI Math Kernel Library PARDISO solver).
                Defaults to ``"direct"``.

        Returns:
            A list of ``Results`` objects for post-processing corresponding to each load
            case.
        """
        # get number of degrees of freedom
        num_dofs = self.mesh.num_nodes() * 2

        # allocate results
        results: list[Results] = []
        row_idxs: list[int] = []
        col_idxs: list[int] = []
        k_list: list[float] = []

        # assemble stiffness matrix
        for el in self.mesh.elements:
            row_idxs.extend(el.element_row_indexes())
            col_idxs.extend(el.element_col_indexes())
            k_list.extend(el.element_stiffness_matrix())

        # construct sparse matrix in COOrdinate format and convert to list of lists
        k = sp_sparse.coo_array(
            arg1=(k_list, (row_idxs, col_idxs)),
            shape=(num_dofs, num_dofs),
            dtype=np.float64,
        ).tolil()

        # for each load case
        for lc in self.load_cases:
            # initialise modified stiffness matrix
            k_mod: sp_sparse.lil_array = k.copy()

            # initialise load vector
            f = np.zeros(num_dofs)
            f_app: npt.NDArray[np.float64] | None = None

            # assemble load vector
            for el in self.mesh.elements:
                # get element load vector
                f_el = el.element_load_vector(acceleration_field=lc.acceleration_field)

                # get element degrees of freedom
                el_dofs = dof_map(node_idxs=tuple(el.node_idxs))

                # add element load vector to global load vector
                f[el_dofs] += f_el

            # apply boundary conditions
            # note these are sorted (load -> spring -> support)
            for boundary_condition in lc.boundary_conditions:
                # check to see if we have finished applying external loads
                if boundary_condition.priority > 0 and f_app is None:
                    f_app = np.copy(f)

                # apply boundary condition
                k_mod, f = boundary_condition.apply_bc(k=k_mod, f=f)

            # ensure f_app has been generated
            if f_app is None:
                f_app = f

            # solve system
            if solver_type == "direct":
                u = solver.solve_direct(k=k_mod, f=f)
            elif solver_type == "pardiso":
                u = solver.solve_pardiso(k=k_mod, f=f)
            else:
                raise ValueError(
                    f"'solver_type' must be 'direct' or 'pardiso', not {solver_type}."
                )

            # post-processing
            res = Results(plane_stress=self, u=u)
            res.calculate_node_forces(k=k)
            res.calculate_reactions(f=f_app)
            res.calculate_stresses(elements=self.mesh.elements)

            # add to results list
            results.append(res)

        return results
