import numpy as np
from typing import Literal

from source.kernels import G, dG_dn_y, dG_dr, r_vec
from source.quadrature import (standard_triangle_quad, 
                               map_to_physical_triangle, 
                               shape_functions_P1)
from source.matrix_assembly import CollocationAssembler


class BEMSolver:
    """
    Boundary Element Method solver for acoustic problems.

    Handles matrix assembly, application of boundary conditions, and
    solution of the linear system.

    Attributes:
        assembler (CollocationAssembler): Assembler object for building matrices.
    """

    def __init__(self, assembler: CollocationAssembler):
        """Initialize the solver.

        Args:
            assembler (CollocationAssembler): Collocation or Galerkin assembler.
        """
        self.assembler = assembler
        self.mesh = assembler.mesh

    def assemble_matrices(self,
                          formulation: Literal["collocation", 
                                               "galerkin"] = "collocation"
                          ) -> dict[str, np.ndarray]:
        """
        Assemble all operator matrices.

        Args:
            formulation (Literal["collocation","galerkin"], optional):
                Which formulation to use. Currently only "collocation"
                is implemented.

        Returns:
            dict[str, np.ndarray]: Dictionary with matrices ``S``, ``D``,
            ``Kp``, ``N``.
        """
        if formulation != "collocation":
            raise NotImplementedError("Galerkin support coming soon.")

        A = {}
        for op in ("S", "D", "Kp", "N"):
            A[op] = self.assembler.assemble(op)
        return A

    def solve_direct(self,
                     matrices: dict[str, np.ndarray],
                     bc_type: Literal["Dirichlet", "Neumann"] = "Neumann",
                     bc_values: np.ndarray | None = None,) -> np.ndarray:
        """
        Solve for the unknown boundary quantity.

        Args:
            matrices (dict[str, np.ndarray]): Operator matrices (``S``, ``D``,
                ``Kp``, ``N``).
            bc_type (Literal["Dirichlet","Neumann"]): Which data is prescribed.
                - "Dirichlet": known acoustic potential φ on Γ.
                - "Neumann": known normal velocity ∂φ/∂n on Γ.
            bc_values (np.ndarray): Prescribed boundary data at nodes,
                shape (num_nodes,).

        Returns:
            np.ndarray: Solution vector for the unknown boundary quantity
            at mesh nodes.
        """
        if bc_values is None:
            try:
                bc_values = self.mesh.velocity_BC
            except AttributeError:
                raise ValueError("No boundary condition values provided.")

        if bc_type == "Dirichlet":
            # Dirichlet given: φ known, ∂φ/∂n unknown
            # BIE:  0.5*φ = D φ - S q
            self.potential_BC = bc_values
            phi = bc_values
            A = -matrices["S"]
            rhs = 0.5 * phi - matrices["D"] @ phi
            sol = np.linalg.solve(A, rhs)     # q = ∂φ/∂n
            self.velocity_BC = sol
            return sol

        if bc_type == "Neumann":
            # Neumann given: q = ∂φ/∂n known, solve for φ
            self.velocity_BC = bc_values
            q = bc_values
            # C = np.diag(self.mesh.jump_coefficients)
            C = 0.5 * np.eye(self.mesh.num_nodes)
            A = matrices["D"] - C
            rhs = matrices["S"] @ q
            sol = np.linalg.solve(A, rhs)
            self.potential_BC = sol
            return sol

        raise ValueError("bc_type must be 'Dirichlet' or 'Neumann'")
    
    def solve_burton_miller(self,
                            bc_type: Literal["Dirichlet", "Neumann"],
                            bc_values: np.ndarray,
                            matrices: dict[str, np.ndarray],
                            alpha: float | None = None,
                            use_jump: Literal["constant", 
                                              "mesh"] = "constant"
                            ) -> np.ndarray:
        """
        Solve the BIE via the Burton–Miller combined formulation.

        This method forms a linear combination of the standard boundary equation
        and its normal-derivative equation to remove spurious resonances.

        The combined equation is taken (for exterior problems) in the form
            (D - C) φ - S q  + i α [ N φ + K' q ] = 0,
        where C is the double-layer jump term (typically 0.5·I on closed smooth
        surfaces).
            S  : single layer (G)
            D  : double layer (∂G/∂n_y)
            K' : adjoint double layer (∂G/∂n_x)
            N  : hypersingular (∂²G/∂n_x∂n_y)

        Given boundary data, the method solves for the complementary unknown:
        - bc_type="Neumann": given q = ∂φ/∂n, solve for φ on Γ
            [(D - C) + i α N] φ = [S + i α K'] q
        - bc_type="Dirichlet": given φ on Γ, solve for q = ∂φ/∂n on Γ
            [-S + i α K'] q = [C - D - i α N] φ

        Args:
            bc_type (Literal["Dirichlet","Neumann"]): Type of prescribed 
                boundary data.
                "Dirichlet" → known potential φ on Γ; returns q.
                "Neumann"   → known normal derivative q on Γ; returns φ.
            bc_values (np.ndarray): Prescribed nodal data (shape (num_nodes,)).
            matrices (dict[str, np.ndarray]): Operator matrices 
                {"S","D","Kp","N"}.
            alpha (float | None): Coupling parameter α. If None, 
                default alpha = i/max(self.mesh.k, 1.0).
            use_jump ({"constant","mesh","auto"}): How to build the jump matrix 
                C.
                - "constant": C = 0.5·I
                - "mesh":     C = diag(self.mesh.jump_coeff)

        Returns:
            np.ndarray: Solution vector (φ for Neumann input, 
                or q for Dirichlet input).
        """
        S = matrices["S"]
        D = matrices["D"]
        Kp = matrices["Kp"]
        N  = matrices["N"]

        # Jump term C
        if use_jump == "mesh" and hasattr(self.mesh, "jump_coeff"):
            C = np.diag(self.mesh.jump_coeff.astype(complex))
        else:
            C = 0.5 * np.eye(self.mesh.num_nodes, dtype=complex)

        if alpha is None:
            alpha = max(float(self.mesh.k), 1.0)

        ialpha = 1j / alpha

        if bc_type == "Neumann":
            # Given q, solve for φ:
            # [(D - C) + i α N] φ = [S + i α K'] q
            q = bc_values.astype(complex, copy=False)
            A = (D - C) + ialpha * N
            rhs = (S + ialpha * Kp) @ q
            phi = np.linalg.solve(A, rhs)
            return phi

        if bc_type == "Dirichlet":
            # Given φ, solve for q:
            # [-S + i α K'] q = [C - D - i α N] φ
            phi = bc_values.astype(complex, copy=False)
            A = (-S) + ialpha * Kp
            rhs = (C - D - ialpha * N) @ phi
            q = np.linalg.solve(A, rhs)
            return q

        raise ValueError("bc_type must be 'Dirichlet' or 'Neumann'")
    
    def evaluate_field(self,
                       field_points: np.ndarray,
                       phi: np.ndarray | None = None,
                       q: np.ndarray | None = None,
                       quad_order: int = 3) -> np.ndarray:
        """
        Evaluate the potential at domain points using boundary solution.

        Args:
            field_points (np.ndarray): Array of M points, shape (M,3).
            phi (np.ndarray | None): Boundary potential at nodes, shape (N,), 
                or None.
            q (np.ndarray | None): Boundary normal derivative at nodes, shape 
                (N,), or None.
            quad_order (int, optional): Triangle quadrature order. 
                Defaults to 3.

        Returns:
            np.ndarray: Complex potential at field points, shape (M,).
        """
        if phi is None:
            try: 
                phi = self.potential_BC
            except AttributeError:
                raise ValueError("Boundary potential not found. Provide as " \
                "Dirichlet BC or run solve_direct to compute from Neumann " \
                "boundary conditions.")
            
        if q is None:
            try:
                q = self.velocity_BC
            except AttributeError:
                raise ValueError("Boundary normal derivative not found. " \
                "Provide as Neumann BC or run solve_direct to compute from " \
                "Dirichlet boundary conditions.")

        xi_eta, w = standard_triangle_quad(quad_order)
        u = np.zeros(field_points.shape[0], dtype=complex)

        for e in range(self.mesh.num_elements):
            conn = self.mesh.mesh_elements[e]
            v0, e1, e2 = self.mesh.v0[e], self.mesh.e1[e], self.mesh.e2[e]
            ny = self.mesh.n_hat[e]

            yq, a2 = map_to_physical_triangle(xi_eta, v0, e1, e2)
            w_phys = w * a2
            Nq = shape_functions_P1(xi_eta)
            if phi is not None:
                phi_q = Nq @ phi[conn]
            if q is not None:
                q_q = Nq @ q[conn]

            _, r_norm, r_hat = r_vec(field_points[:, None, :], yq[None, :, :])
            Gvals = G(r_norm, self.mesh.k)

            if phi is not None:
                dGr = dG_dr(r_norm, Gvals, self.mesh.k)
                ny_b = ny[None, None, :]
                dGdnY = dG_dn_y(r_hat, dGr, ny_b)
                u += np.sum(dGdnY * (phi_q[None, :] * w_phys[None, :]), axis=1)
            if q is not None:
                u -= np.sum(Gvals * (q_q[None, :] * w_phys[None, :]), axis=1)

        return u