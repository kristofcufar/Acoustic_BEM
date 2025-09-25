import numpy as np
from typing import Literal

from source.kernels import G, dG_dn_y, dG_dr, r_vec
from source.quadrature import (standard_triangle_quad, 
                               map_to_physical_triangle, 
                               shape_functions_P1)
from source.matrix_assembly import (CollocationAssembler, 
                                    GalerkinAssembler)


class BEMSolver:
    """
    Boundary Element Method solver for acoustic problems.

    Handles matrix assembly, application of boundary conditions, and
    solution of the linear system.

    Attributes:
        assembler (CollocationAssembler | GalerkinAssembler): Assembler object 
            for building matrices.
    """

    def __init__(self, assembler: CollocationAssembler | GalerkinAssembler):
        """Initialize the solver.

        Args:
            assembler (CollocationAssembler | GalerkinAssembler): Collocation 
                or Galerkin assembler.
        """
        self.assembler = assembler
        self.mesh = assembler.mesh
        self.is_galerkin = isinstance(assembler, GalerkinAssembler)


    def assemble_matrices(self, 
                          ops: tuple[str, ...] = ("S","D","Kp","N", "NReg"),
                          ) -> dict[str, np.ndarray]:
        """
        Assemble selected operator matrices.

        Args:
            ops (tuple[str, ...]): Any subset of {"S","D","Kp","N", "NReg"}.

        Returns:
            dict[str, np.ndarray]: Assembled matrices.
        """
        A: dict[str, np.ndarray] = {}
        for op in ops:
            A[op] = self.assembler.assemble(op)
        return A

    def solve_direct(self,
                     matrices: dict[str, np.ndarray] | None = None,
                     jump_coeff: np.ndarray | None = None) -> np.ndarray:
        """
        Solve for the unknown boundary quantity.

        Args:
            matrices (dict[str, np.ndarray] | None): Pre-assembled operator 
                matrices {"S","D"}. If None, assembles them.
            jump_coeff (np.ndarray | None): Jump coefficients at nodes,
                shape (num_nodes,). If None, uses the mesh's jump_coefficients
                attribute (based on solid angle) if it exists, otherwise 
                defaults to 0.5.

        Returns:
            np.ndarray: Solution vector for the unknown boundary quantity
            at mesh nodes.
        """
        if matrices is None:
            matrices = self.assemble_matrices(ops=("S","D"))

        if self.mesh.Dirichlet_BC is not None:
            bc_type = "Dirichlet"
            bc_values = self.mesh.Dirichlet_BC
        elif self.mesh.Neumann_BC is not None:
            bc_type = "Neumann"
            bc_values = self.mesh.Neumann_BC
        else:
            raise ValueError("No boundary condition values provided.")
            
        S = matrices["S"]
        D = matrices["D"]

        if self.is_galerkin:
            C = self.assembler.assemble("M")

        else:
            if jump_coeff is None:
                try:
                    C = np.diag(self.mesh.jump_coefficients)
                except AttributeError:
                    C = 0.5 * np.eye(self.mesh.num_nodes)
            else:
                C = np.diag(jump_coeff)

        if bc_type == "Dirichlet":            
            phi = bc_values
            A = -S
            rhs = 0.5 * phi - D @ phi
            sol = np.linalg.solve(A, rhs)
            self.velocity_BC = sol
            self.potential_BC = bc_values
            return sol

        if bc_type == "Neumann":
            q = bc_values
            A = D - C
            rhs = S @ q
            sol = np.linalg.solve(A, rhs)
            self.potential_BC = sol
            self.velocity_BC = bc_values
            return sol
    
    def solve_burton_miller(self,
                            matrices: dict[str, np.ndarray] | None = None,
                            jump_coeff: np.ndarray | None = None,
                            alpha: float | None = None,
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
            matrices (dict[str, np.ndarray] | None): Pre-assembled operator 
                matrices {"S","D","Kp","N"}. If None, assembles them.
            jump_coeff (np.ndarray | None): Jump coefficients at nodes,
                shape (num_nodes,). If None, uses the mesh's jump_coefficients
                attribute (based on solid angle) if it exists, otherwise 
                defaults to 0.5.
            alpha (float | None): Coupling parameter α. If None, defaults to 
                α = 1j/max(k,1).

        Returns:
            np.ndarray: Solution vector (φ for Neumann input, 
                or q for Dirichlet input).
        """

        if matrices is None:
            matrices = self.assemble_matrices(ops=("S","D","Kp","N"))

        if self.mesh.Dirichlet_BC is not None:
            bc_type = "Dirichlet"
            bc_values = self.mesh.Dirichlet_BC

        elif self.mesh.Neumann_BC is not None:
            bc_type = "Neumann"
            bc_values = self.mesh.Neumann_BC

        S = matrices["S"]
        D = matrices["D"]
        Kp = matrices["Kp"]
        N  = matrices["N"]

        if self.is_galerkin:
            C = self.assembler.assemble("M")

        else:
            if jump_coeff is None:
                try:
                    C = np.diag(self.mesh.jump_coefficients)
                except AttributeError:
                    C = 0.5 * np.eye(self.mesh.num_nodes)
            else:
                C = np.diag(jump_coeff)

        if alpha is None:
            ialpha = 1j / max(float(self.mesh.k), 1.0)
        else:
            if not np.iscomplexobj(alpha):
                ialpha = 1j * alpha
            else:
                ialpha = alpha

        if bc_type == "Neumann":
            q = bc_values.astype(complex, copy=False)
            A = (D - C) + ialpha * N
            rhs = (S - ialpha * Kp) @ q
            phi = np.linalg.solve(A, rhs)
            self.potential_BC = phi
            self.velocity_BC = bc_values
            return phi

        if bc_type == "Dirichlet":
            phi = bc_values.astype(complex, copy=False)
            A = (-S) + ialpha * Kp
            rhs = (C - D - ialpha * N) @ phi
            q = np.linalg.solve(A, rhs)
            self.velocity_BC = q
            self.potential_BC = bc_values
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