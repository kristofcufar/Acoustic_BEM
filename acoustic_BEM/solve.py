import numpy as np

from acoustic_BEM.kernels import (G, dG_dn_y, 
                                  dG_dr, r_vec)
from acoustic_BEM.quadrature import (standard_triangle_quad, 
                               map_to_physical_triangle_batch,
                               shape_functions_P1)
from acoustic_BEM.matrix_assembly import CollocationAssembler

from tqdm.notebook import tqdm


class BEMSolver:
    """
    Boundary Element Method solver for acoustic problems.

    Handles matrix assembly, application of boundary conditions, and
    solution of the linear system.

    Attributes:
        assembler (CollocationAssembler): Assembler object 
            for building matrices.
    """

    def __init__(self, assembler: CollocationAssembler):
        """Initialize the solver.

        Args:
            assembler (CollocationAssembler): Collocation assembler.
        """
        self.assembler = assembler
        self.mesh = assembler.mesh


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
        Solve equation:

            (D - C) φ = S q
        
        for the unknown boundary quantity. For exterior problems:

        - bc_type="Dirichlet": given φ on Γ, solve for q = ∂φ/∂n on Γ
        - bc_type="Neumann": given q = ∂φ/∂n on Γ, solve for φ on Γ



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

        if jump_coeff is None:
            try:
                C = np.diag(self.mesh.jump_coefficients)
            except AttributeError:
                C = 0.5 * np.eye(self.mesh.num_nodes)
        else:
            C = np.diag(jump_coeff)

        if bc_type == "Neumann":
            q = bc_values
            A = D - C
            rhs = S @ q
            sol = np.linalg.solve(A, rhs)
            self.potential_BC = sol
            self.velocity_BC = bc_values
            return sol

        if bc_type == "Dirichlet":            
            phi = bc_values
            A = S
            rhs = (D - C) @ phi
            sol = np.linalg.solve(A, rhs)
            self.velocity_BC = sol
            self.potential_BC = bc_values
            return sol
    
    def solve_burton_miller(self,
                            matrices: dict[str, np.ndarray] | None = None,
                            jump_coeff: np.ndarray | None = None,
                            alpha: complex = 1j,
                            ) -> np.ndarray:
        """
        Solve the BIE via the Burton–Miller combined formulation.

        This method forms a linear combination of the standard boundary equation
        and its normal-derivative equation to remove spurious resonances.

        The combined equation is taken (for exterior problems) in the form:
            (D - C) φ + α N φ = S q + α (C + K') q,
            [(D - C) + α N] φ = [S + α (C + K')] q

        where C is the double-layer jump term (typically 0.5·I on closed smooth
        surfaces).
            S  : single layer (G)
            D  : double layer (∂G/∂n_y)
            K' : adjoint double layer (∂G/∂n_x)
            N  : hypersingular (∂²G/∂n_x∂n_y)

        Given boundary data, the method solves for the complementary unknown:
        - bc_type="Neumann": given q = ∂φ/∂n, solve for φ on Γ
        - bc_type="Dirichlet": given φ on Γ, solve for q = ∂φ/∂n on Γ

        Args:
            matrices (dict[str, np.ndarray] | None): Pre-assembled operator 
                matrices {"S","D","Kp","N"}. If None, assembles them.
            jump_coeff (np.ndarray | None): Jump coefficients at nodes,
                shape (num_nodes,). If None, uses the mesh's jump_coefficients
                attribute (based on solid angle) if it exists, otherwise 
                defaults to 0.5.
            alpha (complex): Coupling parameter α. Defaults to 1j.

        Returns:
            np.ndarray: Solution vector (φ for Neumann input, 
                or q for Dirichlet input).
        """

        if matrices is None:
            matrices = self.assemble_matrices(ops=("S","D","Kp","NReg"))

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

        if jump_coeff is None:
            try:
                C = np.diag(self.mesh.jump_coefficients)
            except AttributeError:
                C = 0.5 * np.eye(self.mesh.num_nodes)
        else:
            C = np.diag(jump_coeff)

        if bc_type == "Neumann":
            q = bc_values.astype(complex, copy=False)
            A = (D - C) + alpha * N
            rhs = (S + alpha * (C + Kp)) @ q
            phi = np.linalg.solve(A, rhs)
            self.potential_BC = phi
            self.velocity_BC = bc_values
            return phi

        if bc_type == "Dirichlet":
            phi = bc_values.astype(complex, copy=False)
            A = S + alpha * (C + Kp)
            rhs = (D - C + alpha * N) @ phi
            q = np.linalg.solve(A, rhs)
            self.velocity_BC = q
            self.potential_BC = bc_values
            return q

        raise ValueError("bc_type must be 'Dirichlet' or 'Neumann'")
    
    def evaluate_field(self,
                       field_points: np.ndarray,
                       phi: np.ndarray | None = None,
                       q: np.ndarray | None = None,
                       quad_order: int = 3,
                       verbose: bool = True) -> np.ndarray:
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
        Nq = shape_functions_P1(xi_eta)
        yq, a2 = map_to_physical_triangle_batch(xi_eta,
                                               self.mesh.v0, self.mesh.e1, self.mesh.e2)
        w_phys = w[None, :, None] * a2[:, None, None]

        r_norm, r_hat = r_vec(field_points[:, None, None, :], yq[None, :, :, :])[1:]

        Gvals = G(r_norm, self.mesh.k)
        dGr = dG_dr(r_norm, Gvals, self.mesh.k)

        ny_b = self.mesh.n_hat[None, :, None, :]
        dGdnY = dG_dn_y(r_hat, dGr, ny_b)

        u = np.zeros(field_points.shape[0], dtype=complex)

        for e in tqdm(range(self.mesh.num_elements), 
                      desc="Evaluating pressure field at points",
                      disable = not verbose):
            conn = self.mesh.mesh_elements[e]

            phi_q = Nq @ phi[conn]
            q_q = Nq @ q[conn]

            wq = w_phys[e, :, 0]

            u += np.sum(dGdnY[:, e, :] * (phi_q[None, :] * wq[None, :]), axis=1)
            u -= np.sum(Gvals[:, e, :] * (q_q[None, :] * wq[None, :]), axis=1)

        return u