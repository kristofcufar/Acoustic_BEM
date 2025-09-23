import numpy as np
from source.quadrature import standard_triangle_quad, duffy_rule, telles_rule
from source.mesh import Mesh
from source.integrators import ElementIntegratorCollocation


class CollocationAssembler:
    """
    Collocation BEM assembler for the four boundary operators.

    This class computes collocation matrices for:

    - Single-layer potential (``S``)
    - Double-layer potential (``D``)
    - Adjoint double-layer (``Kp``)
    - Hypersingular operator (``N``)

    The assembler holds the mesh and integrator so that expensive data
    (geometry, connectivity, etc.) is reused across operators.
    """

    def __init__(self,
                 mesh: Mesh,
                 integrator: ElementIntegratorCollocation,
                 quad_order: int = 3,
                 near_threshold: float = 2.0):
        """
        Initialize the collocation assembler.

        Args:
            mesh (Mesh): Geometry and discretization data.
            integrator (ElementIntegratorCollocation): Local integration engine.
            quad_order (int, optional): Order of standard triangle quadrature.
                Defaults to 3.
            near_threshold (float, optional): Distance factor for near-singular
                detection. Defaults to 2.0.
        """
        self.mesh = mesh
        self.integrator = integrator
        self.quad_order = quad_order
        self.near_threshold = near_threshold

        self.Nn = mesh.num_nodes
        self.Ne = mesh.num_elements

    def assemble(self, operator: str) -> np.ndarray:
        """
        Assemble the collocation matrix for a boundary operator.

        Args:
            operator (str): One of ``{"S", "D", "Kp", "N", "NReg"}``.

        Returns:
            np.ndarray: Dense matrix of shape (num_nodes, num_nodes) containing
            the collocation coefficients for the selected operator.
        """
        if operator not in {"S", "D", "Kp", "N", "NReg"}:
            raise ValueError(f"Unknown operator {operator}")

        A = np.zeros((self.Nn, self.Nn), dtype=np.complex128)

        for node_idx in range(self.Nn):
            x = self.mesh.mesh_nodes[node_idx]
            n_x = self.mesh.node_n_hat[node_idx]

            sing, near, reg = self.classify_elements(x, node_idx)

            # --- singular elements ---
            if len(sing) > 0:
                xi_eta, w = duffy_rule(n_leg=4)
                vals = self._call_integrator(operator, x, n_x, sing, xi_eta, w)
                for el, row in zip(self.mesh.mesh_elements[sing], vals):
                    for local, node in enumerate(el):
                        A[node_idx, node] += row[local]

            # --- near-singular elements ---
            for elem in near:
                xi_star, eta_star = self.barycentric_projection(
                    x, self.mesh.v0[elem], 
                    self.mesh.e1[elem], self.mesh.e2[elem]
                )
                xi_eta, w = telles_rule(u_star=xi_star, 
                                        v_star=eta_star, 
                                        n_leg=4)
                vals = self._call_integrator(operator, x, n_x,
                                             np.array([elem]), xi_eta, w)
                nodes = self.mesh.mesh_elements[elem]
                for local, node in enumerate(nodes):
                    A[node_idx, node] += vals[0, local]

            # --- regular elements ---
            if len(reg) > 0:
                xi_eta, w = standard_triangle_quad(self.quad_order)
                vals = self._call_integrator(operator, x, n_x, reg, xi_eta, w)
                for el, row in zip(self.mesh.mesh_elements[reg], vals):
                    for local, node in enumerate(el):
                        A[node_idx, node] += row[local]

        return A

    def classify_elements(self, 
                          x: np.ndarray, 
                          node_idx: int) -> tuple[np.ndarray, 
                                                  np.ndarray, 
                                                  np.ndarray]:
        """
        Classify elements relative to a collocation node.

        Args:
            x (np.ndarray): Coordinates of the collocation node, shape (3,).
            node_idx (int): Index of the node in the mesh.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays of element indices
            corresponding to singular, near-singular, and regular categories.
        """
        singular = self.mesh.node_in_el[node_idx]
        d = np.linalg.norm(self.mesh.centroids - x, axis=1)
        near = np.where(d < self.near_threshold * self.mesh.char_length)[0]
        near = np.setdiff1d(near, singular)
        regular = np.setdiff1d(np.arange(self.Ne), np.union1d(singular, near))
        return singular, near, regular

    def _call_integrator(self,
                         operator: str,
                         x: np.ndarray,
                         n_x: np.ndarray,
                         elem_idx: np.ndarray,
                         xi_eta: np.ndarray,
                         w: np.ndarray) -> np.ndarray:
        """
        Dispatch to the correct integrator method.

        Args:
            operator (str): Operator key (``S``, ``D``, ``Kp``, ``N`` or 
                "NReg").
            x (np.ndarray): Collocation point, shape (3,).
            n_x (np.ndarray): Outward normal at the collocation point, shape 
                (3,).
            elem_idx (np.ndarray): Indices of source elements.
            xi_eta (np.ndarray): Quadrature points, shape (Q, 2).
            w (np.ndarray): Quadrature weights, shape (Q,).

        Returns:
            np.ndarray: Local element contributions, shape (len(elem_idx), 3).
        """
        if operator == "S":
            return self.integrator.single_layer_batch(
                x,
                self.mesh.v0[elem_idx],
                self.mesh.e1[elem_idx],
                self.mesh.e2[elem_idx],
                xi_eta,
                w,
            )
        if operator == "D":
            return self.integrator.double_layer_batch(
                x,
                self.mesh.v0[elem_idx],
                self.mesh.e1[elem_idx],
                self.mesh.e2[elem_idx],
                self.mesh.n_hat[elem_idx],
                xi_eta,
                w,
            )
        if operator == "Kp":
            return self.integrator.adjoint_double_layer_batch(
                x,
                n_x,
                self.mesh.v0[elem_idx],
                self.mesh.e1[elem_idx],
                self.mesh.e2[elem_idx],
                xi_eta,
                w,
            )
        
        if operator == "N":
            return self.integrator.hypersingular_batch(
                x,
                n_x,
                self.mesh.v0[elem_idx],
                self.mesh.e1[elem_idx],
                self.mesh.e2[elem_idx],
                self.mesh.n_hat[elem_idx],
                xi_eta,
                w,
            )
        
        if operator == "NReg":
            return self.integrator.hypersingular_batch_reg(
                x,
                n_x,
                self.mesh.v0[elem_idx],
                self.mesh.e1[elem_idx],
                self.mesh.e2[elem_idx],
                self.mesh.n_hat[elem_idx],
                xi_eta,
                w,
            )
        
        raise ValueError(f"Unsupported operator: {operator}")

    @staticmethod
    def barycentric_projection(x: np.ndarray,
                               v0: np.ndarray,
                               e1: np.ndarray,
                               e2: np.ndarray) -> tuple[float, float]:
        """
        Compute barycentric coordinates of the projection of a point onto a 
        triangle.

        Args:
            x (np.ndarray): Observation point in 3D, shape (3,).
            v0 (np.ndarray): First vertex of the triangle, shape (3,).
            e1 (np.ndarray): Edge vector from v0 to v1, shape (3,).
            e2 (np.ndarray): Edge vector from v0 to v2, shape (3,).

        Returns:
            tuple[float, float]: The barycentric coordinates (xi, eta) of the
            projected point in the reference triangle.
        """
        b = x - v0
        M = np.array([[np.dot(e1, e1), np.dot(e1, e2)],
                      [np.dot(e2, e1), np.dot(e2, e2)]])
        rhs = np.array([np.dot(b, e1), np.dot(b, e2)])
        try:
            xi_eta = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            return 1.0 / 3.0, 1.0 / 3.0

        xi, eta = xi_eta
        if xi < 0.0:
            xi = 0.0
        if eta < 0.0:
            eta = 0.0
        if xi + eta > 1.0:
            s = xi + eta
            xi /= s
            eta /= s
        return float(xi), float(eta)