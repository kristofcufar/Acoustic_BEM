import numpy as np
from tqdm.notebook import tqdm

from acoustic_BEM.quadrature import (standard_triangle_quad, 
                               duffy_rule, 
                               telles_rule,
                               barycentric_projection,
                               shape_functions_P1,
                               map_to_physical_triangle_batch)

from acoustic_BEM.mesh import Mesh
from acoustic_BEM.integrators import ElementIntegratorCollocation

class _CollocationCache:
    """
    Geometry + quadrature cache for Collocation. Useful for k-sweeping.
    Caches quadrature points mapped to physical elements for:
      - regular elements (all elements at once)
      - Telles rule (per node-element pair)
      - Duffy rule (per node-element pair)
    """
    def __init__(self, 
                 mesh: Mesh, 
                 quad_order: int):
        self.m = mesh
        self.quad_order = quad_order

        self.xi_eta_reg, self.w_reg = standard_triangle_quad(self.quad_order)

        self.N_reg = shape_functions_P1(self.xi_eta_reg) 

        self._y_reg = None
        self._w_reg_phys = None

        self._telles = {}
        self._duffy  = {}

    def ensure_regular_mapped(self):
        if self._y_reg is not None:
            return
        xi = self.xi_eta_reg; w = self.w_reg
        yq, a2 = map_to_physical_triangle_batch(
            xi, self.m.v0, self.m.e1, self.m.e2
        )
        self._y_reg = yq
        self._w_reg_phys = (w[None, :] * a2[:, None])

    def get_regular(self, 
                    elem_idx: np.ndarray):
        self.ensure_regular_mapped()
        return (self._y_reg[elem_idx], 
                self._w_reg_phys[elem_idx], 
                self.N_reg)

    def get_telles(self, 
                   node_idx: int, 
                   elem: int):
        key = (node_idx, elem)
        if key not in self._telles:
            x = self.m.mesh_nodes[node_idx]
            xi_star, eta_star = barycentric_projection(x, 
                                                       self.m.v0[elem], 
                                                       self.m.e1[elem], 
                                                       self.m.e2[elem])
            xi_eta, w = telles_rule(u_star=xi_star, v_star=eta_star, n_leg=10)
            yq, a2 = map_to_physical_triangle_batch(xi_eta,
                                                    self.m.v0[elem:elem+1],
                                                    self.m.e1[elem:elem+1],
                                                    self.m.e2[elem:elem+1])
            Nq = shape_functions_P1(xi_eta)
            self._telles[key] = (yq[0], w * a2[0], Nq)
        return self._telles[key]

    def get_duffy(self, node_idx: int, elem: int):
        key = (node_idx, elem)
        if key not in self._duffy:

            conn = self.m.mesh_elements[elem]
            try:
                loc = int(np.where(conn == node_idx)[0][0])
            except Exception:
                loc = 0  # fallback
            xi_eta, w = duffy_rule(n_leg=10, sing_vert_int=loc)
            yq, a2 = map_to_physical_triangle_batch(xi_eta,
                                                    self.m.v0[elem:elem+1],
                                                    self.m.e1[elem:elem+1],
                                                    self.m.e2[elem:elem+1])
            Nq = shape_functions_P1(xi_eta)
            self._duffy[key] = (yq[0], w * a2[0], Nq)
        return self._duffy[key]

class CollocationAssembler:
    """
    Collocation BEM assembler for the four boundary operators.

    This class computes collocation matrices for:

    - Single-layer potential (``S``)
    - Double-layer potential (``D``)
    - Adjoint double-layer (``Kp``)
    - Hypersingular operator (``N``)
    - Regularized hypersingular operator (``NReg``)

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

        self.cache = _CollocationCache(mesh, quad_order)

    def assemble(self, operator: str, verbose: bool = True) -> np.ndarray:
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

        for node_idx in tqdm(range(self.Nn), 
                             desc=f"Assembling {operator} matrix",
                             disable = not verbose):
            x = self.mesh.mesh_nodes[node_idx]
            n_x = self.mesh.node_n_hat[node_idx]

            sing, near, reg = self.classify_elements(x, node_idx)

            if len(sing) > 0:
                for elem in sing:
                    if operator == "NReg":
                        conn = self.mesh.mesh_elements[elem]
                        try:
                            loc = int(np.where(conn == node_idx)[0][0])
                        except Exception:
                            loc = 0  # fallback
                        xi_eta, w = duffy_rule(n_leg=10, sing_vert_int=loc)
                        row = self.call_integrator(operator, x, n_x,
                                                   np.array([elem]),
                                                xi_eta, w, Nq=None, n_y=None)
                    else:
                        yq, w_phys, Nq = self.cache.get_duffy(node_idx, elem)
                        n_y = self.mesh.n_hat[elem:elem+1] if operator in \
                            {"D", "N"} else None
                        row = self.call_integrator(operator, x, n_x, 
                                                np.array([elem]), 
                                                yq[None, :, :], 
                                                w_phys[None, :], 
                                                Nq,
                                                n_y)
                    nodes = self.mesh.mesh_elements[elem]
                    for local, node in enumerate(nodes):
                        A[node_idx, node] += row[0, local]

            for elem in near:
                if operator == "NReg":
                    xi_star, eta_star = barycentric_projection(
                        x, self.mesh.v0[elem], 
                        self.mesh.e1[elem], self.mesh.e2[elem]
                    )
                    xi_eta, w = telles_rule(u_star=xi_star, 
                                            v_star=eta_star, 
                                            n_leg=10)
                    row = self.call_integrator(operator, x, n_x,
                                            np.array([elem]),
                                            xi_eta, w, Nq=None, n_y=None)
                else:
                    yq, w_phys, Nq = self.cache.get_telles(node_idx, elem)
                    n_y = self.mesh.n_hat[elem:elem+1] if operator in {"D", "N"}\
                        else None
                    row = self.call_integrator(operator, x, n_x, 
                                            np.array([elem]), 
                                            yq[None, :, :], 
                                            w_phys[None, :], 
                                            Nq,
                                            n_y)
                nodes = self.mesh.mesh_elements[elem]
                for local, node in enumerate(nodes):
                    A[node_idx, node] += row[0, local]

            if len(reg) > 0:
                if operator == "NReg":
                    xi_eta, w = standard_triangle_quad(self.quad_order)
                    vals = self.call_integrator(operator, x, n_x, 
                                                reg, 
                                                xi_eta, 
                                                w, 
                                                Nq=None, 
                                                n_y=None)
                    for el, row in zip(self.mesh.mesh_elements[reg], vals):
                        for local, node in enumerate(el):
                            A[node_idx, node] += row[local]

                else:
                    y_phys, w_phys, N = self.cache.get_regular(reg)
                    n_y = self.mesh.n_hat[reg] if operator in {"D", "N"} else None
                    vals = self.call_integrator(operator, x, n_x, reg,
                                                y_phys, w_phys, N,
                                                n_y)
                    
                    for el, row in zip(self.mesh.mesh_elements[reg], vals):
                        for local, node in enumerate(el):
                            A[node_idx, node] += row[local]
        return A
    
    def call_integrator(self,
                        operator: str,
                        x: np.ndarray,
                        n_x: np.ndarray | None,
                        elem_idx: np.ndarray | None,
                        xi_eta: np.ndarray,
                        w: np.ndarray,
                        Nq: np.ndarray | None,
                        n_y: np.ndarray | None = None) -> np.ndarray:
        """
        Dispatch to the correct integrator method.
        Args:
            operator (str): Operator key (``S``, ``D``, ``Kp``, ``N`` or 
                "NReg").
            x (np.ndarray): Collocation point, shape (3,).
            n_x (np.ndarray): Outward normal at the collocation point, shape 
                (3,).
            elem_idx (np.ndarray): Indices of source elements.
            xi_eta (np.ndarray): Quadrature points, shape (K, Q, 3).
            w (np.ndarray): Quadrature weights, shape (K, Q).
            Nq (np.ndarray): Shape functions at quadrature points, shape (Q, 3).
            n_y (np.ndarray | None): Outward normals at source elements, 
                shape (len(elem_idx), 3). Required for ``D``, ``N`` and 
                ``NReg``.

        Returns:
            np.ndarray: Local element contributions, shape (len(elem_idx), 3).
        """
        
        if operator == "S":
            return self.integrator.single_layer(x = x,
                                                y_phys = xi_eta,
                                                w_phys = w,
                                                N = Nq)
        if operator == "D":
            if n_y is None:
                raise ValueError("n_y must be provided for double-layer \
                                 operator")
            return self.integrator.double_layer(x = x,
                                               y_phys = xi_eta,
                                               w_phys = w,
                                               N = Nq,
                                               n_y = n_y)
        if operator == "Kp":
            return self.integrator.adjoint_double_layer(x = x,
                                                       x_normal = n_x,
                                                       y_phys = xi_eta,
                                                       w_phys = w,
                                                       N = Nq)     
        if operator == "N":
            if n_y is None:
                raise ValueError("n_y must be provided for hypersingular \
                                 operator")
            return self.integrator.hypersingular_layer(x = x,
                                                      x_normal = n_x,
                                                      y_phys = xi_eta,
                                                      w_phys = w,
                                                      N = Nq,
                                                      n_y = n_y)
        if operator == "NReg":
            return self.integrator.hypersingular_layer_reg(x = x,
                                        x_normal = n_x,
                                        y_v0 = self.mesh.v0[elem_idx],
                                        y_e1 = self.mesh.e1[elem_idx],
                                        y_e2 = self.mesh.e2[elem_idx],
                                        y_normals= self.mesh.n_hat[elem_idx],
                                        xi_eta = xi_eta,
                                        w = w,)
        raise ValueError(f"Unsupported operator: {operator}")

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