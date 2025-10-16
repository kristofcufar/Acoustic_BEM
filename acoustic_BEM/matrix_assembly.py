import numpy as np

from acoustic_BEM.quadrature import (standard_triangle_quad, 
                               duffy_rule, 
                               telles_rule,
                               subdivide_triangle_quad,
                               barycentric_projection,
                               shape_functions_P1,
                               map_to_physical_triangle_batch)

from acoustic_BEM.mesh import Mesh
from acoustic_BEM.integrators import (ElementIntegratorCollocation, 
                               ElementIntegratorGalerkin)


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
            xi_eta, w = telles_rule(u_star=xi_star, v_star=eta_star, n_leg=4)
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
            xi_eta, w = duffy_rule(n_leg=4, sing_vert_int=loc)
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

            if len(sing) > 0:
                for elem in sing:
                    if operator == "NReg":
                        xi_eta, w = duffy_rule(n_leg=4, sing_vert_int=0)
                        row = self.call_integrator(operator, x, n_x, 
                                                   np.array([elem]), 
                                                   xi_eta, 
                                                   w, 
                                                   Nq=None,
                                                   n_y=None)
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
                                            n_leg=4)
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
                                        w = w)
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
    
    
class GalerkinAssembler:
    """
    Galerkin BEM assembler for P1–P1 operators.

    Operators:
      - "S"   : single-layer
      - "D"   : double-layer
      - "Kp"  : adjoint double-layer
      - "N"   : hypersingular (regularized)
      - "M"   : jump terms

    The interface mirrors CollocationAssembler as closely as possible.
    """

    def __init__(self,
                 mesh: Mesh,
                 integrator: ElementIntegratorGalerkin,
                 quad_order: int = 3,
                 near_threshold: float = 2.0,
                 near_subdiv_levels: int = 1,
                 self_subdiv_levels: int = 2):
        """
        Args:
            mesh: geometry and discretization.
            integrator: ElementIntegratorGalerkin instance (with wavenumber k).
            quad_order: order for regular triangle quadrature.
            near_threshold: distance factor for near-pair detection, relative
                to element characteristic lengths.
            near_subdiv_levels: extra refinement levels for near pairs.
            self_subdiv_levels: refinement for ex==ey pairs.
        """
        self.mesh = mesh
        self.integrator = integrator
        self.quad_order = quad_order
        self.near_threshold = near_threshold
        self.near_subdiv_levels = near_subdiv_levels
        self.self_subdiv_levels = self_subdiv_levels

        self.Nn = mesh.num_nodes
        self.Ne = mesh.num_elements

        # Base quadrature for regular pairs
        self.xi_eta_reg, self.w_reg = standard_triangle_quad(self.quad_order)

    def assemble(self, operator: str) -> np.ndarray:
        """
        Assemble Galerkin matrix for operator in {"S","D","Kp","N", "M"}.

        Returns:
            (num_nodes, num_nodes) dense complex matrix.
        """
        if operator not in {"S", "D", "Kp", "N", "M"}:
            raise ValueError(f"Unknown operator {operator}")

        A = np.zeros((self.Nn, self.Nn), dtype=np.complex128)

        adjacency = self._build_element_adjacency()

        if operator == "M":
            xi_x, w_x = self.xi_eta_reg, self.w_reg
            for ex in range(self.Ne):
                Bij = self.integrator.jump_block_P1P1(self.mesh, ex, xi_x, w_x)
                self._scatter_add(A, ex, ex, Bij)
            return A
        
        else:
            for ex in range(self.Ne):
                self_set, near_set, reg_set = self._classify_pairs(ex, 
                                                                   adjacency)
                if len(self_set) > 0:
                    xi_x, w_x = standard_triangle_quad(7)
                    xi_y, w_y = standard_triangle_quad(self.quad_order)

                    Bij = self._call_integrator(operator, 
                                        ex, np.array(self_set), 
                                        xi_x, w_x, 
                                        xi_y, w_y)
                    
                    for i, ey in enumerate(self_set):
                        self._scatter_add(A, ex, ey, Bij[i])

                if len(near_set) > 0:
                    xi_x, w_x = subdivide_triangle_quad(
                        self.xi_eta_reg, self.w_reg, 
                        levels=self.near_subdiv_levels
                    )

                    xi_y, w_y = self.xi_eta_reg, self.w_reg

                    Bij = self._call_integrator(operator, 
                                                ex, np.array(near_set), 
                                                xi_x, w_x, 
                                                xi_y, w_y)
                    for i, ey in enumerate(near_set):
                        self._scatter_add(A, ex, ey, Bij[i])

                if len(reg_set) > 0:
                    xi_x, w_x = self.xi_eta_reg, self.w_reg
                    xi_y, w_y = self.xi_eta_reg, self.w_reg
                    Bij = self._call_integrator(operator,
                                                ex, np.array(reg_set),
                                                xi_x, w_x, xi_y, w_y)
                    for i, ey in enumerate(reg_set):
                        self._scatter_add(A, ex, ey, Bij[i])

            return A

    def _call_integrator(self,
                    operator: str,
                    ex: int, ey: int | np.ndarray,
                    xi_x: np.ndarray, w_x: np.ndarray,
                    xi_y: np.ndarray, w_y: np.ndarray) -> np.ndarray:
        """
        Dispatch to the correct Galerkin element block.
        
        Args:
            operator (str): One of {"S","D","Kp","N"}.
            ex (int): Index of "test" element.
            ey (int): Index of "trial" element.
            xi_x (np.ndarray): Quadrature points on "test" element, shape 
                (Qx, 2).
            w_x (np.ndarray): Quadrature weights on "test" element, shape 
                (Qx,).
            xi_y (np.ndarray): Quadrature points on "trial" element, shape 
                (Qy, 2).
            w_y (np.ndarray): Quadrature weights on "trial" element, shape 
                (Qy,).

        Returns:
            Bij (np.ndarray): Local 3x3 element block as a numpy array.
        """
        if operator == "S":
            return self.integrator.single_layer_block_P1P1_batch(
                self.mesh, ex, ey, xi_x, w_x, xi_y, w_y
            )
        if operator == "D":
            return self.integrator.double_layer_block_P1P1_batch(
                self.mesh, ex, ey, xi_x, w_x, xi_y, w_y
            )
        if operator == "Kp":
            return self.integrator.adjoint_double_layer_block_P1P1_batch(
                self.mesh, ex, ey, xi_x, w_x, xi_y, w_y
            )
        if operator == "N":
            return self.integrator.hypersingular_block_P1P1_reg_batch(
                self.mesh, ex, ey, xi_x, w_x, xi_y, w_y
            )
        raise ValueError(f"Unsupported operator: {operator}")

    def _scatter_add(self, 
                     A: np.ndarray, 
                     ex: int, 
                     ey: int, 
                     Bij: np.ndarray) -> None:
        """
        Add 3×3 local block to global matrix at (element ex nodes, 
        element ey nodes).

        Args:
            A (np.ndarray): Global matrix to update.
            ex (int): Index of "test" element.
            ey (int): Index of "trial" element.
            Bij (np.ndarray): Local 3x3 element block to add.

        Returns:
            None: The matrix A is updated in place.
        """
        rows = self.mesh.mesh_elements[ex]
        cols = self.mesh.mesh_elements[ey]

        A[np.ix_(rows, cols)] += Bij

    def _classify_pairs(self, 
                        ex: int, adjacency: list[set[int]]
                        ) -> tuple[list[int], list[int], list[int]]:
        """
        Classify ey into self / near / regular for a fixed ex.
        
        Args:
            ex (int): Index of the "test" element.
            adjacency (list[set[int]]): Element adjacency list.

        Returns:
            tuple[list[int], list[int], list[int]]: Three lists of element
            indices corresponding to self, near, and regular categories.
        """
        self_set = [ex]

        near = set(adjacency[ex])
        if ex in near:
            near.remove(ex)

        cx = self.mesh.centroids[ex]
        dist = np.linalg.norm(self.mesh.centroids - cx, axis=1)
        th = self.near_threshold * np.maximum(self.mesh.char_length, 
                                              self.mesh.char_length[ex])
        by_dist = set(np.where(dist < th)[0])
        by_dist.discard(ex)

        near |= by_dist

        all_e = set(range(self.Ne))
        regular = sorted(all_e - near - set(self_set))
        return self_set, sorted(near), regular

    def _build_element_adjacency(self) -> list[set[int]]:
        """
        Elements touching by at least one node.
        Returns:
            list[set[int]]: For each element, a set of adjacent element 
                indices.
        """
        node_to_elems = self.mesh.node_in_el  # list of arrays
        adj = [set() for _ in range(self.Ne)]
        for e in range(self.Ne):
            nodes = self.mesh.mesh_elements[e]
            touchers: set[int] = set()
            for n in nodes:
                touchers.update(node_to_elems[n].tolist())
            adj[e] = touchers
        return adj