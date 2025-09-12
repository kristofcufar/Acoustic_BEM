import numpy as np
from typing import Tuple, Literal, Optional
from kernels import r_vec, G, dG_dr, dG_dn_y, dG_dn_x, d2G_dn_x_dn_y
from quadrature import (standard_triangle_quad, subdivide_triangle_quad,
                       telles_rule, duffy_rule, map_to_physical_triangle_batch,
                       shape_functions_P1)


class ElementIntegratorCollocation:
    """
    Element-wise integrator for acoustic boundary element method.
    
    Provides efficient vectorized integration methods for single layer, double 
    layer, adjoint double layer, and hypersingular boundary integrals. Supports
    different quadrature rules and both regular and singular integration 
    strategies.
    """
    
    def __init__(self, 
                 k: float,
                 regular_quad_order: int = 3,
                 near_singular_levels: int = 2,
                 singular_n_leg: int = 8,
                 near_singular_threshold: float = 2.0):
        """
        Initialize the element integrator.

        Args:
            k: Wavenumber for Helmholtz kernel.
            regular_quad_order: Quadrature order for regular integrals (1, 3, 
                or 7).
            near_singular_levels: Number of subdivision levels for near-
                singular integrals.
            singular_n_leg: Number of Gauss-Legendre points for singular 
                integrals.
            near_singular_threshold: Distance threshold for near-singular 
                detection (in terms of element characteristic size).
        """
        self.k = k
        self.regular_quad_order = regular_quad_order
        self.near_singular_levels = near_singular_levels
        self.singular_n_leg = singular_n_leg
        self.near_singular_threshold = near_singular_threshold
        
        # Precompute standard quadrature rules
        self.xi_eta_regular, self.w_regular = \
            standard_triangle_quad(regular_quad_order)
        self.xi_eta_near, self.w_near = subdivide_triangle_quad(
            self.xi_eta_regular, self.w_regular, levels=near_singular_levels)

    def get_quadrature_rule(self, 
                           rule_type: Literal["standard", 
                                              "duffy", 
                                              "telles", 
                                              "subdivide"],
                           rule_kwargs: Optional[dict] = None,
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get quadrature points and weights for specified rule.

        Args:
            rule_type: Type of quadrature rule to use.
            rule_kwargs: Additional keyword arguments for the quadrature rule.

        Returns:
            Tuple of (xi_eta, weights) arrays for the specified rule.

        Raises:
            ValueError: If rule_type is not recognized.
        """
        if rule_kwargs is None:
            rule_kwargs = {}
            
        if rule_type == "standard":
            return self.xi_eta_regular, self.w_regular
        elif rule_type == "duffy":
            return duffy_rule(n_leg=rule_kwargs.get('n_leg', 
                                                    self.singular_n_leg))
        elif rule_type == "telles":
            return telles_rule(
                u_star=rule_kwargs.get('u_star', 0.0),
                v_star=rule_kwargs.get('v_star', None),
                sing_vert_int=rule_kwargs.get('sing_vert_int', 0),
                n_leg=rule_kwargs.get('n_leg', self.singular_n_leg))
        elif rule_type == "subdivide":
            return self.xi_eta_near, self.w_near
        else:
            raise ValueError(f"Unknown quadrature rule: {rule_type}")

    def classify_integration_type(self,
                                 x_centroid: np.ndarray,
                                 y_centroids: np.ndarray, 
                                 y_sizes: np.ndarray,
                                 x_elem_idx: Optional[int] = None,
                                 y_elem_indices: Optional[np.ndarray] = None,
                                 ) -> np.ndarray:
        """
        Classify integration types for a batch of element interactions.

        Args:
            x_centroid: Field point centroid, shape (3,).
            y_centroids: Source element centroids, shape (K, 3).
            y_sizes: Characteristic sizes of source elements, shape (K,).
            x_elem_idx: Field element index (for singular detection).
            y_elem_indices: Source element indices, shape (K,).

        Returns:
            Array of strings indicating integration type for each interaction, 
                shape (K,).
            Values are 'singular', 'near_singular', or 'regular'.
        """
        K = len(y_centroids)
        integration_types = np.full(K, 'regular', dtype='<U12')
        
        # Check for singular interactions
        if x_elem_idx is not None and y_elem_indices is not None:
            singular_mask = (y_elem_indices == x_elem_idx)
            integration_types[singular_mask] = 'singular'
            
            # Only check distance for non-singular elements
            regular_mask = ~singular_mask
        else:
            regular_mask = np.ones(K, dtype=bool)
        
        if np.any(regular_mask):
            # Compute distances for non-singular elements
            distances = np.linalg.norm(
                y_centroids[regular_mask] - x_centroid[None, :], axis=1)
            threshold_distances = \
                self.near_singular_threshold * y_sizes[regular_mask]
            
            # Mark near-singular elements
            near_singular_local = distances < threshold_distances
            integration_types[regular_mask] = np.where(
                near_singular_local, 'near_singular', 'regular')
        
        return integration_types

    def single_layer_batch(self, 
                          x: np.ndarray,
                          y_v0: np.ndarray,
                          y_e1: np.ndarray, 
                          y_e2: np.ndarray,
                          xi_eta: np.ndarray,
                          w: np.ndarray) -> np.ndarray:
        """
        Compute single layer integrals for a batch of triangles.
        
        Computes: ∫_T G(x,y) N_j(y) dS_y for j=0,1,2

        Args:
            x: Observation point, shape (3,).
            y_v0: First vertices of source triangles, shape (K, 3).
            y_e1: First edge vectors of source triangles, shape (K, 3).
            y_e2: Second edge vectors of source triangles, shape (K, 3).
            xi_eta: Quadrature points in reference triangle, shape (Q, 2).
            w: Quadrature weights, shape (Q,).

        Returns:
            Local vectors for each triangle, shape (K, 3).
        """
        K = len(y_v0)
        
        # Map quadrature points to physical triangles
        y_phys, a2 = map_to_physical_triangle_batch(xi_eta, y_v0, y_e1, y_e2)
        
        # Compute shape functions at quadrature points
        N = shape_functions_P1(xi_eta)  # (Q, 3)
        
        # Compute Green's function values
        _, r_norm, _ = r_vec(x[None, None, :], y_phys)  # (K, Q)
        G_vals = G(r_norm, self.k)  # (K, Q)
        
        # Physical quadrature weights
        w_phys = w[None, :] * a2[:, None] / 2.0  # (K, Q)
        
        # Integrate: sum over quadrature points
        integrand = w_phys[:, :, None] * G_vals[:, :, None] * N[None, :, :]
        
        return np.sum(integrand, axis=1)  # (K, 3)

    def double_layer_batch(self, 
                          x: np.ndarray,
                          y_v0: np.ndarray,
                          y_e1: np.ndarray,
                          y_e2: np.ndarray, 
                          y_normals: np.ndarray,
                          xi_eta: np.ndarray,
                          w: np.ndarray) -> np.ndarray:
        """
        Compute double layer integrals for a batch of triangles.
        
        Computes: ∫_T ∂G(x,y)/∂n_y N_j(y) dS_y for j=0,1,2

        Args:
            x: Observation point, shape (3,).
            y_v0: First vertices of source triangles, shape (K, 3).
            y_e1: First edge vectors of source triangles, shape (K, 3).
            y_e2: Second edge vectors of source triangles, shape (K, 3).
            y_normals: Normal vectors of source triangles, shape (K, 3).
            xi_eta: Quadrature points in reference triangle, shape (Q, 2).
            w: Quadrature weights, shape (Q,).

        Returns:
            Local vectors for each triangle, shape (K, 3).
        """
        K = len(y_v0)
        
        # Map quadrature points to physical triangles
        y_phys, a2 = map_to_physical_triangle_batch(xi_eta, y_v0, y_e1, y_e2)
        
        # Compute shape functions at quadrature points
        N = shape_functions_P1(xi_eta)  # (Q, 3)
        
        # Compute Green's function and its normal derivative
        _, r_norm, r_hat = r_vec(x[None, None, :], y_phys)  # (K, Q, 3)
        G_vals = G(r_norm, self.k)  # (K, Q)
        dG_dr_vals = dG_dr(r_norm, G_vals, self.k)  # (K, Q)
        
        # Broadcast normals to quadrature points
        n_y_broadcast = y_normals[:, None, :]  # (K, 1, 3)
        dG_dn_y_vals = dG_dn_y(r_hat, dG_dr_vals, n_y_broadcast)  # (K, Q)
        
        # Physical quadrature weights
        w_phys = w[None, :] * a2[:, None] / 2.0  # (K, Q)
        
        # Integrate: sum over quadrature points
        integrand = w_phys[:, :, None] * dG_dn_y_vals[:, :, None] * N[None, :, :]
        
        return np.sum(integrand, axis=1)  # (K, 3)

    def adjoint_double_layer_batch(self, 
                                  x: np.ndarray,
                                  x_normal: np.ndarray,
                                  y_v0: np.ndarray,
                                  y_e1: np.ndarray,
                                  y_e2: np.ndarray,
                                  xi_eta: np.ndarray,
                                  w: np.ndarray) -> np.ndarray:
        """
        Compute adjoint double layer integrals for a batch of triangles.
        
        Computes: ∫_T ∂G(x,y)/∂n_x N_j(y) dS_y for j=0,1,2

        Args:
            x: Observation point, shape (3,).
            x_normal: Normal vector at observation point, shape (3,).
            y_v0: First vertices of source triangles, shape (K, 3).
            y_e1: First edge vectors of source triangles, shape (K, 3).
            y_e2: Second edge vectors of source triangles, shape (K, 3).
            xi_eta: Quadrature points in reference triangle, shape (Q, 2).
            w: Quadrature weights, shape (Q,).

        Returns:
            Local vectors for each triangle, shape (K, 3).
        """
        K = len(y_v0)
        
        # Map quadrature points to physical triangles
        y_phys, a2 = map_to_physical_triangle_batch(xi_eta, y_v0, y_e1, y_e2)
        
        # Compute shape functions at quadrature points
        N = shape_functions_P1(xi_eta)  # (Q, 3)
        
        # Compute Green's function and its normal derivative w.r.t. x
        _, r_norm, r_hat = r_vec(x[None, None, :], y_phys)  # (K, Q, 3)
        G_vals = G(r_norm, self.k)  # (K, Q)
        dG_dr_vals = dG_dr(r_norm, G_vals, self.k)  # (K, Q)
        
        # Broadcast x normal to all points
        n_x_broadcast = np.broadcast_to(x_normal[None, None, :], y_phys.shape)
        dG_dn_x_vals = dG_dn_x(r_hat, dG_dr_vals, n_x_broadcast)  # (K, Q)
        
        # Physical quadrature weights
        w_phys = w[None, :] * a2[:, None] / 2.0  # (K, Q)
        
        # Integrate: sum over quadrature points
        integrand = \
            w_phys[:, :, None] * dG_dn_x_vals[:, :, None] * N[None, :, :]
        
        return np.sum(integrand, axis=1)  # (K, 3)

    def hypersingular_batch_maue(self,
                                x: np.ndarray,
                                x_normal: np.ndarray,
                                y_v0: np.ndarray,
                                y_e1: np.ndarray,
                                y_e2: np.ndarray,
                                y_normals: np.ndarray,
                                xi_eta: np.ndarray,
                                w: np.ndarray) -> np.ndarray:
        """
        Compute hypersingular integrals using Maue's identity.
        
        Uses the identity:
        ∫_T ∂²G/(∂n_x∂n_y) N_j dS = 
            k² ∫_T (n_x·n_y) G N_j dS + ∫_T ∇_y G · ∇_Γ N_j dS

        Args:
            x: Observation point, shape (3,).
            x_normal: Normal vector at observation point, shape (3,).
            y_v0: First vertices of source triangles, shape (K, 3).
            y_e1: First edge vectors of source triangles, shape (K, 3).
            y_e2: Second edge vectors of source triangles, shape (K, 3).
            y_normals: Normal vectors of source triangles, shape (K, 3).
            xi_eta: Quadrature points in reference triangle, shape (Q, 2).
            w: Quadrature weights, shape (Q,).

        Returns:
            Local vectors for each triangle, shape (K, 3).

        Note:
            This implementation assumes the mesh has precomputed inverse metric 
                tensors.
            For a more general implementation, compute these on the fly.
        """
        K = len(y_v0)
        
        # Map quadrature points to physical triangles
        y_phys, a2 = map_to_physical_triangle_batch(xi_eta, y_v0, y_e1, y_e2)
        w_phys = w[None, :] * a2[:, None] / 2.0  # (K, Q)
        
        # Compute dot product of normals
        nx_dot_ny = np.einsum("i,ki->k", x_normal, y_normals)  # (K,)
        
        # --- Part 1: k² (n_x·n_y) ∫ G N_j dS ---
        _, r_norm, r_hat = r_vec(x[None, None, :], y_phys)  # (K, Q)
        G_vals = G(r_norm, self.k)  # (K, Q)
        N_vals = shape_functions_P1(xi_eta)  # (Q, 3)
        
        part1 = np.einsum("kq,k,qj,kq->kj", 
                          G_vals, nx_dot_ny, N_vals, w_phys) * (self.k**2)
        
        # --- Part 2: ∫ (∇_y G) · (∇_Γ N_j) dS ---
        # Compute ∇_y G = -(dG/dr) * r_hat
        dG_dr_vals = dG_dr(r_norm, G_vals, self.k)  # (K, Q)
        grad_y_G = -dG_dr_vals[:, :, None] * r_hat  # (K, Q, 3)
        
        # Compute surface gradients of shape functions
        # Reference gradients: ∇_ξη N = [[-1,-1], [1,0], [0,1]]
        dN_ref = np.array([[-1.0, -1.0],
                          [ 1.0,  0.0],
                          [ 0.0,  1.0]], dtype=np.float64)  # (3, 2)
        
        # Compute inverse metric tensor G^{-1} for each triangle
        # G_ij = e_i · e_j where e_1, e_2 are edge vectors
        G11 = np.einsum("ki,ki->k", y_e1, y_e1)  # (K,)
        G12 = np.einsum("ki,ki->k", y_e1, y_e2)  # (K,) 
        G22 = np.einsum("ki,ki->k", y_e2, y_e2)  # (K,)
        
        det_G = G11 * G22 - G12 * G12  # (K,)
        det_G = np.where(np.abs(det_G) < 1e-14, 1e-14, det_G)
        
        Ginv = np.zeros((K, 2, 2))
        Ginv[:, 0, 0] = G22 / det_G
        Ginv[:, 0, 1] = -G12 / det_G
        Ginv[:, 1, 0] = -G12 / det_G
        Ginv[:, 1, 1] = G11 / det_G
        
        # Compute contravariant basis vectors g^α
        g1 = Ginv[:, 0, 0, None] * y_e1 + Ginv[:, 0, 1, None] * y_e2  # (K, 3)
        g2 = Ginv[:, 1, 0, None] * y_e1 + Ginv[:, 1, 1, None] * y_e2  # (K, 3)
        g = np.stack([g1, g2], axis=1)  # (K, 2, 3)
        
        # Surface gradients: ∇_Γ N_j = Σ_α g^α (∂N_j/∂ξ_α)
        grad_N = np.einsum("kao,ja->kjo", g, dN_ref)
        
        # Compute dot products and integrate
        dot_products = np.einsum("kqo,kjo->kjq", grad_y_G, grad_N)  # (K, 3, Q)
        part2 = np.einsum("kjq,kq->kj", dot_products, w_phys)  # (K, 3)
        
        return part1 + part2

    def hypersingular_batch_direct(self,
                                  x: np.ndarray,
                                  x_normal: np.ndarray,
                                  y_v0: np.ndarray,
                                  y_e1: np.ndarray,
                                  y_e2: np.ndarray,
                                  y_normals: np.ndarray,
                                  xi_eta: np.ndarray,
                                  w: np.ndarray) -> np.ndarray:
        """
        Compute hypersingular integrals directly.
        
        Computes: ∫_T ∂²G/(∂n_x∂n_y) N_j(y) dS_y for j=0,1,2

        Args:
            x: Observation point, shape (3,).
            x_normal: Normal vector at observation point, shape (3,).
            y_v0: First vertices of source triangles, shape (K, 3).
            y_e1: First edge vectors of source triangles, shape (K, 3).
            y_e2: Second edge vectors of source triangles, shape (K, 3).
            y_normals: Normal vectors of source triangles, shape (K, 3).
            xi_eta: Quadrature points in reference triangle, shape (Q, 2).
            w: Quadrature weights, shape (Q,).

        Returns:
            Local vectors for each triangle, shape (K, 3).

        Note:
            This method directly computes the second normal derivative.
            Use hypersingular_batch_maue for better numerical stability.
        """
        K = len(y_v0)
        
        # Map quadrature points to physical triangles
        y_phys, a2 = map_to_physical_triangle_batch(xi_eta, y_v0, y_e1, y_e2)
        
        # Compute shape functions at quadrature points
        N = shape_functions_P1(xi_eta)  # (Q, 3)
        
        # Compute second normal derivatives
        _, r_norm, r_hat = r_vec(x[None, None, :], y_phys)  # (K, Q, 3)
        G_vals = G(r_norm, self.k)  # (K, Q)
        dG_dr_vals = dG_dr(r_norm, G_vals, self.k)  # (K, Q)
        
        # Broadcast normals
        n_x_broadcast = np.broadcast_to(x_normal[None, None, :], y_phys.shape)
        n_y_broadcast = y_normals[:, None, :]  # (K, 1, 3)
        
        d2G_vals = d2G_dn_x_dn_y(r_hat, 
                                 dG_dr_vals, n_x_broadcast, n_y_broadcast)
        
        # Physical quadrature weights
        w_phys = w[None, :] * a2[:, None] / 2.0  # (K, Q)
        
        # Integrate: sum over quadrature points
        integrand = w_phys[:, :, None] * d2G_vals[:, :, None] * N[None, :, :]
        
        return np.sum(integrand, axis=1)  # (K, 3)
    

class ElementIntegratorGalerkin:
    """
    Element–element (P1–P1) Galerkin integrator for acoustic BEM.

    Provides 3x3 local blocks for the single layer (S), double layer (D),
    adjoint double layer (K'), and hypersingular (N) operators. Quadrature
    selection mirrors the collocation class API (standard / subdivide / duffy 
    / telles).
    """

    def __init__(self,
                 k: float,
                 regular_quad_order: int = 3,
                 near_singular_levels: int = 2,
                 singular_n_leg: int = 8,
                 near_singular_threshold: float = 2.0):
        """
        Initialize the Galerkin element–element integrator.

        Args:
            k: Wavenumber for Helmholtz kernel.
            regular_quad_order: Triangle quadrature order for regular pairs 
                (1,3,7).
            near_singular_levels: Subdivision depth for near-singular pairs.
            singular_n_leg: Gauss–Legendre points for Duffy/Telles.
            near_singular_threshold: Threshold d/min(hx,hy) for near 
                classification.
        """
        self.k = k
        self.regular_quad_order = regular_quad_order
        self.near_singular_levels = near_singular_levels
        self.singular_n_leg = singular_n_leg
        self.near_singular_threshold = near_singular_threshold

        self.xi_eta_regular, self.w_regular = \
            standard_triangle_quad(regular_quad_order)
        self.xi_eta_near, self.w_near = subdivide_triangle_quad(
            self.xi_eta_regular, self.w_regular, levels=near_singular_levels
        )

        self._dtype = np.complex128

    def get_quadrature_rule(self,
                            rule_type: Literal["standard", 
                                            "duffy", 
                                            "telles", 
                                            "subdivide"],
                            rule_kwargs: Optional[dict] = None,
                            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get quadrature nodes and weights for one triangle.

        Args:
            rule_type: 'standard' | 'duffy' | 'telles' | 'subdivide'.
            rule_kwargs: Extra parameters for Duffy/Telles.

        Returns:
            (xi_eta, w): arrays of shape (Q,2) and (Q,).
        """
        if rule_kwargs is None:
            rule_kwargs = {}

        if rule_type == "standard":
            return self.xi_eta_regular, self.w_regular
        elif rule_type == "duffy":
            return duffy_rule(n_leg=rule_kwargs.get("n_leg",
                                                     self.singular_n_leg))
        elif rule_type == "telles":
            return telles_rule(
                u_star=rule_kwargs.get("u_star", 0.0),
                v_star=rule_kwargs.get("v_star", None),
                sing_vert_int=rule_kwargs.get("sing_vert_int", 0),
                n_leg=rule_kwargs.get("n_leg", self.singular_n_leg),
            )
        elif rule_type == "subdivide":
            return self.xi_eta_near, self.w_near
        else:
            raise ValueError(f"Unknown quadrature rule: {rule_type}")

    def classify_integration_type(self,
                                  x_centroid: np.ndarray,
                                  y_centroids: np.ndarray,
                                  y_sizes: np.ndarray,
                                  x_elem_idx: Optional[int] = None,
                                  y_elem_indices: Optional[np.ndarray] = None,
                                  ) -> np.ndarray:
        """
        Classify (ex, ey) pairs as 'singular' (ex==ey), 'near_singular', or 
            'regular'.

        Args:
            x_centroid: Centroid of observation element ex, shape (3,).
            y_centroids: Centroids of candidate source elements ey, shape 
                (K,3).
            y_sizes: Characteristic sizes (e.g., sqrt(area)) of ey elements, 
                shape (K,).
            x_elem_idx: Observation element index ex.
            y_elem_indices: Indices of ey elements, shape (K,).

        Returns:
            dtype '<U12' array of length K with values in {'singular',
                'near_singular','regular'}.
        """
        K = len(y_centroids)
        types = np.full(K, "regular", dtype="<U12")

        if x_elem_idx is not None and y_elem_indices is not None:
            singular_mask = (y_elem_indices == x_elem_idx)
            types[singular_mask] = "singular"
            check_mask = ~singular_mask
        else:
            check_mask = np.ones(K, dtype=bool)

        if np.any(check_mask):
            d = np.linalg.norm(y_centroids[check_mask] - x_centroid[None, :], 
                               axis=1)
            thr = self.near_singular_threshold * y_sizes[check_mask]
            near = d < thr
            tmp = np.where(near, "near_singular", "regular")
            types[check_mask] = tmp

        return types

    def single_layer_block_P1P1(self,
                                mesh,
                                ex: int,
                                ey: int,
                                xi_eta_x: np.ndarray, w_x: np.ndarray,
                                xi_eta_y: np.ndarray, w_y: np.ndarray,
                                ) -> np.ndarray:
        """
        3x3 local Galerkin block for S: ∬ φ_i(x) G(x,y) φ_j(y) dΓ_x dΓ_y.
        """
        v0x, e1x, e2x, _ = self._geom_from_mesh(mesh, ex)
        v0y, e1y, e2y, _ = self._geom_from_mesh(mesh, ey)

        xq, a2x = self._map_to_element(v0x, e1x, e2x, xi_eta_x)           # (Qx,3), scalar
        yq, a2y = self._map_to_element(v0y, e1y, e2y, xi_eta_y)           # (Qy,3), scalar
        wX = (w_x * (a2x / 2.0)).astype(self._dtype, copy=False)          # (Qx,)
        wY = (w_y * (a2y / 2.0)).astype(self._dtype, copy=False)          # (Qy,)

        Nx, _ = self._p1_shapes_and_surface_grads(xi_eta_x, e1x, e2x)     # (Qx,3),(3,3)
        Ny, _ = self._p1_shapes_and_surface_grads(xi_eta_y, e1y, e2y)     # (Qy,3),(3,3)

        _, rxy, _ = r_vec(xq[:, None, :], yq[None, :, :])                  # (Qx,Qy)
        Gxy = G(rxy, self.k).astype(self._dtype, copy=False)

        L = (Nx * wX[:, None]).astype(self._dtype, copy=False)            # (Qx,3)
        R = (Ny * wY[:, None]).astype(self._dtype, copy=False)            # (Qy,3)
        B = np.einsum("qi, qk, kj -> ij", L, Gxy, R, optimize=True)
        return B

    def double_layer_block_P1P1(self,
                                mesh,
                                ex: int,
                                ey: int,
                                xi_eta_x: np.ndarray, w_x: np.ndarray,
                                xi_eta_y: np.ndarray, w_y: np.ndarray,
                                ) -> np.ndarray:
        """
        3x3 local Galerkin block for D: ∬ φ_i(x) ∂G/∂n_y φ_j(y) dΓ_x dΓ_y.
        """
        v0x, e1x, e2x, _  = self._geom_from_mesh(mesh, ex)
        v0y, e1y, e2y, ny = self._geom_from_mesh(mesh, ey)

        xq, a2x = self._map_to_element(v0x, e1x, e2x, xi_eta_x)
        yq, a2y = self._map_to_element(v0y, e1y, e2y, xi_eta_y)
        wX = (w_x * (a2x / 2.0)).astype(self._dtype, copy=False)
        wY = (w_y * (a2y / 2.0)).astype(self._dtype, copy=False)

        Nx, _ = self._p1_shapes_and_surface_grads(xi_eta_x, e1x, e2x)
        Ny, _ = self._p1_shapes_and_surface_grads(xi_eta_y, e1y, e2y)

        _, rxy, rhat = r_vec(xq[:, None, :], yq[None, :, :])               # (Qx,Qy,3)
        Gxy = G(rxy, self.k).astype(self._dtype, copy=False)
        dGr = dG_dr(rxy, Gxy, self.k).astype(self._dtype, copy=False)
        ny_b = ny[None, None, :]                                           # (1,1,3)
        dGdnY = dG_dn_y(rhat, dGr, ny_b).astype(self._dtype, copy=False)   # (Qx,Qy)

        L = (Nx * wX[:, None]).astype(self._dtype, copy=False)
        R = (Ny * wY[:, None]).astype(self._dtype, copy=False)
        B = np.einsum("qi, qk, kj -> ij", L, dGdnY, R, optimize=True)
        return B

    def adjoint_double_layer_block_P1P1(self,
                                        mesh,
                                        ex: int,
                                        ey: int,
                                        xi_eta_x: np.ndarray, w_x: np.ndarray,
                                        xi_eta_y: np.ndarray, w_y: np.ndarray,
                                    ) -> np.ndarray:
        """
        3x3 local Galerkin block for K': ∬ ∂G/∂n_x φ_i(x) φ_j(y) dΓ_x dΓ_y.
        """
        v0x, e1x, e2x, nx = self._geom_from_mesh(mesh, ex)
        v0y, e1y, e2y, _  = self._geom_from_mesh(mesh, ey)

        xq, a2x = self._map_to_element(v0x, e1x, e2x, xi_eta_x)
        yq, a2y = self._map_to_element(v0y, e1y, e2y, xi_eta_y)
        wX = (w_x * (a2x / 2.0)).astype(self._dtype, copy=False)
        wY = (w_y * (a2y / 2.0)).astype(self._dtype, copy=False)

        Nx, _ = self._p1_shapes_and_surface_grads(xi_eta_x, e1x, e2x)
        Ny, _ = self._p1_shapes_and_surface_grads(xi_eta_y, e1y, e2y)

        _, rxy, rhat = r_vec(xq[:, None, :], yq[None, :, :])               # (Qx,Qy,3)
        Gxy = G(rxy, self.k).astype(self._dtype, copy=False)
        dGr = dG_dr(rxy, Gxy, self.k).astype(self._dtype, copy=False)
        nx_b = nx[None, None, :]                                           # (1,1,3)
        dGdnX = dG_dn_x(rhat, dGr, nx_b).astype(self._dtype, copy=False)   # (Qx,Qy)

        L = (Nx * wX[:, None]).astype(self._dtype, copy=False)
        R = (Ny * wY[:, None]).astype(self._dtype, copy=False)
        B = np.einsum("qi, qk, kj -> ij", L, dGdnX, R, optimize=True)
        return B

    def hypersingular_block_P1P1_maue(self,
                                      mesh,
                                      ex: int,
                                      ey: int,
                                      xi_eta_x: np.ndarray, w_x: np.ndarray,
                                      xi_eta_y: np.ndarray, w_y: np.ndarray,
                                      ) -> np.ndarray:
        """
        3x3 Galerkin hypersingular block (symmetric Maue/weak form):

        N_ij = ∬ [ ∇_Γ φ_i(x)·∇_Γ φ_j(y) + k² (n_x·n_y) φ_i(x)φ_j(y) ] G(x,y) dΓ_x dΓ_y
        """
        v0x, e1x, e2x, nx = self._geom_from_mesh(mesh, ex)
        v0y, e1y, e2y, ny = self._geom_from_mesh(mesh, ey)

        xq, a2x = self._map_to_element(v0x, e1x, e2x, xi_eta_x)
        yq, a2y = self._map_to_element(v0y, e1y, e2y, xi_eta_y)
        wX = (w_x * (a2x / 2.0)).astype(self._dtype, copy=False)
        wY = (w_y * (a2y / 2.0)).astype(self._dtype, copy=False)

        Nx, gradNx = self._p1_shapes_and_surface_grads(xi_eta_x, e1x, e2x)  # (Qx,3),(3,3)
        Ny, gradNy = self._p1_shapes_and_surface_grads(xi_eta_y, e1y, e2y)  # (Qy,3),(3,3)

        _, rxy, _ = r_vec(xq[:, None, :], yq[None, :, :])                   # (Qx,Qy)
        Gxy = G(rxy, self.k).astype(self._dtype, copy=False)

        # k^2 (n_x·n_y) φ_i φ_j term
        nx_dot_ny = float(np.dot(nx, ny))
        L_phi = (Nx * wX[:, None]).astype(self._dtype, copy=False)
        R_phi = (Ny * wY[:, None]).astype(self._dtype, copy=False)
        B_mass = (self.k**2) * nx_dot_ny * np.einsum("qi,qk,kj->ij", L_phi, Gxy, R_phi, optimize=True)

        # ∇Γφ_i · ∇Γφ_j term: gradients are constant on affine P1 triangles
        Ggrad = gradNx @ gradNy.T                                            # (3,3), real
        scalar = np.einsum("qk,q,k->", Gxy, wX, wY, optimize=True)          # complex scalar
        B_grad = (scalar * Ggrad).astype(self._dtype, copy=False)

        return B_grad + B_mass
