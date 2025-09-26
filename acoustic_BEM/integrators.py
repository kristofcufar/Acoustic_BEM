import numpy as np
from acoustic_BEM.kernels import r_vec, G, dG_dr, dG_dn_y, dG_dn_x, d2G_dn_x_dn_y
from acoustic_BEM.quadrature import (map_to_physical_triangle_batch, 
                               shape_functions_P1,
                               shape_function_gradients_P1)

from acoustic_BEM.mesh import Mesh

class ElementIntegratorCollocation:
    """
    Element-wise integrator for acoustic boundary element method.
    
    Provides efficient vectorized integration methods for single layer, double 
    layer, adjoint double layer, and hypersingular boundary integrals.
    """
    
    def __init__(self, 
                 k: float,):
        """
        Initialize the element integrator.

        Args:
            k: Wavenumber for Helmholtz kernel.
        """
        self.k = k

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
            x (np.ndarray): Observation point, shape (3,).
            y_v0 (np.ndarray): First vertices of source triangles, shape 
                (K, 3).
            y_e1 (np.ndarray): First edge vectors of source triangles, shape
                (K, 3).
            y_e2 (np.ndarray): Second edge vectors of source triangles, shape
                (K, 3).
            xi_eta (np.ndarray): Quadrature points in reference triangle, shape
                (Q, 2).
            w (np.ndarray): Quadrature weights, shape (Q,).

        Returns:
            Local vectors for each triangle, shape (K, 3).
        """
        y_phys, a2 = map_to_physical_triangle_batch(xi_eta, y_v0, y_e1, y_e2)
        N = shape_functions_P1(xi_eta)
        
        _, r_norm, _ = r_vec(x[None, None, :], y_phys)
        G_vals = G(r_norm, self.k)
    
        w_phys = w[None, :] * a2[:, None]
        integrand = w_phys[:, :, None] * G_vals[:, :, None] * N[None, :, :]
        
        return np.sum(integrand, axis=1)

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
            x (np.ndarray): Observation point, shape (3,).
            y_v0 (np.ndarray): First vertices of source triangles, shape 
                (K, 3).
            y_e1 (np.ndarray): First edge vectors of source triangles, shape
                (K, 3).
            y_e2 (np.ndarray): Second edge vectors of source triangles, shape
                (K, 3).
            y_normals (np.ndarray): Normal vectors of source triangles, shape
                (K, 3).
            xi_eta (np.ndarray): Quadrature points in reference triangle, shape
                (Q, 2).
            w (np.ndarray): Quadrature weights, shape (Q,).

        Returns:
            Local vectors for each triangle, shape (K, 3).
        """
        y_phys, a2 = map_to_physical_triangle_batch(xi_eta, y_v0, y_e1, y_e2)
        N = shape_functions_P1(xi_eta)
        
        _, r_norm, r_hat = r_vec(x[None, None, :], y_phys)
        G_vals = G(r_norm, self.k)
        dG_dr_vals = dG_dr(r_norm, G_vals, self.k)

        n_y_broadcast = y_normals[:, None, :]
        dG_dn_y_vals = dG_dn_y(r_hat, dG_dr_vals, n_y_broadcast)

        w_phys = w[None, :] * a2[:, None]
        integrand = w_phys[:, :, None] * \
            dG_dn_y_vals[:, :, None] * N[None, :, :]
        
        return np.sum(integrand, axis=1)

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
            x (np.ndarray): Observation point, shape (3,).
            x_normal (np.ndarray): Normal vector at observation point, shape
                (3,).
            y_v0 (np.ndarray): First vertices of source triangles, shape
                (K, 3).
            y_e1 (np.ndarray): First edge vectors of source triangles, shape
                (K, 3).
            y_e2 (np.ndarray): Second edge vectors of source triangles, shape
                (K, 3).
            xi_eta (np.ndarray): Quadrature points in reference triangle, shape
                (Q, 2).
            w (np.ndarray): Quadrature weights, shape (Q,).

        Returns:
            Local vectors for each triangle, shape (K, 3).
        """
        y_phys, a2 = map_to_physical_triangle_batch(xi_eta, y_v0, y_e1, y_e2)
        N = shape_functions_P1(xi_eta)  

        _, r_norm, r_hat = r_vec(x[None, None, :], y_phys)
        G_vals = G(r_norm, self.k)
        dG_dr_vals = dG_dr(r_norm, G_vals, self.k)
        
        n_x_broadcast = np.broadcast_to(x_normal[None, None, :], y_phys.shape)
        dG_dn_x_vals = dG_dn_x(r_hat, dG_dr_vals, n_x_broadcast)
        
        w_phys = w[None, :] * a2[:, None]
        
        integrand = \
            w_phys[:, :, None] * dG_dn_x_vals[:, :, None] * N[None, :, :]
        
        return np.sum(integrand, axis=1)

    def hypersingular_batch(self,
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
            x (np.ndarray): Observation point, shape (3,).
            x_normal (np.ndarray): Normal vector at observation point, shape
                (3,).
            y_v0 (np.ndarray): First vertices of source triangles, shape
                (K, 3).
            y_e1 (np.ndarray): First edge vectors of source triangles, shape
                (K, 3).
            y_e2 (np.ndarray): Second edge vectors of source triangles, shape
                (K, 3).
            y_normals (np.ndarray): Normal vectors of source triangles, shape
                (K, 3).
            xi_eta (np.ndarray): Quadrature points in reference triangle, shape
                (Q, 2).
            w (np.ndarray): Quadrature weights, shape (Q,).

        Returns:
            Local vectors for each triangle, shape (K, 3).
        """
        y_phys, a2 = map_to_physical_triangle_batch(xi_eta, y_v0, y_e1, y_e2)
        N = shape_functions_P1(xi_eta)
        
        _, r_norm, r_hat = r_vec(x[None, None, :], y_phys)
        G_vals = G(r_norm, self.k)

        n_x_broadcast = np.broadcast_to(x_normal[None, None, :], y_phys.shape)
        n_y_broadcast = y_normals[:, None, :]
        
        d2G_vals = d2G_dn_x_dn_y(r_hat = r_hat,
                                 r = r_norm,
                                 n_x = n_x_broadcast,
                                 n_y = n_y_broadcast,
                                 G_vals = G_vals,
                                 k = self.k)
        
        w_phys = w[None, :] * a2[:, None]
        
        integrand = w_phys[:, :, None] * d2G_vals[:, :, None] * N[None, :, :]
        
        return np.sum(integrand, axis=1)
    
    def hypersingular_batch_reg(self,
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
        """
        K = len(y_v0)
        y_phys, a2 = map_to_physical_triangle_batch(xi_eta, y_v0, y_e1, y_e2)
        w_phys = w[None, :] * a2[:, None]
        
        nx_dot_ny = np.einsum("i,ki->k", x_normal, y_normals)
        
        _, r_norm, r_hat = r_vec(x[None, None, :], y_phys)
        G_vals = G(r_norm, self.k)
        N_vals = shape_functions_P1(xi_eta)
        
        part1 = np.einsum("kq,k,qj,kq->kj", 
                          G_vals, nx_dot_ny, N_vals, w_phys) * (self.k**2)
        
        dG_dr_vals = dG_dr(r_norm, G_vals, self.k)
        grad_y_G = -dG_dr_vals[:, :, None] * r_hat
        
        dN_ref = np.array([[-1.0, -1.0],
                          [ 1.0,  0.0],
                          [ 0.0,  1.0]], dtype=np.float64)
        
        G11 = np.einsum("ki,ki->k", y_e1, y_e1)
        G12 = np.einsum("ki,ki->k", y_e1, y_e2)
        G22 = np.einsum("ki,ki->k", y_e2, y_e2)
        
        det_G = G11 * G22 - G12 * G12
        det_G = np.where(np.abs(det_G) < 1e-14, 1e-14, det_G)
        
        Ginv = np.zeros((K, 2, 2))
        Ginv[:, 0, 0] = G22 / det_G
        Ginv[:, 0, 1] = -G12 / det_G
        Ginv[:, 1, 0] = -G12 / det_G
        Ginv[:, 1, 1] = G11 / det_G
        
        g1 = Ginv[:, 0, 0, None] * y_e1 + Ginv[:, 0, 1, None] * y_e2
        g2 = Ginv[:, 1, 0, None] * y_e1 + Ginv[:, 1, 1, None] * y_e2 
        g = np.stack([g1, g2], axis=1)
        
        grad_N = np.einsum("kao,ja->kjo", g, dN_ref)
        
        dot_products = np.einsum("kqo,kjo->kjq", grad_y_G, grad_N)
        part2 = np.einsum("kjq,kq->kj", dot_products, w_phys)
        
        return part1 + part2
    

class ElementIntegratorGalerkin:
    """
    Element–element (P1–P1) Galerkin integrator for acoustic BEM.

    Provides 3x3 local blocks for the single layer (S), double layer (D),
    adjoint double layer (K'), and hypersingular (N) operators. Quadrature
    selection mirrors the collocation class API (standard / subdivide / duffy 
    / telles).
    """

    def __init__(self,
                 k: float,):
        """
        Initialize the Galerkin element–element integrator.

        Args:
            k (float): Wavenumber for Helmholtz kernel.
        """
        self.k = k
        self._dtype = np.complex128
    
    def single_layer_block_P1P1_batch(self,
                                      mesh: Mesh,
                                      ex: int,
                                      ey: np.ndarray,
                                      xi_eta_x: np.ndarray, w_x: np.ndarray,
                                      xi_eta_y: np.ndarray, w_y: np.ndarray,
                                      ) -> np.ndarray:

        v0x, e1x, e2x = mesh.v0[ex], \
                        mesh.e1[ex], \
                        mesh.e2[ex]
        
        v0y, e1y, e2y = mesh.v0[ey], \
                        mesh.e1[ey], \
                        mesh.e2[ey]
        
        xq, a2x = map_to_physical_triangle_batch(
            xi_eta_x, v0x[None, :], e1x[None, :], e2x[None, :])
        
        xq = xq[0]
        
        yq, a2y = map_to_physical_triangle_batch(
            xi_eta_y, v0y, e1y, e2y)

        wX = (w_x * float(a2x[0])).astype(self._dtype, copy=False)
        wY = (w_y[None, :] * a2y[:, None]).astype(self._dtype, copy=False)

        Nx = shape_functions_P1(xi_eta_x)
        Ny = shape_functions_P1(xi_eta_y)

        L = (Nx * wX[:, None]).astype(self._dtype, copy=False)
        R = (Ny[None, :, :] * wY[:, :, None]).astype(self._dtype, copy=False)
        
        diff = xq[:, None, None, :] - yq[None, :, :, :]
        rxy = np.linalg.norm(diff, axis=-1)
        Gxy = G(rxy, self.k).astype(self._dtype, copy=False)

        return np.einsum('qi,qbk,bkj->bij', L, Gxy, R, optimize=True)
    
    def double_layer_block_P1P1_batch(self,
                                      mesh,
                                      ex: int,
                                      ey: np.ndarray,
                                      xi_eta_x: np.ndarray, w_x: np.ndarray,
                                      xi_eta_y: np.ndarray, w_y: np.ndarray,
                                      ) -> np.ndarray:
        v0x, e1x, e2x = mesh.v0[ex],\
                        mesh.e1[ex],\
                        mesh.e2[ex]
        
        v0y, e1y, e2y = mesh.v0[ey],\
                        mesh.e1[ey],\
                        mesh.e2[ey]
        
        ny = mesh.n_hat[ey]
        xq, a2x = map_to_physical_triangle_batch(
            xi_eta_x, v0x[None, :], e1x[None, :], e2x[None, :])
        
        xq = xq[0]

        yq, a2y = map_to_physical_triangle_batch(
            xi_eta_y, v0y, e1y, e2y)
        
        wX = (w_x * float(a2x[0])).astype(self._dtype, copy=False)
        wY = (w_y[None, :] * a2y[:, None]).astype(self._dtype, copy=False)

        Nx = shape_functions_P1(xi_eta_x)
        Ny = shape_functions_P1(xi_eta_y)

        L = (Nx * wX[:, None]).astype(self._dtype, copy=False)
        R = (Ny[None, :, :] * wY[:, :, None]).astype(self._dtype, copy=False)

        diff = xq[:, None, None, :] - yq[None, :, :, :]
        rxy = np.linalg.norm(diff, axis=-1)
        rhat = diff / rxy[:, :, :, None]
        Gxy = G(rxy, self.k).astype(self._dtype, copy=False)
        dGr = dG_dr(rxy, Gxy, self.k).astype(self._dtype, copy=False)
        ndoty = np.einsum('qbk i, b i -> qbk', rhat, ny)
        K = -dGr * ndoty

        return np.einsum("qi, qbk, bkj -> bij", L, K, R, optimize=True)

    def adjoint_double_layer_block_P1P1_batch(self,
                                        mesh,
                                        ex: int,
                                        ey: np.ndarray,
                                        xi_eta_x: np.ndarray, w_x: np.ndarray,
                                        xi_eta_y: np.ndarray, w_y: np.ndarray,
                                    ) -> np.ndarray:
        """
        3x3 local Galerkin block for K': ∬ ∂G/∂n_x φ_i(x) φ_j(y) dΓ_x dΓ_y.

        Args:
            mesh (Mesh): Mesh object with v0, e1, e2, elements, normals
                attributes.
            ex (int): Index of observation element.
            ey (np.ndarray): Indices of source elements.
            xi_eta_x (np.ndarray): Quadrature points for x element, shape
                (Qx, 2).
            w_x (np.ndarray): Quadrature weights for x element, shape (Qx,).
            xi_eta_y (np.ndarray): Quadrature points for y element, shape
                (Qy, 2).
            w_y (np.ndarray): Quadrature weights for y element, shape (Qy,).

        Returns:
            np.ndarray: 3x3 local Galerkin block as a numpy array.
        """
        v0x, e1x, e2x = mesh.v0[ex], \
                        mesh.e1[ex], \
                        mesh.e2[ex]

        v0y, e1y, e2y = mesh.v0[ey], \
                        mesh.e1[ey], \
                        mesh.e2[ey]

        nx = mesh.n_hat[ex]

        xq, a2x = map_to_physical_triangle_batch(
            xi_eta_x, v0x[None, :], e1x[None, :], e2x[None, :])

        xq = xq[0]

        yq, a2y = map_to_physical_triangle_batch(
            xi_eta_y, v0y, e1y, e2y)
        
        wX = (w_x * float(a2x[0])).astype(self._dtype, copy=False)
        wY = (w_y[None, :] * a2y[:, None]).astype(self._dtype, copy=False)

        Nx = shape_functions_P1(xi_eta_x)
        Ny = shape_functions_P1(xi_eta_y)

        diff = xq[:, None, None, :] - yq[None, :, :, :]
        rxy = np.linalg.norm(diff, axis=-1)
        rhat = diff / rxy[:, :, :, None]
        Gxy = G(rxy, self.k).astype(self._dtype, copy=False)
        dGr = dG_dr(rxy, Gxy, self.k).astype(self._dtype, copy=False)
        
        ndotx = np.einsum('qbk i, i -> qbk', rhat, nx)
        K = dGr * ndotx

        L = (Nx * wX[:, None]).astype(self._dtype, copy=False)
        R = (Ny[None, :, :] * wY[:, :, None]).astype(self._dtype, copy=False)
        return np.einsum("qi, qbk, bkj -> bij", L, K, R, optimize=True)
    
    def hypersingular_block_P1P1_reg_batch(self, 
                                           mesh: Mesh, 
                                           ex: int, 
                                           ey: np.ndarray, 
                                           xi_eta_x: np.ndarray, 
                                           w_x: np.ndarray, 
                                           xi_eta_y: np.ndarray, 
                                           w_y: np.ndarray) -> np.ndarray:
        """
        3x3 Galerkin hypersingular block with regularization:
        N_ij = ∬ [ ∇_Γ φ_i(x)·∇_Γ φ_j(y) + 
            k² (n_x·n_y) φ_i(x)φ_j(y) ] G(x,y)dΓ_x dΓ_y

        Args:
            mesh (Mesh): Mesh object with v0, e1, e2, elements, normals
                attributes.
            ex (int): Index of observation element.
            ey (np.ndarray): Indices of source elements.
            xi_eta_x (np.ndarray): Quadrature points for x element, shape
                (Qx, 2).
            w_x (np.ndarray): Quadrature weights for x element, shape (Qx,).
            xi_eta_y (np.ndarray): Quadrature points for y element, shape
                (Qy, 2).
            w_y (np.ndarray): Quadrature weights for y element, shape (Qy,).

        Returns:
            np.ndarray: 3x3 local Galerkin block as a numpy array.
        """
        v0x, e1x, e2x = mesh.v0[ex], \
                        mesh.e1[ex], \
                        mesh.e2[ex]
        v0y, e1y, e2y = mesh.v0[ey], \
                        mesh.e1[ey], \
                        mesh.e2[ey]
        
        nx = mesh.n_hat[ex]
        ny = mesh.n_hat[ey]

        xq, a2x = map_to_physical_triangle_batch(
            xi_eta_x, v0x[None], e1x[None], e2x[None]
        )
        yq, a2y = map_to_physical_triangle_batch(
            xi_eta_y, v0y, e1y, e2y
        )
        xq = xq[0]

        wX = (w_x * float(a2x[0])).astype(self._dtype, copy=False)
        wY = (w_y[None, :] * a2y[:, None]).astype(self._dtype, copy=False)

        diff = xq[:, None, None, :] - yq[None, :, :, :]
        rxy = np.linalg.norm(diff, axis=-1)
        Gxy = G(rxy, self.k).astype(self._dtype, copy=False)

        J0 = np.einsum("q,bk,qbk->b", wX, wY, Gxy, optimize=True)
        Sblk = self.single_layer_block_P1P1_batch(mesh, ex, ey, 
                                            xi_eta_x, w_x, 
                                            xi_eta_y, w_y)
        
        nxny = np.einsum("i,bi->b", nx, ny)
        term2 = (self.k**2) * nxny[:, None, None] * Sblk

        dN_ref = shape_function_gradients_P1()
        Jx = np.stack([e1x, e2x], axis = -1)
        Jy = np.stack([e1y, e2y], axis = 2)

        Gx = np.linalg.inv(Jx.T @ Jx)
        Gy = np.linalg.inv(np.einsum('bmi,bmj->bij', Jy, Jy, optimize=True))

        gradx = Jx @ Gx @ dN_ref.T
        grady = np.einsum('bmi,bij,aj->bma', Jy, Gy, dN_ref, optimize=True)

        gamma = np.einsum('im,bmj->bij', gradx.T, grady, optimize=True)
        term1 = (J0[:, None, None] * gamma).astype(self._dtype, copy=False)

        return term1 + term2
    
    def jump_block_P1P1(self,
                        mesh: Mesh,
                        ex: int,
                        xi_eta_x: np.ndarray,
                        w_x: np.ndarray,
                       ) -> np.ndarray:
        """
        3x3 local Galerkin block for the jump term:
            C_ij = ∫_Γ  C(x) φ_i(x) φ_j(x) dΓ_x

        Args:
            mesh: Mesh with element geometry and (optionally) 
                jump_coefficients.
            ex:   Index of the (single) element to integrate over.
            xi_eta_x, w_x: quadrature rule on the reference triangle.

        Returns:
            3x3 complex block (dtype matches self._dtype).
        """
        v0x, e1x, e2x = mesh.v0[ex], mesh.e1[ex], mesh.e2[ex]
        xq, a2x = map_to_physical_triangle_batch(
            xi_eta_x, v0x[None, :], e1x[None, :], e2x[None, :]
        )

        a2x = float(a2x[0])
        wX  = (w_x * a2x).astype(self._dtype, copy=False)

        Nx = shape_functions_P1(xi_eta_x)

        node_ids = mesh.mesh_elements[ex]
        if not hasattr(mesh, "jump_coefficients"):
            c_val = np.full(3, 0.5, dtype=self._dtype)
        else:
            c_val = mesh.jump_coefficients[node_ids]

        C = Nx @ c_val
        B = Nx.T @ ((C * wX)[:, None] * Nx)

        return B