import numpy as np
from acoustic_BEM.kernels import (r_vec, G, dG_dr, 
                                  dG_dn_y, dG_dn_x, 
                                  d2G_dn_x_dn_y,
                                  reflect_points,
                                  reflect_vectors)
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
                 k: float,
                 kernel_mode: str = 'free',
                 plane_point: np.ndarray | None = None,
                 plane_normal: np.ndarray | None = None,):
        """
        Initialize the element integrator.

        Args:
            k: Wavenumber for Helmholtz kernel.
            kernel_mode: 'free' for free space, 'plane' for half-space with
                plane boundary.
            plane_point: A point on the plane boundary (required if kernel_mode
                is 'plane').
            plane_normal: Normal vector of the plane boundary (required if 
                kernel_mode is 'plane').
        """
        self.k = k
        self._dtype = np.complex128

        if kernel_mode not in ['free', 'halfspace_neumann']:
            raise ValueError("Invalid kernel_mode. Must be 'free' or "
                             "'halfspace_neumann'.")
        self.kernel_mode = kernel_mode

        if self.kernel_mode == 'halfspace_neumann':
            if plane_point is None or plane_normal is None:
                raise ValueError("plane_point and plane_normal must be "
                                 "provided for 'halfspace_neumann' mode.")
            
            n = np.asarray(plane_normal, dtype=np.float64)
            n /= np.linalg.norm(n)
            self.plane_point = np.asarray(plane_point, dtype=np.float64)
            self.plane_normal = n

    def single_layer(self, 
                     x: np.ndarray, 
                     y_phys: np.ndarray, 
                     w_phys: np.ndarray, 
                     N: np.ndarray) -> np.ndarray:
        """
        Compute single layer potential integral:
        ∫_T G(x,y) N_j(y) dS_y

        Args:
            x: Observation point, shape (3,).
            y_phys: Physical quadrature points on the source triangle, shape 
                (Q, 3).
            w_phys: Quadrature weights on the source triangle, shape (Q,).
            N: Shape function values at quadrature points, shape (Q, 3).

        Returns:
            Local vector for the triangle, shape (3,).
        """
        r = r_vec(x[None, None, :], y_phys)[1]
        Gv = G(r, self.k)
        acc =  np.sum((w_phys[:, :, None] * Gv[:, :, None]) * N[None, :, :], 
                      axis=1)
        
        if self.kernel_mode == 'halfspace_neumann':
            y_img = reflect_points(y_phys, 
                                   self.plane_point,
                                   self.plane_normal)
            r_img = r_vec(x[None, None, :], y_img)[1]
            Gv_img = G(r_img, self.k)
            acc += np.sum((w_phys[:, :, None] * Gv_img[:, :, None]) * \
                            N[None, :, :], axis=1)
            
        return acc

    def double_layer(self, 
                     x: np.ndarray, 
                     y_phys: np.ndarray, 
                     w_phys: np.ndarray, 
                     N: np.ndarray, 
                     n_y: np.ndarray) -> np.ndarray:
        """
        Compute double layer potential integral:
        ∫_T ∂G(x,y)/∂n_y N_j(y) dS_y

        Args:
            x: Observation point, shape (3,).
            y_phys: Physical quadrature points on the source triangle, shape 
                (Q, 3).
            w_phys: Quadrature weights on the source triangle, shape (Q,).
            N: Shape function values at quadrature points, shape (Q, 3).
            n_y: Normal vector at source triangle, shape (3,).

        Returns:
            Local vector for the triangle, shape (3,).
        """

        r, rhat = r_vec(x[None, None, :], y_phys)[1:]
        Gv = G(r, self.k)
        dGr = dG_dr(r, Gv, self.k)
        dGdnY = dG_dn_y(rhat, dGr, n_y[:, None, :])
        acc =  np.sum((w_phys[:, :, None] * dGdnY[:, :, None]) * N[None, :, :], 
                      axis=1)
        
        if self.kernel_mode == 'halfspace_neumann':
            y_img = reflect_points(y_phys, 
                                   self.plane_point,
                                   self.plane_normal)
            n_y_img = reflect_vectors(n_y, self.plane_normal)
            r_img, rhat_img = r_vec(x[None, None, :], y_img)[1:]
            Gv_img = G(r_img, self.k)
            dGr_img = dG_dr(r_img, Gv_img, self.k)
            dGdnY_img = dG_dn_y(rhat_img, dGr_img, n_y_img[:, None, :])
            acc += np.sum((w_phys[:, :, None] * dGdnY_img[:, :, None]) * \
                            N[None, :, :], axis=1)
            
        return acc

    def adjoint_double_layer(self, 
                             x: np.ndarray, 
                             x_normal: np.ndarray, 
                             y_phys: np.ndarray, 
                             w_phys: np.ndarray, 
                             N: np.ndarray) -> np.ndarray:
        """
        Compute adjoint double layer potential integral:
        ∫_T ∂G(x,y)/∂n_x N_j(y) dS_y

        Args:
            x: Observation point, shape (3,).
            x_normal: Normal vector at observation point, shape (3,).
            y_phys: Physical quadrature points on the source triangle, shape
                (Q, 3).
            w_phys: Quadrature weights on the source triangle, shape (Q,).
            N: Shape function values at quadrature points, shape (Q, 3).

        Returns:
            Local vector for the triangle, shape (3,).
        """
        r, rhat = r_vec(x[None, None, :], y_phys)[1:]
        Gv = G(r, self.k)
        dGr = dG_dr(r, Gv, self.k)
        nx = np.broadcast_to(x_normal[None, None, :], y_phys.shape)
        dGdnX = dG_dn_x(rhat, dGr, nx)
        acc =  np.sum((w_phys[:, :, None] * dGdnX[:, :, None]) * N[None, :, :], 
                      axis=1)
        
        if self.kernel_mode == 'halfspace_neumann':
            y_img = reflect_points(y_phys, 
                                   self.plane_point,
                                   self.plane_normal)
            r_img, rhat_img = r_vec(x[None, None, :], y_img)[1:]
            Gv_img = G(r_img, self.k)
            dGr_img = dG_dr(r_img, Gv_img, self.k)
            dGdnX_img = dG_dn_x(rhat_img, dGr_img, x_normal[None, None, :])
            acc += np.sum((w_phys[:, :, None] * dGdnX_img[:, :, None]) *
                            N[None, :, :], axis=1)
            
        return acc

    def hypersingular_layer(self, 
                            x: np.ndarray, 
                            x_normal: np.ndarray, 
                            y_phys: np.ndarray, 
                            w_phys: np.ndarray, 
                            N: np.ndarray, 
                            n_y: np.ndarray) -> np.ndarray:
        """
        Compute adjoint double layer potential integral:
        ∫_T ∂²G(x,y)/(∂n_x∂n_y) N_j(y) dS_y
        Args:
            x: Observation point, shape (3,).
            x_normal: Normal vector at observation point, shape (3,).
            y_phys: Physical quadrature points on the source triangle, shape
                (Q, 3).
            w_phys: Quadrature weights on the source triangle, shape (Q,).
            N: Shape function values at quadrature points, shape (Q, 3).
            n_y: Normal vector at source triangle, shape (3,).

        Returns:
            Local vector for the triangle, shape (3,).
        """
        #TODO: recheck the halfspace implementationl and if all broadcasting is correct
        r, rhat = r_vec(x[None, None, :], y_phys)[1:]
        Gv = G(r, self.k)
        d2 = d2G_dn_x_dn_y(r_hat=rhat, 
                           r=r, 
                           n_x=np.broadcast_to(x_normal[None, None, :], 
                                               y_phys.shape),
                           n_y=n_y[:, None, :], G_vals=Gv, k=self.k)
        acc = np.sum((w_phys[:, :, None] * d2[:, :, None]) * N[None, :, :], 
                      axis=1)
        
        if self.kernel_mode == 'halfspace_neumann':
            y_img = reflect_points(y_phys, 
                                   self.plane_point,
                                   self.plane_normal)
            n_y_img = reflect_vectors(n_y, self.plane_normal)
            r_img, rhat_img = r_vec(x[None, None, :], y_img)[1:]
            Gv_img = G(r_img, self.k)
            d2_img = d2G_dn_x_dn_y(
                r_hat=rhat_img, 
                r=r_img, 
                n_x=np.broadcast_to(x_normal[None, None, :], y_phys.shape),
                n_y=n_y_img[:, None, :], 
                G_vals=Gv_img, 
                k=self.k)
            acc += np.sum((w_phys[:, :, None] * d2_img[:, :, None]) * \
                          N[None, :, :], axis=1)
            
        return acc
    
    def hypersingular_layer_reg(self,
                                x: np.ndarray,
                                x_normal: np.ndarray,
                                y_v0: np.ndarray,
                                y_e1: np.ndarray,
                                y_e2: np.ndarray,
                                y_normals: np.ndarray,
                                xi_eta: np.ndarray,
                                w: np.ndarray,) -> np.ndarray:
        """
        Compute hypersingular integrals (regularised).
        
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
        N_vals = shape_functions_P1(xi_eta)
        
        nx_dot_ny = np.einsum("i,ki->k", x_normal, y_normals)
        r_norm, r_hat = r_vec(x[None, None, :], y_phys)[1:]
        G_vals = G(r_norm, self.k)
        
        dG_dr_vals = dG_dr(r_norm, G_vals, self.k)
        grad_y_G = -dG_dr_vals[:, :, None] * r_hat

        part1 = np.einsum("kq,k,qj,kq->kj", 
                          G_vals, nx_dot_ny, N_vals, w_phys) * (self.k**2)
        
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

        if self.kernel_mode == 'halfspace_neumann':
            y_img = reflect_points(y_phys, 
                                   self.plane_point,
                                   self.plane_normal)
            y_normals_img = reflect_vectors(y_normals, self.plane_normal)
            r_norm_img, r_hat_img = r_vec(x[None, None, :], y_img)[1:]
            G_img = G(r_norm_img, self.k)
            dGr_img = dG_dr(r_norm_img, G_img, self.k)
            grad_y_G_img = -dGr_img[:, :, None] * r_hat_img

            nx_dot_ny_img = np.einsum("i,ki->k", x_normal, y_normals_img)
            part1_img = self.k**2 * np.einsum("kq,k,qj,kq->kj",
                                                G_img, 
                                                nx_dot_ny_img, 
                                                N_vals, 
                                                w_phys)
            
            dot_img = np.einsum("kqo,kjo->kjq", grad_y_G_img, grad_N)
            part2_img = np.einsum("kjq,kq->kj", dot_img, w_phys)

            part1 += part1_img
            part2 += part2_img

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