import numpy as np
from typing import Tuple, Literal, Optional
from source.kernels import r_vec, G, dG_dr, dG_dn_y, dG_dn_x, d2G_dn_x_dn_y
from source.quadrature import (standard_triangle_quad, subdivide_triangle_quad,
                       telles_rule, duffy_rule, map_to_physical_triangle_batch,
                       shape_functions_P1)

from source.mesh import Mesh

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
            k (float): Wavenumber for Helmholtz kernel.
            regular_quad_order (int): Quadrature order for regular integrals.
            near_singular_levels (int): Number of subdivision levels for near-
                singular integrals.
            singular_n_leg (int): Number of Gauss-Legendre points for singular
                integrals.
            near_singular_threshold (float): Distance threshold for near-sing.
                detection (in terms of element characteristic size).
            singular_threshold (float): Distance threshold for singular
                detection (in terms of element characteristic size).
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

    def single_layer_block_P1P1(self,
                                mesh: Mesh,
                                ex: int,
                                ey: int,
                                xi_eta_x: np.ndarray, w_x: np.ndarray,
                                xi_eta_y: np.ndarray, w_y: np.ndarray,
                                ) -> np.ndarray:
        """
        3x3 local Galerkin block for S: ∬ φ_i(x) G(x,y) φ_j(y) dΓ_x dΓ_y.

        Args:
            mesh (Mesh): Mesh object containing element information.
            ex (int): Index of observation element.
            ey (int): Index of source element.
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
        
        xq, a2x = map_to_physical_triangle_batch(
            xi_eta_x, v0x[None, :], e1x[None, :], e2x[None, :])
        
        yq, a2y = map_to_physical_triangle_batch(
            xi_eta_y, v0y[None, :], e1y[None, :], e2y[None, :])
        
        xq = xq[0]
        yq = yq[0]

        a2x = float(a2x[0])
        a2y = float(a2y[0])

        wX = (w_x * a2x).astype(self._dtype, copy=False)
        wY = (w_y * a2y).astype(self._dtype, copy=False)

        Nx = shape_functions_P1(xi_eta_x)
        Ny = shape_functions_P1(xi_eta_y)

        rxy = r_vec(xq[:, None, :], yq[None, :, :])[1]
        Gxy = G(rxy, self.k).astype(self._dtype, copy=False)

        L = (Nx * wX[:, None]).astype(self._dtype, copy=False) 
        R = (Ny * wY[:, None]).astype(self._dtype, copy=False)
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

        Args:
            mesh (Mesh): Mesh object with v0, e1, e2, elements, normals 
                attributes.
            ex (int): Index of observation element.
            ey (int): Index of source element.
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
        
        ny = mesh.n_hat[ey]

        xq, a2x = map_to_physical_triangle_batch(
            xi_eta_x, v0x[None, :], e1x[None, :], e2x[None, :])
        
        yq, a2y = map_to_physical_triangle_batch(
            xi_eta_y, v0y[None, :], e1y[None, :], e2y[None, :])
        
        xq = xq[0]
        yq = yq[0]

        a2x = float(a2x[0])
        a2y = float(a2y[0])
        
        wX = (w_x * a2x).astype(self._dtype, copy=False)
        wY = (w_y * a2y).astype(self._dtype, copy=False)

        Nx = shape_functions_P1(xi_eta_x)
        Ny = shape_functions_P1(xi_eta_y)

        rxy, rhat = r_vec(xq[:, None, :], yq[None, :, :])[1:]  
        Gxy = G(rxy, self.k).astype(self._dtype, copy=False)
        dGr = dG_dr(rxy, Gxy, self.k).astype(self._dtype, copy=False)
        ny_b = ny[None, None, :]   
        dGdnY = dG_dn_y(rhat, dGr, ny_b).astype(self._dtype, copy=False) 

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

        Args:
            mesh (Mesh): Mesh object with v0, e1, e2, elements, normals
                attributes.
            ex (int): Index of observation element.
            ey (int): Index of source element.
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
        
        yq, a2y = map_to_physical_triangle_batch(
            xi_eta_y, v0y[None, :], e1y[None, :], e2y[None, :])
        
        xq = xq[0]
        yq = yq[0]

        a2x = float(a2x[0])
        a2y = float(a2y[0])
        
        wX = (w_x * a2x).astype(self._dtype, copy=False)
        wY = (w_y * a2y).astype(self._dtype, copy=False)

        Nx = shape_functions_P1(xi_eta_x)
        Ny = shape_functions_P1(xi_eta_y)

        rxy, rhat = r_vec(xq[:, None, :], yq[None, :, :])[1:]
        Gxy = G(rxy, self.k).astype(self._dtype, copy=False)
        dGr = dG_dr(rxy, Gxy, self.k).astype(self._dtype, copy=False)
        nx_b = nx[None, None, :]
        dGdnX = dG_dn_x(rhat, dGr, nx_b).astype(self._dtype, copy=False)

        L = (Nx * wX[:, None]).astype(self._dtype, copy=False)
        R = (Ny * wY[:, None]).astype(self._dtype, copy=False)
        B = np.einsum("qi, qk, kj -> ij", L, dGdnX, R, optimize=True)
        return B

    def hypersingular_block_P1P1(self,
                                mesh,
                                ex: int,
                                ey: int,
                                xi_eta_x: np.ndarray, w_x: np.ndarray,
                                xi_eta_y: np.ndarray, w_y: np.ndarray,
                                ) -> np.ndarray:
        """
        3x3 Galerkin hypersingular block:

        N_ij = ∬ [ ∇_Γ φ_i(x)·∇_Γ φ_j(y) + k² (n_x·n_y) φ_i(x)φ_j(y) ] G(x,y) 
        dΓ_x dΓ_y

        Args:
            mesh (Mesh): Mesh object with v0, e1, e2, elements, normals
                attributes.
            ex (int): Index of observation element.
            ey (int): Index of source element.
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
            xi_eta_x, v0x[None, :], e1x[None, :], e2x[None, :])
        
        yq, a2y = map_to_physical_triangle_batch(
            xi_eta_y, v0y[None, :], e1y[None, :], e2y[None, :])
        
        xq = xq[0]
        yq = yq[0]

        a2x = float(a2x[0])
        a2y = float(a2y[0])
        
        wX = (w_x * a2x).astype(self._dtype, copy=False)
        wY = (w_y * a2y).astype(self._dtype, copy=False)

        Nx = shape_functions_P1(xi_eta_x)
        Ny = shape_functions_P1(xi_eta_y)

        _, rxy, rhat = r_vec(xq[:, None, :], yq[None, :, :]) 
        Gxy = G(rxy, self.k)
        nx_b = nx[None, None, :] 
        ny_b = ny[None, None, :] 
        d2Gxy = d2G_dn_x_dn_y(r_hat=rhat, r=rxy,
                            n_x=nx_b, n_y=ny_b,
                            G_vals=Gxy, k=self.k) 
        
        L = (Nx * wX[:, None]).astype(self._dtype, copy=False) 
        R = (Ny * wY[:, None]).astype(self._dtype, copy=False) 
        B = np.einsum("qi, qk, kj -> ij", L, d2Gxy, R, optimize=True) 

        return B
