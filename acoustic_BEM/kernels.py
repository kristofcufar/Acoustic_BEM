import numpy as np
from scipy import special as sp
from typing import Callable

def r_vec(x: np.ndarray, 
          y: np.ndarray) -> np.ndarray:
    """
    Compute the vector from points y to points x.

    r_vec = x - y
    r_norm = ||r_vec||
    r_hat = r_vec / r_norm

    Args:
        x (np.ndarray): Array of shape (..., 3) representing points x.
        y (np.ndarray): Array of shape (..., 3) representing points y.

    Returns:
        r_vec (np.ndarray): Array of shape (..., 3) representing the vector 
            from y to x.
        r_norm (np.ndarray): Array of shape (...) representing the norm of 
            r_vec.
        r_hat (np.ndarray): Array of shape (..., 3) representing the unit 
            vector in the direction of r_vec.
    """
    r_vec_ = x - y
    r_norm = np.linalg.norm(r_vec_, axis=-1)
    r_norm = np.where(r_norm == 0, 1e-16, r_norm)  # Avoid division by zero
    r_hat = r_vec_ / r_norm[..., np.newaxis]
    return r_vec_, r_norm, r_hat

def G(r_norm: np.ndarray, 
      k: float) -> np.ndarray:
    """
    Compute the Green's function for the Helmholtz equation in 3D.

    G(r) = e^{ikr}/(4π r)

    Args:
        r_norm (np.ndarray): Array of shape (...) representing the distance 
            between source and field points.
        k (float): Wavenumber.

    Returns:
        G (np.ndarray): Array of shape (...) representing the Green's function.
    """
    return np.exp(1j * k * r_norm) / (4 * np.pi * r_norm)

def dG_dr(r_norm: np.ndarray, 
          G: np.ndarray,
          k: float) -> np.ndarray:
    """
    Compute the derivative of the Green's function with respect to r.

    For G(r) = e^{ikr}/(4π r):
        dG/dr = (ik - 1/r) * G

    Args:
        r_norm (np.ndarray): Array of shape (...) representing the distance 
            between source and field points.
        G (np.ndarray): Array of shape (...) representing the Green's function.
        k (float): Wavenumber.

    Returns:
        dG_dr (np.ndarray): Array of shape (...) representing the derivative 
            of the Green's function with respect to r.
    """
    return G * (1j * k - 1 / r_norm)

def d2G_dr2(r: np.ndarray, 
            G_vals: np.ndarray, 
            k: float) -> np.ndarray:
    """
    Second radial derivative d^2G/dr^2 for 3D Helmholtz.

    For G(r) = e^{ikr}/(4π r):
        d^2G/dr^2 = (-k^2 - 2i k / r + 2 / r^2) * G

    Args:
        r (np.ndarray): Array of shape (...) representing the distance 
            between source and field points.
        G_vals (np.ndarray): Array of shape (...) representing the Green's
            function values G(r,k).
        k (float): Wavenumber.

    Returns:
        d2G_dr2: Same shape as r (complex).
    """

    eps = np.finfo(float).eps
    r_safe = np.where(r == 0, eps, r)
    return ((-k**2) - (2j * k) / r_safe + 2.0 / (r_safe**2)) * G_vals

def dG_dn_y(r_hat: np.ndarray,
            dG_dr: np.ndarray,
            n_y: np.ndarray) -> np.ndarray:
    """
    Compute the normal derivative of the Green's function with respect to
    the source point y.

    For G(r) = e^{ikr}/(4π r) and dG/dr = (ik - 1/r) G:
        ∂G/∂n_y = -dG/dr (r_hat · n_y)

    Args:
        r_hat (np.ndarray): Array of shape (..., 3) representing the unit 
            vector in the direction from y to x.
        dG_dr (np.ndarray): Array of shape (...) representing the derivative 
            of the Green's function with respect to r.
        n_y (np.ndarray): Array of shape (..., 3) representing the normal 
            vector at point y.

    Returns:
        dG_dn_y (np.ndarray): Array of shape (...) representing the normal 
            derivative of the Green's function with respect to y.
    """
    return -dG_dr * np.einsum('...i,...i->...', r_hat, n_y)

def dG_dn_x(r_hat: np.ndarray,
            dG_dr: np.ndarray,
            n_x: np.ndarray) -> np.ndarray:
    """
    Compute the normal derivative of the Green's function with respect to
    the field point x.

    For G(r) = e^{ikr}/(4π r) and dG/dr = (ik - 1/r) G:
        ∂G/∂n_x = dG/dr (r_hat · n_x)

    Args:
        r_hat (np.ndarray): Array of shape (..., 3) representing the unit 
            vector in the direction from y to x.
        dG_dr (np.ndarray): Array of shape (...) representing the derivative 
            of the Green's function with respect to r.
        n_x (np.ndarray): Array of shape (..., 3) representing the normal 
            vector at point x.

    Returns:
        dG_dn_x (np.ndarray): Array of shape (...) representing the normal 
            derivative of the Green's function with respect to x.
    """
    return dG_dr * np.einsum('...i,...i->...', r_hat, n_x)


def d2G_dn_x_dn_y(r_hat: np.ndarray,
                  r: np.ndarray,
                  n_x: np.ndarray,
                  n_y: np.ndarray,
                  G_vals: np.ndarray,
                  k: float,
                  ) -> np.ndarray:
    """
    Direct hypersingular kernel ∂²G/(∂n_x ∂n_y) for 3D Helmholtz.

    Uses the identity:
        ∂²G/∂n_x∂n_y
        = - f''(r) (r_hat·n_x)(r_hat·n_y)
          - (f'(r)/r) [ (n_x·n_y) - (r_hat·n_x)(r_hat·n_y) ],

    where f'(r) = dG/dr = (ik - 1/r) G, and
          f''(r) = d²G/dr² = (-k² - 2ik/r + 2/r²) G.

    Args:
        r_hat (np.ndarray): Array of shape (..., 3) representing the unit 
            vector in the direction from y to x.
        r (np.ndarray): Array of shape (...,) representing the distance 
            between source and field points.
        n_x (np.ndarray): Array of shape (..., 3) representing the normal 
            vector at point x.
        n_y (np.ndarray): Array of shape (..., 3) representing the normal 
            vector at point y.
        G_vals (np.ndarray): Array of shape (...,) representing the Green's
            function values G(r,k).
        k (float): Wavenumber.

    Returns:
        d2G_dn_x_dn_y (np.ndarray): Array of shape (...) representing the 
            second normal derivative of the Green's function with respect to 
            both x and y.
    """
    r = np.where(r == 0, 1e-16, r)  # Avoid division by zero
    # f'(r) and f''(r)
    dGr = dG_dr(r, G_vals, k)
    d2Gr = d2G_dr2(r, G_vals, k)

    nx_dot_ny = np.einsum("...i,...i->...", n_x, n_y)
    nx_dot_rh = np.einsum("...i,...i->...", n_x, r_hat)
    ny_dot_rh = np.einsum("...i,...i->...", n_y, r_hat)

    term1 = -d2Gr * (nx_dot_rh * ny_dot_rh)
    term2 = -(dGr / r) * (nx_dot_ny - nx_dot_rh * ny_dot_rh)

    return term1 + term2

### Half-space Green's function with impedance boundary condition

class ImpedanceGreen3D:
    """
    Half-space Green's function with impedance boundary condition.
    """
    def __init__(self, 
                 rho0: float, 
                 c0: float, 
                 Zs_fn: Callable[[float], complex],
                 nodes_per_panel: int = 3,
                 plane_normal: np.ndarray = np.array([0.0, 0.0, 1.0]),
                 plane_point: np.ndarray = np.array([0.0, 0.0, 0.0])):
        self.Z0 = rho0 * c0
        self.Zs_fn = Zs_fn
        self.nodes = nodes_per_panel

        self.plane_normal = plane_normal / np.linalg.norm(plane_normal)
        self.plane_point = plane_point

        # Orthogonal basis in the plane
        tmp = np.array([1.0, 0.0, 0.0]) if abs(self.plane_normal[0]) < 0.9 \
            else np.array([0.0, 1.0, 0.0])
        e1 = tmp - (tmp @ self.plane_normal) * self.plane_normal
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(self.plane_normal, e1)

        self.B = np.stack([e1, e2, self.plane_normal], axis=1)
        self.B_inv = np.linalg.inv(self.B)

    def _to_local_points(self, P):
        if P.ndim == 1:
            return self.B_inv @ (P - self.plane_point)
        return ((P - self.plane_point) @ self.B_inv.T)

    def _to_local_vec(self, v):
        v = np.asarray(v, float)
        return self.B_inv @ v

    def _build_qpack(self, 
                     k: float):
        """
        Build quadrature pack for Sommerfeld integral.
        Args:
            k (float): Wavenumber.
        Returns:
            None: Stores quadrature in self._qpack.
        """
        n = self.nodes
        x, w = np.polynomial.legendre.leggauss(n)
        xm = 0.5 * k * (x + 1.0)
        w1 = 0.5 * k * w 
        kt1 = xm   
        kz1 = np.sqrt(np.maximum(k*k - kt1*kt1, 0.0)) 

        xs, ws = np.polynomial.legendre.leggauss(n)
        s = 0.5 * (xs + 1.0)
        kt2 = k + s / (1.0 - s)
        jac = 1.0 / (1.0 - s)**2
        w2 = 0.5 * ws * jac   
        kz2 = 1j * np.sqrt(kt2*kt2 - k*k) 

        Zs = self.Zs_fn(k)
        Z0 = self.Z0
        cos1 = kz1 / k
        cos2 = kz2 / k
        R1 = (Zs * cos1 - Z0) / (Zs * cos1 + Z0)
        R2 = (Zs * cos2 - Z0) / (Zs * cos2 + Z0)
        C1 = R1 * kt1 / (2.0j * kz1)
        C2 = R2 * kt2 / (2.0j * kz2)
        self._qpack = {
            "k": k,
            "kt1": kt1, "w1": w1, "kz1": kz1, "C1": C1,
            "kt2": kt2, "w2": w2, "kz2": kz2, "C2": C2,
        }

    def _ensure_qpack(self, k: float):
        qp = getattr(self, "_qpack", None)
        if qp is None or qp["k"] != k:
            self._build_qpack(k)

    def sommerfeld_pressure(self, 
                            k: float, 
                            rho: np.ndarray, 
                            z_plus: np.ndarray) -> np.ndarray:
        """
        Vectorized Sommerfeld pressure. Integrates over transverse
        wavenumber kt using precomputed quadrature.

        Args:
            k (float): Wavenumber.
            rho (np.ndarray): Horizontal distance(s) (M,) or (1, M).
            z_plus (np.ndarray): Vertical sum(s) clamped to fluid side
                (M,) or (1, M).
        Returns:
            imp (np.ndarray): Sommerfeld pressure (M,) complex.
        """
        self._ensure_qpack(k)
        qp = self._qpack
        kt1, w1, kz1, C1 = qp["kt1"], qp["w1"], qp["kz1"], qp["C1"]
        kt2, w2, kz2, C2 = qp["kt2"], qp["w2"], qp["kz2"], qp["C2"]

        rho = np.asarray(rho, dtype=float).reshape(-1)
        zp  = np.asarray(z_plus, dtype=float).reshape(-1)

        J1 = sp.j0(np.outer(kt1, rho)) 
        J2 = sp.j0(np.outer(kt2, rho))

        P1 = np.exp(1j * np.outer(kz1, zp)) 
        P2 = np.exp(1j * np.outer(kz2, zp))

        I1 = (w1 * C1)[:, None] * (P1 * J1) 
        I2 = (w2 * C2)[:, None] * (P2 * J2)

        return (I1.sum(axis=0) + I2.sum(axis=0)) / (2.0 * np.pi)

    def G_hs(self, 
                 x: np.ndarray, 
                 Y: np.ndarray, 
                 k: float) -> np.ndarray:
        """
        Half-space Green's function with impedance boundary condition.

        G_hs(x,y) = G(x,y) + G_imp(x,y)
        where G_imp is the Sommerfeld integral.

        Args:
            x (np.ndarray): Field point (3,) array.
            Y (np.ndarray): Source points (M,3) array.
            k (float): Wavenumber.

        Returns:
            G_hs (np.ndarray): Half-space Green's function values (M,) complex.
        """

        r = np.linalg.norm(Y - x[None, :], axis=1)
        G0 = np.zeros_like(r, dtype=np.complex128)
        m = r > 0.0
        G0[m] = np.exp(1j * k * r[m]) / (4.0 * np.pi * r[m])

        xL = self._to_local_points(x)
        YL = self._to_local_points(Y)
        rho = np.hypot(YL[:,0]-xL[0], YL[:,1]-xL[1])
        z_plus = np.maximum(xL[2], 0.0) + np.maximum(YL[:,2], 0.0)

        imp = self.sommerfeld_pressure(k, rho, z_plus)
        return G0 + imp
    
    def dG_dn_y_hs(self, 
                       x: np.ndarray, 
                       Y: np.ndarray, 
                       ny: np.ndarray, 
                       k: float) -> np.ndarray:
        """
        Derivative ∂G_hs/∂n_y for one x (with normal ny) and many Y (Q,3).
        Args:
            x (np.ndarray): Field point (3,) array.
            Y (np.ndarray): Source points (Q,3) array.
            ny (np.ndarray): Normal at source point y (3,) array.
            k (float): Wavenumber.

        Returns:
            dG_dn_y_hs (np.ndarray): Normal derivative values (Q,) complex.
        """
        self._ensure_qpack(k); qp = self._qpack
        kt1, w1, kz1, C1 = qp["kt1"], qp["w1"], qp["kz1"], qp["C1"]
        kt2, w2, kz2, C2 = qp["kt2"], qp["w2"], qp["kz2"], qp["C2"]

        xL = self._to_local_points(x)
        YL = self._to_local_points(Y)
        nyL = self._to_local_vec(ny)

        rho = np.hypot(YL[:,0]-xL[0], YL[:,1]-xL[1]); Q = rho.size
        zp  = np.maximum(xL[2],0.0) + np.maximum(YL[:,2],0.0)

        nz = float(nyL[2]); nty = nyL[:2]
        dt = YL[:,:2] - xL[:2]
        s = np.zeros(Q); mask = rho>0
        s[mask] = (dt[mask] @ nty) / rho[mask]

        J10 = sp.j0(np.outer(kt1, rho))
        J11 = sp.j1(np.outer(kt1, rho))
        P1  = np.exp(1j * np.outer(kz1, zp))

        J20 = sp.j0(np.outer(kt2, rho))
        J21 = sp.j1(np.outer(kt2, rho))
        P2  = np.exp(1j * np.outer(kz2, zp))

        A1 = (1j * kz1)[:, None] * nz * J10
        B1 = -(kt1[:, None] * J11) * s[None, :]
        A2 = (1j * kz2)[:, None] * nz * J20
        B2 = -(kt2[:, None] * J21) * s[None, :]

        I1 = (w1 * C1)[:, None] * P1 * (A1 + B1)
        I2 = (w2 * C2)[:, None] * P2 * (A2 + B2)
        
        return (I1.sum(axis=0) + I2.sum(axis=0)) / (2.0 * np.pi)

    def dG_dn_x_hs(self, 
                   x: np.ndarray, 
                   nx: np.ndarray, 
                   Y: np.ndarray, 
                   k: float) -> np.ndarray:
        """Derivative ∂G_hs/∂n_x for one x (with normal nx) and many Y (Q,3).
        Args:
            x (np.ndarray): Field point (3,) array.
            nx (np.ndarray): Normal at field point x (3,) array.
            Y (np.ndarray): Source points (Q,3) array.
            k (float): Wavenumber.

        Returns:
            dG_dn_x_hs (np.ndarray): Normal derivative values (Q,) complex.
        """
        self._ensure_qpack(k); qp = self._qpack
        kt1, w1, kz1, C1 = qp["kt1"], qp["w1"], qp["kz1"], qp["C1"]
        kt2, w2, kz2, C2 = qp["kt2"], qp["w2"], qp["kz2"], qp["C2"]

        xL = self._to_local_points(x)
        YL = self._to_local_points(Y)
        nxL = self._to_local_vec(nx)

        rho = np.hypot(YL[:,0]-xL[0], YL[:,1]-xL[1]); Q = rho.size
        zp  = np.maximum(xL[2],0.0) + np.maximum(YL[:,2],0.0)

        nz = float(nxL[2]); ntx = nxL[:2]
        dt = xL[:2] - YL[:,:2]
        s = np.zeros(Q); mask = rho>0
        s[mask] = (dt[mask] @ ntx) / rho[mask]

        J10 = sp.j0(np.outer(kt1, rho))
        J11 = sp.j1(np.outer(kt1, rho))
        P1 = np.exp(1j * np.outer(kz1, zp))

        J20 = sp.j0(np.outer(kt2, rho))
        J21 = sp.j1(np.outer(kt2, rho))
        P2 = np.exp(1j * np.outer(kz2, zp))

        A1 = (1j * kz1)[:, None] * nz * J10
        B1 = -(kt1[:, None] * J11) * s[None, :]
        A2 = (1j * kz2)[:, None] * nz * J20
        B2 = -(kt2[:, None] * J21) * s[None, :]

        I1 = (w1 * C1)[:, None] * P1 * (A1 + B1)
        I2 = (w2 * C2)[:, None] * P2 * (A2 + B2)
        return (I1.sum(axis=0) + I2.sum(axis=0)) / (2.0 * np.pi)

    def grad_y_imp(self, 
                   x: np.ndarray, 
                   Y: np.ndarray, 
                   k: float) -> np.ndarray:
        """
        Compute the gradient with respect to y of the impedance.
        Args:
            x (np.ndarray): Field point (3,) array.
            Y (np.ndarray): Source points (Q,3) array.
            k (float): Wavenumber.

        Returns:
            grad_y_imp (np.ndarray): Gradient values (Q,3) complex.
        """
        self._ensure_qpack(k); qp = self._qpack
        kt1, w1, kz1, C1 = qp["kt1"], qp["w1"], qp["kz1"], qp["C1"]
        kt2, w2, kz2, C2 = qp["kt2"], qp["w2"], qp["kz2"], qp["C2"]

        xL = self._to_local_points(x)
        YL = self._to_local_points(Y)

        dt = YL[:,:2] - xL[:2]
        rho = np.hypot(dt[:,0], dt[:,1]); Q = rho.size
        zp  = np.maximum(xL[2],0.0) + np.maximum(YL[:,2],0.0)

        uh = np.zeros((Q,2)); m = rho>0; uh[m] = dt[m]/rho[m,None]

        J10 = sp.j0(np.outer(kt1, rho))
        J11 = sp.j1(np.outer(kt1, rho))
        P1 = np.exp(1j * np.outer(kz1, zp))

        J20 = sp.j0(np.outer(kt2, rho))
        J21 = sp.j1(np.outer(kt2, rho))
        P2 = np.exp(1j * np.outer(kz2, zp))

        T1 = (w1 * C1)[:, None] * P1 * (-(kt1[:, None] * J11))
        T2 = (w2 * C2)[:, None] * P2 * (-(kt2[:, None] * J21))
        t_sum = (T1.sum(axis=0) + T2.sum(axis=0)) / (2.0 * np.pi)

        Z1 = (w1 * C1)[:, None] * P1 * (1j * kz1[:, None] * J10)
        Z2 = (w2 * C2)[:, None] * P2 * (1j * kz2[:, None] * J20)
        z_sum = (Z1.sum(axis=0) + Z2.sum(axis=0)) / (2.0 * np.pi)

        out = np.zeros((Q, 3), dtype=np.complex128)
        out[:, 0:2] = (t_sum[:, None] * uh) 
        out[:, 2]   = z_sum
        return out

    def grad_x_imp(self, 
                   x: np.ndarray, 
                   Y: np.ndarray, 
                   k: float) -> np.ndarray:
        """
        Compute the gradient with respect to x of the impedance.
        Args:
            x (np.ndarray): Field point (3,) array.
            Y (np.ndarray): Source points (Q,3) array.
            k (float): Wavenumber.

        Returns:
            grad_x_imp (np.ndarray): Gradient values (Q,3) complex.
        """
        self._ensure_qpack(k); qp = self._qpack
        kt1, w1, kz1, C1 = qp["kt1"], qp["w1"], qp["kz1"], qp["C1"]
        kt2, w2, kz2, C2 = qp["kt2"], qp["w2"], qp["kz2"], qp["C2"]

        xL = self._to_local_points(x)
        YL = self._to_local_points(Y)

        dt = xL[:2] - YL[:,:2]
        rho = np.hypot(dt[:,0], dt[:,1]); Q = rho.size
        zp  = np.maximum(xL[2],0.0) + np.maximum(YL[:,2],0.0)

        uh = np.zeros((Q,2)); m = rho>0; uh[m] = dt[m]/rho[m,None]

        J10 = sp.j0(np.outer(kt1, rho))
        J11 = sp.j1(np.outer(kt1, rho))
        P1 = np.exp(1j * np.outer(kz1, zp))

        J20 = sp.j0(np.outer(kt2, rho))
        J21 = sp.j1(np.outer(kt2, rho))
        P2 = np.exp(1j * np.outer(kz2, zp))

        T1 = (w1 * C1)[:, None] * P1 * (-(kt1[:, None] * J11))
        T2 = (w2 * C2)[:, None] * P2 * (-(kt2[:, None] * J21))
        t_sum = (T1.sum(axis=0) + T2.sum(axis=0)) / (2.0 * np.pi)

        Z1 = (w1 * C1)[:, None] * P1 * (1j * kz1[:, None] * J10)
        Z2 = (w2 * C2)[:, None] * P2 * (1j * kz2[:, None] * J20)
        z_sum = (Z1.sum(axis=0) + Z2.sum(axis=0)) / (2.0 * np.pi)

        out = np.zeros((Q, 3), dtype=np.complex128)
        out[:, 0:2] = (t_sum[:, None] * uh)
        out[:, 2]   = z_sum
        return out
    
    def d2G_dnxdny_imp(self,
                       x: np.ndarray, 
                       nx: np.ndarray,
                       Y: np.ndarray, 
                       ny: np.ndarray,
                       k: float) -> np.ndarray:
        """
        Compute the hypersingular impedance kernel ∂²G_imp/(∂n_x ∂n_y).
        Args:
            x (np.ndarray): Field point (3,) array.
            nx (np.ndarray): Normal at field point x (3,) array.
            Y (np.ndarray): Source points (Q,3) array.
            ny (np.ndarray): Normal at source point y (3,) array.
            k (float): Wavenumber.

        Returns:
            d2G_dnxdny_imp (np.ndarray): Hypersingular values (Q,) complex.
        """
        self._ensure_qpack(k); qp = self._qpack
        kt1, w1, kz1, C1 = qp["kt1"], qp["w1"], qp["kz1"], qp["C1"]
        kt2, w2, kz2, C2 = qp["kt2"], qp["w2"], qp["kz2"], qp["C2"]

        xL = self._to_local_points(x)
        YL = self._to_local_points(Y)
        nxL = self._to_local_vec(nx)
        nyL = self._to_local_vec(ny)

        dt = YL[:,:2] - xL[:2]
        rho = np.hypot(dt[:,0], dt[:,1]); Q = rho.size
        zp  = np.maximum(xL[2],0.0) + np.maximum(YL[:,2],0.0)

        nxy = nxL[:2]; nyy = nyL[:2]
        s_y = np.zeros(Q); s_x = np.zeros(Q)
        m = rho>0
        s_y[m] = (dt[m] @ nyy) / rho[m]
        s_x[m] = ((-dt[m]) @ nxy) / rho[m]

        dot_t = float(nxy @ nyy)
        ds_y_dx = np.zeros(Q); ds_y_dx[m] = (-(dot_t) - s_y[m]*s_x[m]) / rho[m]

        if np.ndim(nyy) == 1:
            dot_t = float(nxy @ nyy)
        else:
            dot_t = np.einsum('i,i->', nxy, nyy)
        ds_y_dx[m] = (-(dot_t) - s_y[m] * s_x[m]) / rho[m]

        r1 = np.outer(kt1, rho); r2 = np.outer(kt2, rho)
        J10 = sp.j0(r1)
        J20 = sp.j0(r2)
        J11 = sp.j1(r1)
        J21 = sp.j1(r2)
        dJ11 = sp.jvp(1, r1, 1)
        dJ21 = sp.jvp(1, r2, 1)
        P1 = np.exp(1j * np.outer(kz1, zp))
        P2 = np.exp(1j * np.outer(kz2, zp))

        B1   = J10
        By1  = -(kt1[:, None] * J11) * s_y[None, :]
        Bx1  = -(kt1[:, None] * J11) * s_x[None, :]
        Bxy1 = - ( (kt1[:, None]**2) * dJ11 * (s_x[None, :] * s_y[None, :]) + \
                  (kt1[:, None] * J11) * ds_y_dx[None, :] )

        B2   = J20
        By2  = -(kt2[:, None] * J21) * s_y[None, :]
        Bx2  = -(kt2[:, None] * J21) * s_x[None, :]
        Bxy2 = - ( (kt2[:, None]**2) * dJ21 * (s_x[None, :] * s_y[None, :]) + \
                  (kt2[:, None] * J21) * ds_y_dx[None, :] )

        nxz = float(nx[2]); nyz = float(ny[2])

        A1  = P1
        Ax1 = (1j * kz1)[:, None] * nxz * P1
        Ay1 = (1j * kz1)[:, None] * nyz * P1
        A2  = P2
        Ax2 = (1j * kz2)[:, None] * nxz * P2
        Ay2 = (1j * kz2)[:, None] * nyz * P2

        Axy1 = (-(kz1**2))[:, None] * (nxz * nyz) * P1
        Axy2 = (-(kz2**2))[:, None] * (nxz * nyz) * P2

        I1 = (w1 * C1)[:, None] * ( Axy1*B1 + Ax1*By1 + Ay1*Bx1 + A1*Bxy1 )
        I2 = (w2 * C2)[:, None] * ( Axy2*B2 + Ax2*By2 + Ay2*Bx2 + A2*Bxy2 )

        return (I1.sum(axis=0) + I2.sum(axis=0)) / (2.0 * np.pi)

    def _R(self, 
           kt: float, 
           k: float) -> complex:
        """
        Compute the reflection coefficient R(kt,k) for the impedance
        boundary condition.

        R(kt,k) = (Zs * cos(theta) - Z0) / (Zs * cos(theta) + Z0)
        where cos(theta) = kz/k, and
            kz = sqrt(k^2 - kt^2) for kt <= k (propagating)
            kz = i sqrt(kt^2 - k^2) for kt > k (evanescent)

        Args:
            kt (float): Transverse wavenumber.
            k (float): Wavenumber.

        Returns:
            R (complex): Reflection coefficient.
        """
        if kt <= k:
            kz = np.sqrt(max(k*k - kt*kt, 0.0))
        else:
            kz = 1j * np.sqrt(kt*kt - k*k)
        Zs = self.Zs_fn(k)
        cos_th = kz / k
        return (Zs * cos_th - self.Z0) / (Zs * cos_th + self.Z0)

    def _integrate_finite(self, 
                          f, 
                          a: float, 
                          b: float) -> complex:
        """
        Calculate the integral of f from a to b using adaptive
        Gauss-Legendre quadrature.

        Args:
            f (Callable[[float], complex]): Function to integrate.
            a (float): Lower limit.
            b (float): Upper limit.
        Returns:
            integral (complex): Integral value.
        """
        x, w = np.polynomial.legendre.leggauss(self.nodes)
        xm = 0.5*(a+b); xr = 0.5*(b-a)
        vals = np.vectorize(f, otypes=[np.complex128])(xm + xr*x)
        return xr * np.sum(w * vals)

    def _integrate_semi_inf(self, 
                            f, 
                            a: float) -> complex:
        """
        Integrate f from a to +inf using a change of variables.
        Uses the transformation s = (t - a)/(t - a + 1), which maps
        t in [a, +inf) to s in [0, 1). The Jacobian is dt/ds = 1/(1-s)^2.

        Args:
            f (Callable[[float], complex]): Function to integrate.
            a (float): Lower limit.
        Returns:
            integral (complex): Integral value.
        """
        g = lambda s: f(a + s/(1.0 - s)) / (1.0 - s)**2
        return self._integrate_finite(g, 0.0, 1.0)

    def integrand(self, 
                  kt: float,
                  k: float,
                  rho: float, 
                  z_plus: float) -> complex:
        """
        Compute the integrand for the Sommerfeld pressure integral.
        f(kt) = [R(kt,k) * exp(ikz z_plus) * J0(kt rho) * kt] / (2ikz)
        where kz = sqrt(k^2 - kt^2) for kt <= k (propagating)
              kz = i sqrt(kt^2 - k^2) for kt > k (evanescent)
        J0(0) = 1
        Args:
            kt (float): Transverse wavenumber.
            k (float): Wavenumber.
            rho (float): Horizontal distance.
            z_plus (float): Vertical sum clamped to fluid side.

        Returns:
            f(kt) (complex): Integrand value.
        """
        J0 = 1.0 if (kt == 0.0 and rho == 0.0) else sp.j0(kt * rho)
        kz = np.sqrt(max(k*k - kt*kt, 0.0)) if kt <= k \
            else 1j*np.sqrt(kt*kt - k*k)
        return (self._R(kt, k) * np.exp(1j * kz * z_plus) * J0 * kt) \
            / (2.0j * kz)