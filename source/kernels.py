import numpy as np

def r_vec(x: np.ndarray, 
          y: np.ndarray) -> np.ndarray:
    """
    Compute the vector from points y to points x.

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

    Args:
        r_norm (np.ndarray): Array of shape (...) representing the distance 
            between source and field points.
        G (np.ndarray): Array of shape (...) representing the Green's function.
        k (float): Wavenumber.

    Returns:
        dG_dr (np.ndarray): Array of shape (...) representing the derivative 
            of the Green's function with respect to r.
    """
    return G * (1j * k - 1 / r_norm) / r_norm

def dG_dn_y(r_hat: np.ndarray,
            dG_dr: np.ndarray,
            n_y: np.ndarray) -> np.ndarray:
    """
    Compute the normal derivative of the Green's function with respect to
    the source point y.

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
                 dG_dr: np.ndarray,
                 n_x: np.ndarray,
                 n_y: np.ndarray) -> np.ndarray:
    """
    Compute the second normal derivative of the Green's function with respect
    to both the field point x and the source point y.

    Args:
        r_hat (np.ndarray): Array of shape (..., 3) representing the unit 
            vector in the direction from y to x.
        dG_dr (np.ndarray): Array of shape (...) representing the derivative 
            of the Green's function with respect to r.
        n_x (np.ndarray): Array of shape (..., 3) representing the normal 
            vector at point x.
        n_y (np.ndarray): Array of shape (..., 3) representing the normal 
            vector at point y.

    Returns:
        d2G_dn_x_dn_y (np.ndarray): Array of shape (...) representing the 
            second normal derivative of the Green's function with respect to 
            both x and y.
    """
    term1 = (1 / r_hat.shape[-1]) * np.einsum('...i,...i->...', n_x, n_y)
    term2 = np.einsum('...i,...i->...', r_hat, n_x) * \
        np.einsum('...i,...i->...', r_hat, n_y)
    return dG_dr * (term1 - term2)