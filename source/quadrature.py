import warnings
import numpy as np

def map_to_physical_triangle(xi_eta: np.ndarray,
                             v0: np.ndarray,
                             e1: np.ndarray,
                             e2: np.ndarray
                             ) -> tuple[np.ndarray, float]:
    
    """
    Map (xi,eta) in the standard reference triangle (0<=xi,eta, xi+eta<=1)
    to physical coordinates y = v0 + xi*e1 + eta*e2.

    Args:
        xi_eta (np.ndarray): Array of shape (N, 2) representing the
            quadrature points in barycentric coordinates.
        v0 (np.ndarray): Array of shape (3,) representing the first
            vertex of the triangle.
        e1 (np.ndarray): Array of shape (3,) representing the edge
            vector from v0 to v1.
        e2 (np.ndarray): Array of shape (3,) representing the edge
            vector from v0 to v2.
    
    Returns:
        y_phys (np.ndarray): Array of shape (N, 3) representing the
            quadrature points in physical coordinates.
        a2 (float): Jacobian scale (||e1×e2||), i.e. twice the *physical* 
            triangle area.
    """
    y_phys = v0[None, :] + \
            xi_eta[:, [0]] * e1[None, :] + \
            xi_eta[:, [1]] * e2[None, :]
    a2 = np.linalg.norm(np.cross(e1, e2))
    return y_phys, a2

def map_to_physical_triangle_batch(xi_eta: np.ndarray,
                                   v0: np.ndarray,
                                   e1: np.ndarray,
                                   e2: np.ndarray) -> tuple[np.ndarray, 
                                                            np.ndarray]:
    """
    Vectorized mapping for K triangles at once.

    Args:
        xi_eta (np.ndarray): Array of shape (Q, 2) representing the
            quadrature points in barycentric coordinates.
        v0 (np.ndarray): Array of shape (K, 3) representing the first
            vertex of each triangle.
        e1 (np.ndarray): Array of shape (K, 3) representing the edge
            vector from v0 to v1 for each triangle.
        e2 (np.ndarray): Array of shape (K, 3) representing the edge
            vector from v0 to v2 for each triangle.

    Returns:
        y (np.ndarray): Array of shape (K, Q, 3) representing the
            quadrature points in physical coordinates.
        a2 (np.ndarray): Array of shape (K,) representing the Jacobian
            scale (||e1×e2||), i.e. twice the *physical* triangle area.
    """
    xi = xi_eta[:, 0][None, :]
    eta = xi_eta[:, 1][None, :]
    y = v0[:, None, :] + \
        xi[..., None]*e1[:, None, :] + \
        eta[..., None]*e2[:, None, :]
    a2 = np.linalg.norm(np.cross(e1, e2), axis=1)
    return y, a2

def shape_functions_P1(xi_eta: np.ndarray) -> np.ndarray:
    """
    Compute the P1 (linear) shape functions at given barycentric coordinates.

    Args:
        xi_eta (np.ndarray): Array of shape (N, 2) representing the 
            barycentric coordinates (xi, eta) in the reference triangle.

    Returns:
        N (np.ndarray): Array of shape (N, 3) representing the values of 
            the three P1 shape functions at the given points.
    """
    N = np.empty((xi_eta.shape[0], 3), dtype=xi_eta.dtype)
    N[:,1] = xi_eta[:,0]
    N[:,2] = xi_eta[:,1]
    N[:,0] = 1.0 - N[:,1] - N[:,2]
    return N

def shape_function_gradients_P1() -> np.ndarray:
    """
    Compute the gradients of the P1 (linear) shape functions in the 
    reference triangle.

    Returns:
        dN_dxi (np.ndarray): Array of shape (3, 2) representing the 
            gradients of the three P1 shape functions with respect to 
            (xi, eta).
    """
    dN_dxi = np.array([[-1.0, -1.0],
                       [ 1.0,  0.0],
                       [ 0.0,  1.0]])
    return dN_dxi

def standard_triangle_quad(order: int = 1,
                           ) -> tuple[np.ndarray, np.ndarray]:
    
    """
    Generate quadrature points and weights for a standard triangle.
    The standard triangle has vertices at (0,0), (1,0), and (0,1).

    Args:
        order (int, optional): Order of the quadrature. Supported orders are 
            1, 3, and 7. Default is 1.


    Returns:
        quad_points (np.ndarray): Array of shape (N, 2) representing the 
            quadrature points in barycentric coordinates.
        quad_weights (np.ndarray): Array of shape (N,) representing the 
            quadrature weights.
    """
    
    if order == 1:
        quad_points = np.array([[1/3, 1/3]])
        quad_weights = np.array([0.5])

    elif order == 3:
        quad_points = np.array([[1/6, 1/6],
                                [2/3, 1/6],
                                [1/6, 2/3]])
        quad_weights = np.array([1/6, 1/6, 1/6])

    elif order == 7:
        quad_points = np.array([[1/3, 1/3],
                                [0.0597158717, 0.4701420641],
                                [0.4701420641, 0.0597158717],
                                [0.4701420641, 0.4701420641],
                                [0.7974269853, 0.1012865073],
                                [0.1012865073, 0.7974269853],
                                [0.1012865073, 0.1012865073]])
        quad_weights = np.array([0.225,
                                 0.1323941527,
                                 0.1323941527,
                                 0.1323941527,
                                 0.1259391805,
                                 0.1259391805,
                                 0.1259391805]) * 0.5
        
    else:
        raise ValueError("Unsupported quadrature order. Supported orders are "
                         "1, 3, and 7.")
    
    pts = np.asarray(quad_points, dtype=np.float64, order='C')
    w = np.asarray(quad_weights, dtype=np.float64, order='C')

    return pts, w

def refined_triangle_quad(xi_eta: np.ndarray,
                          weights: np.ndarray,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """
    Refine a single triangle into four smaller triangles and adjust the
    quadrature points and weights accordingly.

    Args:
        xi_eta (np.ndarray): Array of shape (N, 2) representing the
            quadrature points in barycentric coordinates.
        weights (np.ndarray): Array of shape (N,) representing the
            quadrature weights.

    Returns:
        xi_eta_ref (np.ndarray): Array of shape (4N, 2) representing the
            refined quadrature points in barycentric coordinates.
        w_ref (np.ndarray): Array of shape (4N,) representing the
            refined quadrature weights.
    """
    X = xi_eta
    xi  = X[:, 0]
    eta = X[:, 1]
    Wq = weights * 0.25

    X1 = np.column_stack((      0.5 * xi,            0.5 * eta))
    X2 = np.column_stack((0.5 + 0.5 * xi,            0.5 * eta))
    X3 = np.column_stack((      0.5 * xi,      0.5 + 0.5 * eta))
    X4 = np.column_stack((0.5 - 0.5 * xi, 0.5 * xi + 0.5 * eta))

    xi_eta_ref = np.vstack((X1, X2, X3, X4))
    w_ref      = np.concatenate((Wq, Wq, Wq, Wq))

    return xi_eta_ref, w_ref

def subdivide_triangle_quad(xi_eta: np.ndarray,
                            weights: np.ndarray,
                            levels: int = 1,
                            ) -> tuple[np.ndarray, np.ndarray]:
    """
    Recursively refine a triangle into smaller triangles and adjust the
    quadrature points and weights accordingly.

    Args:
        xi_eta (np.ndarray): Array of shape (N, 2) representing the
            quadrature points in barycentric coordinates.
        weights (np.ndarray): Array of shape (N,) representing the
            quadrature weights.
        levels (int, optional): Number of refinement levels. Each level
            subdivides each triangle into four smaller triangles. Default is 1.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Refined quadrature points and weights.
    """
    pts = np.asarray(xi_eta, dtype=float)
    w   = np.asarray(weights, dtype=float)

    if levels < 1:
        warnings.warn("Number of refinement levels must be at least 1. "
                      "Returning original points and weights.")
        return pts, w

    for _ in range(levels):
        pts, w = refined_triangle_quad(pts, w)

    pts = np.asarray(pts, dtype=np.float64, order='C')
    w   = np.asarray(w, dtype=np.float64, order='C')

    return pts, w

def duffy_rule(n_leg: int = 8,
               sing_vert_int: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate quadrature points and weights for a triangle using the Duffy 
    transformation from a square.

    Args:
        n_leg (int, optional): Number of Gauss-Legendre points along one edge 
            of the square. The total number of quadrature points will be 
            n_leg*(n_leg+1)/2. Default is 8.
        sing_vert_int (int, optional): Vertex index (0, 1, or 2) where the
            singularity is located. Default is 0.

    Returns:
        quad_points (np.ndarray): Array of shape (N, 2) representing the 
            quadrature points in barycentric coordinates.
        quad_weights (np.ndarray): Array of shape (N,) representing the 
            quadrature weights.
    """
    
    u, wu = gauss_legendre_1d(n_leg)
    v, wv = gauss_legendre_1d(n_leg)

    # vectorized without meshgrid
    XI  = np.multiply.outer(u, (1.0 - v))
    ETA = np.multiply.outer(u, v)
    w   = (np.multiply.outer(wu, wv) * u[:, None]).ravel()

    pts = np.stack([XI.ravel(), ETA.ravel()], axis=1)
    if sing_vert_int != 0:
        pts = permute_to_vertex(pts, sing_vert_int)

    pts = np.asarray(pts, dtype=np.float64, order='C')
    w   = np.asarray(w, dtype=np.float64, order='C')

    return pts, w

def telles_rule(u_star: float,
                v_star: float | None = None,
                sing_vert_int: int = 0,
                n_leg: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate quadrature points and weights for a triangle using the Telles 
    transformation from a square.

    'The transformation is a cubic polynomial that maps the Gauss-Legendre 
    points in [0, 1] onto itself, and has zero Jacobian at the singularity
    location u_star. The parameter s0 is a reference GL point that controls
    the clustering of points around the singularity.'

    Args:
        u_star (float): u-coordinate of the singularity in the reference
            triangle (0 <= u_star <= 1).
        v_star (float | None): v-coordinate of the singularity in the reference
            triangle (0 <= v_star <= 1, u_star + v_star <= 1).
        sing_vert_int (int, optional): Vertex index (0, 1, or 2) where the
            singularity is located. Default is 0.
        n_leg (int, optional): Number of Gauss-Legendre points along one edge
            of the square. Default is 8.

    Returns:
        quad_points (np.ndarray): Array of shape (N, 2) representing the
            quadrature points in barycentric coordinates.
        quad_weights (np.ndarray): Array of shape (N,) representing the
            quadrature weights.
    """

    u_nodes, wu = gauss_legendre_1d(n_leg)
    v_nodes, wv = gauss_legendre_1d(n_leg)

    s0_u = np.clip(u_star, 0.1, 0.9)        

    u_map, du = telles_cubic_1d(u_nodes, u_star, s0_u)
    if v_star is not None:
        s0_v = np.clip(v_star, 0.1, 0.9)
        v_map, dv = telles_cubic_1d(v_nodes, v_star, s0_v)
    else:
        v_map, dv = v_nodes, np.ones_like(v_nodes)

    XI  = np.multiply.outer(u_map, (1.0 - v_map))
    ETA = np.multiply.outer(u_map, v_map)
    w   = (np.multiply.outer(wu, wv) * \
           np.multiply.outer(du, dv) * u_map[:, None]).ravel()

    pts = np.stack([XI.ravel(), ETA.ravel()], axis=1)
    if sing_vert_int != 0:
        pts = permute_to_vertex(pts, sing_vert_int)

    pts = np.asarray(pts, dtype=np.float64, order='C')
    w   = np.asarray(w, dtype=np.float64, order='C')
    
    return pts, w

def gauss_legendre_1d(n:int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Gauss-Legendre quadrature points and weights on the interval
    [0, 1] and weights with correct scaling (sum(weights) = 1).

    Args:
        n (int): Number of quadrature points.

    Returns:
        points (np.ndarray): Array of shape (n,) representing the quadrature 
            points.
        weights (np.ndarray): Array of shape (n,) representing the quadrature 
            weights.
    """
    
    if n < 1:
        raise ValueError("Number of quadrature points must be at least 1.")
    
    points, weights = np.polynomial.legendre.leggauss(n)
    points = 0.5 * (points + 1.0)
    weights = 0.5 * weights
    return points, weights

def permute_to_vertex(xi_eta: np.ndarray,
                      sing_vert_int: int) -> np.ndarray:
    """
    Permute the barycentric coordinates (xi, eta) to place the singularity
    at the specified vertex.

    Args:
        xi_eta (np.ndarray): Array of shape (N, 2) representing the 
            barycentric coordinates (xi, eta).
        sing_vert_int (int): Vertex index (0, 1, or 2) where the singularity 
            is located.

    Returns:
        xi_eta_perm (np.ndarray): Array of shape (N, 2) representing the 
            permuted barycentric coordinates.
    """
    
    if sing_vert_int == 0:
        return xi_eta
    elif sing_vert_int == 1:
        xi = xi_eta[:, 0]
        eta = xi_eta[:, 1]
        xi_perm = eta
        eta_perm = 1.0 - xi - eta
        return np.column_stack([xi_perm, eta_perm])
    elif sing_vert_int == 2:
        xi = xi_eta[:, 0]
        eta = xi_eta[:, 1]
        xi_perm = 1.0 - xi - eta
        eta_perm = xi
        return np.column_stack([xi_perm, eta_perm])
    else:
        raise ValueError("sing_vert_int must be 0, 1, or 2.")
    
def telles_cubic_1d(u: np.ndarray,
                    t0: float,
                    s0: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the Telles cubic transformation to the 1D Gauss-Legendre points.
    Fallback to original points if the transformation matrix is singular.

    Args:
        u (np.ndarray): Array of shape (N,) representing the Gauss-Legendre 
            points in [0, 1].
        t0 (float): Location of the singularity in [0, 1].
        s0 (float, optional): Reference GL point for clustering. Default is 
            0.5.

    Returns:
        u_telles (np.ndarray): Array of shape (N,) representing the 
            transformed points.
        du_telles_du (np.ndarray): Array of shape (N,) representing the 
            derivative of the transformation with respect to u.
    """

    M = np.array([[1.0, 1.0, 1.0],
                  [s0**3, s0**2, s0],
                  [3* s0**2, 2*s0, 1.0]])
    
    rhs = np.array([1.0, t0, 0.0])

    try:
        a, b, c = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        return u, np.ones_like(u)

    u_telles = a * u**3 + b * u**2 + c * u
    du_telles_du = 3 * a * u**2 + 2 * b * u + c

    if np.any(du_telles_du <= 0):
        return u.copy(), np.ones_like(u)

    return u_telles, du_telles_du