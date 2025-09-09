import numpy as np

def map_to_physical_triangle(xi_eta: np.ndarray,
                             vertex_coords: np.ndarray,
                             ) -> tuple[np.ndarray, float]:
    
    """
    Map (xi,eta) in the standard reference triangle (0<=xi,eta, xi+eta<=1)
    to physical coordinates y = v0 + xi*e1 + eta*e2.

    Args:
        xi_eta (np.ndarray): Array of shape (N, 2) representing the
            quadrature points in barycentric coordinates.
        vertex_coords (np.ndarray): Array of shape (3, 3) representing the 
            coordinates of the triangle vertices.
    
    Returns:
        y_phys (np.ndarray): Array of shape (N, 3) representing the
            quadrature points in physical coordinates.
        a2 (float): Jacobian scale (||e1×e2||), i.e. twice the *physical* 
            triangle area.
    """

    v0, v1, v2 = vertex_coords
    e1 = v1 - v0
    e2 = v2 - v0
    y_phys = v0[None, :] + \
            xi_eta[:, [0]] * e1[None, :] + \
            xi_eta[:, [1]] * e2[None, :]
    a2 = np.linalg.norm(np.cross(e1, e2))
    return y_phys, a2

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
    xi = xi_eta[:, 0]
    eta = xi_eta[:, 1]
    N = np.column_stack([1.0 - xi - eta, xi, eta])
    return N

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
    
    return quad_points, quad_weights

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

    U, V = np.meshgrid(u, v, indexing='ij')
    W = np.outer(wu, wv)

    XI  = U * (1.0 - V)
    ETA = U * V
    quad_weights = (W * U).ravel() 

    quad_points = np.stack([XI.ravel(), ETA.ravel()], axis=1)
    if sing_vert_int != 0:
        quad_points = permute_to_vertex(quad_points, sing_vert_int) 
    return quad_points, quad_weights

def telles_rule(u_star: float,
                v_star: float | None = None,
                sing_vert_int: int = 0,
                n_leg: int = 8,
                s0: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate quadrature points and weights for a triangle using the Telles 
    transformation from a square.

    'The transformation is a cubic polynomial that maps the Gauss-Legendre 
    points in [0, 1] onto itself, and has zero Jacobian at the singularity
    location u_star. The parameter s0 controls the clustering of points near
    the singularity.'

    Args:
        u_star (float): u-coordinate of the singularity in the reference
            triangle (0 <= u_star <= 1).
        v_star (float | None): v-coordinate of the singularity in the reference
            triangle (0 <= v_star <= 1, u_star + v_star <= 1).
        sing_vert_int (int, optional): Vertex index (0, 1, or 2) where the
            singularity is located. Default is 0.
        n_leg (int, optional): Number of Gauss-Legendre points along one edge
            of the square. Default is 8.
        s0 (float, optional): Reference GL point for clustering. Default is 
            0.5.

    Returns:
        quad_points (np.ndarray): Array of shape (N, 2) representing the
            quadrature points in barycentric coordinates.
        quad_weights (np.ndarray): Array of shape (N,) representing the
            quadrature weights.
    """

    u_nodes, w_u = gauss_legendre_1d(n_leg)
    v_nodes, w_v = gauss_legendre_1d(n_leg)

    u_mapped, du_mapped_du = telles_cubic_1d(u_nodes, u_star, s0)
    if v_star is None:
        v_mapped, dv_mapped_dv = v_nodes, np.ones_like(v_nodes)
    else:
        v_mapped, dv_mapped_dv = telles_cubic_1d(v_nodes, v_star, s0)

    U, V = np.meshgrid(u_mapped, v_mapped, indexing='ij')
    dU, dV = np.meshgrid(du_mapped_du, dv_mapped_dv, indexing='ij')
    W_u, W_v = np.meshgrid(w_u, w_v, indexing='ij')

    XI = U * (1.0 - V)
    ETA = U * V

    quad_points = np.column_stack([XI.ravel(), ETA.ravel()])
    quad_weights = (W_u * W_v * dU * dV * U).ravel()

    if sing_vert_int != 0:
        quad_points = permute_to_vertex(quad_points, sing_vert_int)

    return quad_points, quad_weights

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