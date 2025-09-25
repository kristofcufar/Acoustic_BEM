import numpy as np

class Mesh:
    def __init__(self,
                 mesh_nodes: np.ndarray,
                 mesh_elements: np.ndarray,
                 Neumann_BC: np.ndarray | None,
                 Dirichlet_BC: np.ndarray | None,
                 frequency: float,
                 c0: float = 343.0,
                 rho0: float = 1.225,):
        
        """
        Initialize the Core class for acoustic boundary element method. 
        Initializes:
            - mesh_nodes: Array of shape (N, 3) representing the coordinates 
                of the mesh nodes.
            - mesh_elements: Array of shape (M, 3) representing the 
                connectivity of the mesh elements.
            - Neumann_BC: Array of shape (N,) or (N, 3) representing the Neumann 
                boundary conditions at the mesh nodes. If shape is (N, 3), it is
                projected onto the mesh normals.
            - Dirichlet_BC: Array of shape (N,) representing the Dirichlet 
                boundary conditions at the mesh nodes.
            - frequency: Frequency of the structure vibration.
            - c0: Speed of sound in m/s. Default is 343.0 m/s.
            - rho0: Density of the medium in kg/m^3. Default is 1.225 kg/m^3.

        Args:
            mesh_nodes (np.ndarray): Array of shape (N, 3) representing the 
                coordinates of the mesh nodes.
            mesh_elements (np.ndarray): Array of shape (M, 3) representing the 
                connectivity of the mesh elements.
            velocity_BC (np.ndarray): Array of shape (N,) representing the 
                velocity boundary conditions at the mesh nodes.
            frequency (float): Frequency of the acoustic wave in Hz.
            c0 (float, optional): Speed of sound in m/s. Default is 343.0 m/s.
            rho0 (float, optional): Density of the medium in kg/m^3. Default 
                is 1.225 kg/m^3.
        """
        
        self.mesh_nodes = mesh_nodes
        self.mesh_elements = mesh_elements

        if Dirichlet_BC is None and Neumann_BC is not None:
            self.Neumann_BC = Neumann_BC
            self.Dirichlet_BC = None
        elif Neumann_BC is None and Dirichlet_BC is not None:
            self.Dirichlet_BC = Dirichlet_BC
            self.Neumann_BC = None
        else:
            raise ValueError("Either Neumann_BC or Dirichlet_BC must be " \
            "provided, not both.")
        
        self.frequency = frequency
        self.c0 = c0
        self.rho0 = rho0
        self.num_nodes = mesh_nodes.shape[0]
        self.num_elements = mesh_elements.shape[0]

        self.omega = 2 * np.pi * frequency
        self.k = self.omega / c0

        self.precompute_elements()
        self.node_normals()
        self.get_characteristic_length()
        self.compute_jump_coefficients()
        if Neumann_BC is not None:
            self.project_velocity_bc()

    def precompute_elements(self):
        """
        Precompute geometric properties of the mesh elements. Initializes:
            - v0: First vertex of each triangle.
            - e1: Edge vector from v0 to v1.
            - e2: Edge vector from v0 to v2.
            - a2: Twice the area of each triangle (||e1×e2|| Jacobian).
            - n_hat: Unit normal vector of each triangle.
            - centroids: Centroid of each triangle.
            - areas: Area of each triangle.
            - node_normals: Area-weighted normals at each mesh node.
            - char_length: Characteristic length of the mesh.
        """
        v0 = self.mesh_nodes[self.mesh_elements[:, 0], :]
        v1 = self.mesh_nodes[self.mesh_elements[:, 1], :]
        v2 = self.mesh_nodes[self.mesh_elements[:, 2], :]
        e1 = v1 - v0
        e2 = v2 - v0
        cross = np.cross(e1, e2)
        a2 = np.linalg.norm(cross, axis=1)
        n_hat = cross / (a2[:, np.newaxis] + 1e-300)
        centroids = (v0 + v1 + v2) / 3.0
        areas = 0.5 * a2
        node_in_el = [self.node_in_element(i) for i in range(self.num_nodes)]

        self.v0 = v0
        self.e1 = e1
        self.e2 = e2
        self.a2 = a2
        self.n_hat = n_hat
        self.centroids = centroids
        self.areas = areas
        self.node_in_el = node_in_el

    def node_normals(self) -> np.ndarray:
        """
        Compute area-weighted normals at each mesh node.

        Returns:
            node_normals (np.ndarray): Array of shape (N, 3) representing the
                area-weighted normals at each mesh node.
        """
        node_normals = np.zeros((self.num_nodes, 3))
        for elem in range(self.num_elements):
            for i in range(3):
                node_normals[self.mesh_elements[elem, i], :] += \
                    self.n_hat[elem, :] * self.areas[elem] / 3.0
                
        node_normals /= np.linalg.norm(node_normals, axis=1)[:, np.newaxis] \
                        + 1e-300
        self.node_n_hat = node_normals
    
    def get_characteristic_length(self) -> float:
        """
        Compute the characteristic length of the mesh.

        Returns:
            char_length (float): Characteristic length of the mesh.
        """
        self.char_length = np.maximum.reduce([np.linalg.norm(self.e1, axis=1),
                                  np.linalg.norm(self.e2, axis=1),
                                  np.linalg.norm(self.e1 - self.e2, axis=1)])
    
    def node_in_element(self, node_idx: int) -> np.ndarray:
        """
        Get the indices of elements connected to a given node.

        Args:
            node_idx (int): Index of the node.

        Returns:
            elements (np.ndarray): Array of element indices connected to the 
                given node.
        """
        return np.where(self.mesh_elements == node_idx)[0]
    
    def compute_jump_coefficients(self) -> np.ndarray:
        """
        Compute jump coefficients C(x) for the double-layer operator. Uses the
        Van Oosterom and Strackee method to compute the solid angle.

        For each node, compute the interior solid angle Ω(x) subtended by
        the adjacent triangular elements, then set

            C(x) = Ω(x) / (4π).

        For smooth closed surfaces, this tends to 0.5 everywhere.
        For open or non-smooth surfaces, values reflect the local geometry.

        Returns:
            np.ndarray: Jump coefficients, shape (num_nodes,).
        """
        coeff = np.zeros(self.num_nodes, dtype=float)

        for i in range(self.num_nodes):
            elems = self.node_in_el[i]
            x0 = self.mesh_nodes[i]
            n0 = self.node_n_hat[i]
            obs = x0 + 1e-6 * n0

            solid_angle = 0.0
            for e in elems:
                v0 = self.mesh_nodes[self.mesh_elements[e, 0]]
                v1 = self.mesh_nodes[self.mesh_elements[e, 1]]
                v2 = self.mesh_nodes[self.mesh_elements[e, 2]]

                r0 = v0 - obs
                r1 = v1 - obs
                r2 = v2 - obs

                a = np.linalg.norm(r0)
                b = np.linalg.norm(r1)
                c = np.linalg.norm(r2)
                triple = np.dot(r0, np.cross(r1, r2))
                denom = (a*b*c
                        + np.dot(r0, r1)*c
                        + np.dot(r1, r2)*a
                        + np.dot(r2, r0)*b)

                angle = 2.0 * np.arctan2(triple, denom)
                solid_angle += angle

            coeff[i] = abs(solid_angle) / (4.0 * np.pi)

        self.jump_coefficients = coeff
        return coeff

    def project_velocity_bc(self) -> None:
        """Project vector velocity boundary data onto the mesh normals.

        If ``self.velocity_BC`` is given as full 3-D velocity components
        with shape (num_nodes, 3), this replaces it by its scalar
        projection along the nodal normals:

            q_i = v_i · n_i

        where ``v_i`` is the velocity vector at node i and ``n_i`` is the
        unit outward normal stored in ``self.node_n_hat``.

        Does nothing if ``velocity_BC`` is already 1-D.

        Raises:
            ValueError: if velocity_BC has an unexpected shape.
        """
        if self.Neumann_BC.ndim == 1:
            return  # already scalar

        if self.Neumann_BC.shape == (self.num_nodes, 3):
            self.Neumann_BC = np.einsum(
                "ij,ij->i", self.Neumann_BC, self.node_n_hat
            )
            return

        raise ValueError(
            f"Neumann_BC has shape {self.Neumann_BC.shape}, "
            f"expected ({self.num_nodes},) or ({self.num_nodes},3)"
        )