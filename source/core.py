import numpy as np
from tqdm import tqdm

from kernels import r_vec, G, dG_dr, dG_dn_y, dG_dn_x, d2G_dn_x_dn_y

class Core:
    def __init__(self,
                 mesh_nodes: np.ndarray,
                 mesh_elements: np.ndarray,
                 velocity_BC: np.ndarray,
                 frequency: float,
                 c0: float = 343.0,
                 rho0: float = 1.225,):
        
        """
        Initialize the Core class for acoustic boundary element method.

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
    
        Returns:
            None
        """
        
        self.mesh_nodes = mesh_nodes
        self.mesh_elements = mesh_elements
        self.velocity_BC = velocity_BC
        self.frequency = frequency
        self.c0 = c0
        self.rho0 = rho0
        self.num_nodes = mesh_nodes.shape[0]
        self.num_elements = mesh_elements.shape[0]

        self.omega = 2 * np.pi * frequency
        self.k = self.omega / c0