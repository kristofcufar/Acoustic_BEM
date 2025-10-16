"""
Boundary Element Method for Exterior Acoustic problems
"""

__version__ = "0.0.1"
from .mesh import Mesh
from .integrators import (ElementIntegratorCollocation, 
                          ElementIntegratorGalerkin)
from .matrix_assembly import (CollocationAssembler, 
                              GalerkinAssembler)
from .solve import BEMSolver

from .kernels import (r_vec, G, dG_dr, 
                          dG_dn_y, dG_dn_x, 
                          d2G_dn_x_dn_y,
                          ImpedanceGreen3D)

__all__ = ["Mesh", "BEMSolver",
            "CollocationAssembler", "GalerkinAssembler", 
            "ElementIntegratorCollocation", "ElementIntegratorGalerkin"]
__version__ = "0.0.0"