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

__all__ = ["Mesh", "BEMSolver",
            "CollocationAssembler", "GalerkinAssembler", 
            "ElementIntegratorCollocation", "ElementIntegratorGalerkin"]
__version__ = "0.0.0"