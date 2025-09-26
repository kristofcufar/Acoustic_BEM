"""
Boundary Element Method for Exterior Acoustic problems
"""

__version__ = "0.0.1"
from .mesh import Mesh
from .solve import BEMSolver

__all__ = ["Mesh", "BEMSolver"]
__version__ = "0.0.0"