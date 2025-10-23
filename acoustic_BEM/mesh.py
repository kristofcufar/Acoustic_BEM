import numpy as np
from acoustic_BEM.geometry import Object, Field

class Mesh:
    def __init__(self,
                 source_object: Object,
                 peripheral_objects: list[Object],
                 field: Field):
        
        """
        TODO: Complete docstring
        """
        if not source_object.frequency:
            raise ValueError("Source object must have a defined frequency of"\
            " oscillation.")
        if not source_object.Neumann_BC and not source_object.Dirichlet_BC:
            raise ValueError("Source object must have at least one boundary "
                             "condition defined (Neumann or Dirichlet).")
        self.source_object = source_object
        self.peripheral_objects = peripheral_objects
        self.field = field

        self.frequency = source_object.frequency
        self.c0 = source_object.c0
        self.rho0 = source_object.rho0
        
        self.merge()



    def merge(self):
        """
        Merge multiple mesh components into a single mesh. Initializes the Mesh
        object attributes, named in accordance with the downstream API.
        """
        # mesh_nodes, mesh_elements, Neumann_BC, Dirichlet_BC
        # v0, e1, e2, a2, n_hat, centroids, areas, node_in_el
        # node_n_hat
        # char_length, jump_coefficients

        components = [self.source_object] + self.peripheral_objects

        pass