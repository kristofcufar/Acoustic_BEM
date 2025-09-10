import numpy as np
from tqdm import tqdm

from .kernels import r_vec, G, dG_dr, dG_dn_y, dG_dn_x, d2G_dn_x_dn_y
from .quadrature import (standard_triangle_quad,
                        duffy_rule,
                        telles_rule,
                        map_to_physical_triangle)


