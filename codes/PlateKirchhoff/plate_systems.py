
from fe_system import assemble_system
import fem_solver

import numpy as np

def plate_system_3param_XY_E(my_fe_data, param):
    # arrange
    my_fe_data.material[:,0] = param[1]
    
    # Matrix assembly
    KMAT, MMAT = assemble_system(my_fe_data)
    DMAT = np.zeros_like(MMAT)

    n = KMAT.shape[0]
    fload = np.zeros((n,1))
    fload[0,0] = 1
    
    yFEM = fem_solver.solve_dense_system_response(KMAT, MMAT, DMAT, fload, [param[0]])
    # filter z translation
    yFEM = np.real(yFEM[0::4, :])

    return yFEM.reshape(my_fe_data.grid_shape)