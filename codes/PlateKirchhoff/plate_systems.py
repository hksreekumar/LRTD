
from fe_system import assemble_system
import fem_solver

import numpy as np

def plate_system_2param_XY(my_fe_data, param):
    # arrange
    my_fe_data.material[:,0] = param[1]
    my_fe_data.material[:,1] = param[2]
    
    # Matrix assembly
    KMAT, MMAT = assemble_system(my_fe_data)
    DMAT = np.zeros_like(MMAT)

    n = KMAT.shape[0]
    fload = np.zeros((n,1))
    fload[0,0] = 1
    
    yFEM = fem_solver.solve_dense_system_response(KMAT, MMAT, DMAT, fload, [param[0]])
    # filter z translation
    yFEM = np.real(yFEM[0::4, :])

    return yFEM

def plate_system_5param_XY(my_fe_data, param):
    # arrange
    my_fe_data.material[:,0] = param[1] # Youngs modulus
    my_fe_data.material[:,1] = param[2] # Poisson's ratio
    my_fe_data.material[:,2] = param[3] # Density  
    
    my_fe_data.pDim[2] = param[4] # thickness 
    
    # Matrix assembly
    KMAT, MMAT = assemble_system(my_fe_data)
    DMAT = np.zeros_like(MMAT)

    n = KMAT.shape[0]
    fload = np.zeros((n,1))
    fload[0,0] = 1
    
    yFEM = fem_solver.solve_dense_system_response(KMAT, MMAT, DMAT, fload, [param[0]])
    # filter z translation
    yFEM = np.real(yFEM[0::4, :])

    return yFEM