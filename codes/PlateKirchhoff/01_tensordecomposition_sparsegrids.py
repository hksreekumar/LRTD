import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import sys
sys.path.append('../..') # source
sys.path.append('./') # source

import time

from codes.PlateKirchhoff.element_matrices import build_element_stiffness, build_element_mass
from codes.PlateKirchhoff.fe_system import fe_data, assemble_load, assemble_system
from codes.PlateKirchhoff import fem_solver

from codes.PlateKirchhoff.plate_systems import plate_system_3param_XY_E

solverTic = time.time()
# parameters

#coord = np.array([[0.,  0. ], [1.,  0. ], [1.,  0.7], [0.,  0.7]])
c_s = np.array([1., 0.7, 0.003])

# Structured nodal coordinates 
nElem_x = 5                                    # Number of elements in x direction
nElem_y = 5                                    # Number of elements in y direction

coord = np.zeros(((nElem_y+1)*(nElem_x+1),2)) # Two coords per node (x,y)
a_x = c_s[0] / nElem_x
a_y = c_s[1] / nElem_y
for m in range(nElem_x+1):
    for n in range(nElem_y+1):
        coord[m*(nElem_y+1)+n] = np.array([(m)*a_x, (n)*a_y])


# Generate element connectivity
connect = np.zeros((nElem_y*nElem_x,4), dtype=np.int32) # Four nodes per element (linear quadrilateral)
for m in range(nElem_x):
    for n in range(nElem_y):
        elem_connect = np.zeros(4, dtype=np.int32)
        elem_connect[0] = (m)*(nElem_y+1) + (n)
        elem_connect[1] = elem_connect[0] + (nElem_y+1)
        elem_connect[2] = elem_connect[1] + 1
        elem_connect[3] = elem_connect[0] + 1
        connect[m*nElem_y+n] = elem_connect

# Material properties
mat = np.array([7.0e+10, 3.4e-01, 2.7e+03])
material = np.zeros((nElem_x*nElem_y, 3))
material[:,:] = mat    # Emod, Poissons ratio, Density (Same material for all elements / homogeneous domain)

n_unkwn_elem_disp = connect.shape[0]*4
solverToc = time.time()
solverTime = solverToc - solverTic
print('>> Time for mesh gen: ' + str(round(solverTime,3)) + ' seconds.')

# arrange
my_fe_data = fe_data()
my_fe_data.coord = coord
my_fe_data.material = material
my_fe_data.pDim = c_s
my_fe_data.connect = connect
my_fe_data.bc = np.array([])
my_fe_data.grid_shape = (nElem_x+1, nElem_y+1)


solverTic = time.time()
data_slice = plate_system_3param_XY_E(my_fe_data,[100, 7e10])
solverToc = time.time()
solverTime = solverToc - solverTic
print('>> Time for single solver call: ' + str(round(solverTime,3)) + ' seconds.')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(coord[:,0],coord[:,1], data_slice.flatten(), cmap='viridis')
plt.show()


# Sparse grids