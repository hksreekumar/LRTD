'''
Code snippet from lecture Computational Acoustics
Technische Universität Braunschweig, Institute for Acoustics

All rights reserved.

Author: Harikrishnan Sreekumar
'''
import numpy as np
from element_matrices import build_element_stiffness, build_element_mass

class fe_data:
    '''Class holder for FE data'''
    def __init__(self) -> None:
        self.coord = None
        self.connect = None
        self.pDim = None
        self.loadNode = None
        self.material = None
        self.bc = None
        self.grid_shape = None

def assemble_system(fe_data):
    '''Function to assemble the element matrices'''
    nUnknNode      = 4                         # number of unknowns per node
    nNodes         = fe_data.coord.shape[0]            # number of nodes
    nEle           = fe_data.connect.shape[0]          # number of elements
    nNodesEle      = fe_data.connect.shape[1]          # number of nodes per element
    nUnknEleDisp   = nUnknNode * nNodesEle     # number of unknowns per element                                        
    nUnkn          = nUnknNode * nNodes        # total number of unknowns
    nLoadNode      = 1 # loadNode.shape[0]     # number of nodal   loads
    if np.size(fe_data.bc):
        nBc        = fe_data.bc.shape[1]               # number of displacement boundary conditions
    else:
        nBc        = 0                         # number of displacement boundary conditions

    # conditions
    c_s = fe_data.pDim
    iBc = np.zeros((nBc,2))
    rBc = np.zeros((nBc,1))
    for i in range(nBc):    
        iBc[i] = fe_data.bc[0:2,i]
        rBc[i] = fe_data.bc[2,i]

    # Assembling of Global Stiffness Matrix
    K = np.zeros((nUnkn, nUnkn))
    ipos = np.zeros(nUnknEleDisp)
    for i in range(nEle):
        KEle = build_element_stiffness(fe_data.material[i], fe_data.coord[fe_data.connect[i].astype(int)],  c_s, nUnknEleDisp)
        z = 0
        for m in range(nNodesEle):       
            for k in range(nUnknNode):         
                ipos[z] = (fe_data.connect[i,m]) * nUnknNode + k
                z += 1

        # Insertion of the element matrix into the global stiffness matrix K:
        for j1 in range(len(ipos)):
            for j2 in range(len(ipos)):
                K[int(ipos[j1]),int(ipos[j2])] += KEle[j1,j2]

    # Assembling of Global Mass Matrix
    M = np.zeros((nUnkn,nUnkn)) 
    for i in range(nEle):
        MEle = build_element_mass(fe_data.material[i], fe_data.coord[fe_data.connect[i].astype(int)],  c_s, nUnknEleDisp)
        #print(MEle)
        z = 0
        for m in range(nNodesEle):       
            for k in range(nUnknNode):         
                ipos[z] = (fe_data.connect[i,m]) * nUnknNode + k
                z += 1

        # Insertion of the element matrix into the global mass matrix M:
        for j1 in range(len(ipos)):
            for j2 in range(len(ipos)):
                M[int(ipos[j1]),int(ipos[j2])] += MEle[j1,j2]

    # Informative prints
    #print('Total number of nodes: ' + str(nNodes))
    #print('Total number of elements: ' + str(nEle))
    #print('Total number of degrees of freedom: ' + str(nUnkn))

    # Implementation of the displacement boundary conditions:
    del_ = np.array([])
    for n in range(nBc):    
        ipos = ((iBc[n,0]) * nUnknNode + iBc[n,1])-1
        # stiffness
        K[int(ipos)] = np.zeros(nUnkn)
        K[:,int(ipos)] = np.zeros(nUnkn)
        K[int(ipos),int(ipos)] = 1
        # mass
        M[int(ipos)] = np.zeros(nUnkn)
        M[:,int(ipos)] = np.zeros(nUnkn)
        del_ = np.append(del_,ipos)

    return K, M

def assemble_load(fe_data):
    '''Function to assemble the load vector'''
    nUnknNode      = 4                         # number of unknowns per node
    nNodes         = fe_data.coord.shape[0]            # number of nodes
    nEle           = fe_data.connect.shape[0]          # number of elements
    nNodesEle      = fe_data.connect.shape[1]          # number of nodes per element
    nUnknEleDisp   = nUnknNode * nNodesEle     # number of unknowns per element                                        
    nUnkn          = nUnknNode * nNodes        # total number of unknowns
    nLoadNode      = 1 # loadNode.shape[0]     # number of nodal   loads
    
    iLoadNode = np.zeros((nLoadNode,2))
    rLoadNode = np.zeros((nLoadNode,1))
    for i in range(nLoadNode):
        iLoadNode[i] = fe_data.loadNode[0:2]
        rLoadNode[i] = fe_data.loadNode[2]

    # Assembling of global force vector
    F = np.zeros(nUnkn)
    # Insertion of the nodal loads into the load vector f:
    for n in range(nLoadNode):
        ipos = ((iLoadNode[n,0]-1) * nUnknNode + iLoadNode[n,1]-1)
        F[int(ipos)] += rLoadNode[n]
    
    if np.size(fe_data.bc):
        nBc        = fe_data.bc.shape[1]               # number of displacement boundary conditions
    else:
        nBc        = 0                         # number of displacement boundary conditions

    # conditions
    c_s = fe_data.pDim
    iBc = np.zeros((nBc,2))
    rBc = np.zeros((nBc,1))
    for i in range(nBc):    
        iBc[i] = fe_data.bc[0:2,i]
        rBc[i] = fe_data.bc[2,i]

    # Implementation of the displacement boundary conditions:
    del_ = np.array([])
    for n in range(nBc):    
        ipos = ((iBc[n,0]) * nUnknNode + iBc[n,1])-1
        F[int(ipos)] = 0
        del_ = np.append(del_,ipos)

    return F