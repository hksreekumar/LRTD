'''
Code snippet from lecture Computational Acoustics
Technische Universität Braunschweig, Institute for Acoustics

All rights reserved.

Author: Harikrishnan Sreekumar
'''
import numpy as np

def poly_hermite(p, l):
    '''Hermite Polynomial of third degree'''
    return np.array([(2-3*p+np.power(p,3))/4, (1-p-np.power(p,2)+np.power(p,3))*l/4, (2+3*p-np.power(p,3))/4,
                     (-1-p+np.power(p,2)+np.power(p,3))*l/4])

def poly_hermite_d(p, l):
    '''First Derivative Hermite'''
    return np.array([(-3+3*np.power(p,2))/4, (-1-2*p+3*np.power(p,2))*l/4, (3-3*np.power(p,2))/4,
                     (-1+2*p+3*np.power(p,2))*l/4])

def poly_hermite_dd(p, l):
    '''Second Derivative Hermite'''
    return np.array([(6*p)/4, (-2+6*p)*l/4, (-6*p)/4, (2+6*p)*l/4])

def ind_dof_xi(l):
    '''Return dof indices coordinate xi'''
    lrs = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    return np.array([lrs[l,0], lrs[l,0]+1, lrs[l,0], lrs[l,0]+1])

def ind_dof_eta(l):
    '''Return dof indices coordinate eta'''
    lrs = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
    return np.array([lrs[l,1], lrs[l,1], lrs[l,1]+1, lrs[l,1]+1])