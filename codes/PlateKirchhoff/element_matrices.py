'''
Code snippet from lecture Computational Acoustics
Technische Universität Braunschweig, Institute for Acoustics

All rights reserved.

Author: Harikrishnan Sreekumar
'''
import numpy as np
from codes.PlateKirchhoff.shape import poly_hermite, poly_hermite_d, poly_hermite_dd, ind_dof_xi, ind_dof_eta

def build_element_stiffness(material, coord, c_s, n_unkwn_elem_disp):
    '''
    Function for building the element stiffness matrix for quadrilateral Kirchoff plate elements
    using bilinear Lagrange shape functions for geometry approximation.
    '''
    # Gauss points
    gp = np.array([np.sqrt((3+2*np.sqrt(6/5))/7), -np.sqrt((3+2*np.sqrt(6/5))/7),
                   np.sqrt((3-2*np.sqrt(6/5))/7), -np.sqrt((3-2*np.sqrt(6/5))/7)])
    # Weights
    weight = np.array([(18-np.sqrt(30))/36, (18-np.sqrt(30))/36, (18+np.sqrt(30))/36, (18+np.sqrt(30))/36 ])

    EModulus = material[0]
    nu = material[1]
    rho = material[2]
    t = c_s[2]

    C = EModulus*np.power(t,3)/(12*(1-np.power(nu,2))) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, 0.5*(1-nu)]])
    
    # Building the element stiffness matrix
    KEle = np.zeros((n_unkwn_elem_disp, n_unkwn_elem_disp))

    # Geometrical ansatz
    N = np.zeros((len(gp), len(gp), 4))
    N_xi = np.zeros((len(gp), len(gp), 4))
    N_eta = np.zeros((len(gp), len(gp), 4))
    
    for ii in range(len(gp)):
        for jj in range(len(gp)):
            N[jj,:,ii] = [0.25 * (1.0 - gp[ii]) * (1.0 - gp[jj]),
                        0.25 * (1.0 + gp[ii]) * (1.0 - gp[jj]),
                        0.25 * (1.0 + gp[ii]) * (1.0 + gp[jj]),
                        0.25 * (1.0 - gp[ii]) * (1.0 + gp[jj])]

            N_xi[jj,:,ii] = [-0.25 * (1.0 - gp[jj]),
                           0.25 * (1.0 - gp[jj]),
                           0.25 * (1.0 + gp[jj]),
                           -0.25 * (1.0 + gp[jj])]

            N_eta[jj,:,ii] = [-0.25 * (1.0 - gp[ii]),
                            -0.25 * (1.0 + gp[ii]),
                            0.25 * (1.0 + gp[ii]),
                            0.25 * (1.0 - gp[ii])]

    # P3 Kirchoff Element Ansatz for 4 DOFs per node
    for ii in range(len(gp)):
        xi = gp[ii]

        for jj in range(len(gp)):
            eta = gp[jj]
            
            # Jacobian
            J = np.matmul(np.transpose(coord), np.transpose([N_xi[jj,:,ii], N_eta[jj,:,ii]]))            
            detJ = np.linalg.det(J)
            a_x = J[0,0]   # Edge length in x direction
            a_y = J[1,1]   # Edge length in y direction

            # B_i = [d^2H_i(xi,eta)/dxi^2,
            #        d^2H_i(xi,eta)/deta^2,
            #        d^2H_i(xi,eta)/dxieta]
            H_eta = poly_hermite(eta,a_y)
            ddH_xi = poly_hermite_dd(xi,a_x)
            H_xi = poly_hermite(xi,a_x)
            ddH_eta = poly_hermite_dd(eta,a_y)
            dH_xi = poly_hermite_d(xi,a_x)
            #print(xi,a_x)
            #print(dH_xi)
            dH_eta = poly_hermite_d(eta,a_y)

            # H_l(xi,eta) = H_r(xi)H_s(eta)
            # Hbar_l(xi,eta) = H_r+1(xi)H_s(eta)
            # Hbarbar_l(xi,eta) = H_r(xi)H_s+1(eta)
            # {l,r,s} = {{1,1,1},{2,3,1},{3,3,3},{4,1,3}}
            ddH_xio = np.zeros(16)
            ddH_etao = np.zeros(16)
            ddH_xietao = np.zeros(16)
            for ih in range(4):
                ddH_xio[ih*4:ih*4+4] = ddH_xi[ind_dof_xi(ih)] * H_eta[ind_dof_eta(ih)] / (np.power(a_x,2))
                ddH_etao[ih*4:ih*4+4] = H_xi[ind_dof_xi(ih)] * ddH_eta[ind_dof_eta(ih)] / (np.power(a_y,2))
                ddH_xietao[ih*4:ih*4+4] = dH_xi[ind_dof_xi(ih)] * dH_eta[ind_dof_eta(ih)] / (a_x*a_y)
            # Calculating the element stiffness matrix KEle:
            for ki in range(16):
                for kj in range(16):
                    KEle[ki,kj] += (ddH_xio[ki]*C[0,0]*ddH_xio[kj]
                                    + ddH_xio[ki]*C[0,1]*ddH_etao[kj]
                                    + ddH_etao[ki]*C[1,0]*ddH_xio[kj]
                                    + ddH_etao[ki]*C[1,1]*ddH_etao[kj]
                                    + ddH_xietao[ki]*C[2,2]*ddH_xietao[kj]*4)*detJ*weight[ii]*weight[jj]
    return KEle

def build_element_mass(material, coord, c_s, n_unkwn_elem_disp):
    '''
    Function for building the element mass matrix for quadrilateral Kirchoff plate elements
    using bilinear Lagrange shape functions for geometry approximation.
    '''
    # Gauss points
    gp = np.array([np.sqrt((3+2*np.sqrt(6/5))/7), -np.sqrt((3+2*np.sqrt(6/5))/7),
                   np.sqrt((3-2*np.sqrt(6/5))/7), -np.sqrt((3-2*np.sqrt(6/5))/7)])
    # Weights
    weight = np.array([(18-np.sqrt(30))/36, (18-np.sqrt(30))/36, (18+np.sqrt(30))/36, (18+np.sqrt(30))/36 ])

    EModulus = material[0]
    nu = material[1]
    rho = material[2]
    t = c_s[2]

    C = EModulus*np.power(t,3)/(12*(1-np.power(nu,2))) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, 0.5*(1-nu)]])

    # Building the element stiffness matrix
    MEle = np.zeros((n_unkwn_elem_disp, n_unkwn_elem_disp))

    # Geometrical ansatz
    N = np.zeros((len(gp), len(gp), 4))
    N_xi = np.zeros((len(gp), len(gp), 4))
    N_eta = np.zeros((len(gp), len(gp), 4))
    
    for ii in range(len(gp)):
        for jj in range(len(gp)):
            N[jj,:,ii] = [0.25 * (1.0 - gp[ii]) * (1.0 - gp[jj]),
                        0.25 * (1.0 + gp[ii]) * (1.0 - gp[jj]),
                        0.25 * (1.0 + gp[ii]) * (1.0 + gp[jj]),
                        0.25 * (1.0 - gp[ii]) * (1.0 + gp[jj])]

            N_xi[jj,:,ii] = [-0.25 * (1.0 - gp[jj]),
                           0.25 * (1.0 - gp[jj]),
                           0.25 * (1.0 + gp[jj]),
                           -0.25 * (1.0 + gp[jj])]

            N_eta[jj,:,ii] = [-0.25 * (1.0 - gp[ii]),
                            -0.25 * (1.0 + gp[ii]),
                            0.25 * (1.0 + gp[ii]),
                            0.25 * (1.0 - gp[ii])]

    # P3 Kirchoff Element Ansatz for 4 DOFs per node
    for ii in range(len(gp)):
        xi = gp[ii]

        for jj in range(len(gp)):
            eta = gp[jj]

            # Jacobian
            J = np.matmul(np.transpose(coord), np.transpose([N_xi[jj,:,ii], N_eta[jj,:,ii]]))
            detJ = np.linalg.det(J)
            a_x = J[0,0]   # Edge length in x direction
            a_y = J[1,1]   # Edge length in y direction

            H_eta = poly_hermite(eta,a_y)
            H_xi = poly_hermite(xi,a_x)

            H_o = np.zeros(16)
            for ih in range(4):
                H_o[ih*4:ih*4+4] = np.multiply(H_xi[ind_dof_xi(ih)], H_eta[ind_dof_eta(ih)])

            # Calculating the element mass matrix MEle:
            for ki in range(16):
                for kj in range(16):
                    MEle[ki,kj] += H_o[ki]*rho*t*H_o[kj]*detJ*weight[ii]*weight[jj]
    return MEle