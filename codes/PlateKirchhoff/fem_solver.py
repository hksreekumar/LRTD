import numpy as np
import scipy as sp

# @brief function to compute frf for a set of frequencies
def solve_dense_system(KMAT, MMAT, DMAT, BMAT, CMAT, freq):
    n_inp = BMAT.shape[1]
    n_out = CMAT.shape[1]
    H_ROM = np.zeros((n_out, n_inp, len(freq)), dtype=complex)
    for id_freq, every_freq in enumerate(freq):
        omega = 2*np.pi*every_freq
        K_DYN = -omega*omega*MMAT + 1j*omega*DMAT + KMAT
        pdt = np.linalg.solve(K_DYN, BMAT)
        pdt = np.matmul(CMAT.T.conjugate(), pdt)

        H_ROM[:,:,id_freq] = pdt

    return H_ROM

# @brief function to compute frf for a set of frequencies
def solve_dense_system_response(KMAT, MMAT, DMAT, f, freq):
    n_inp = 1
    n_out = KMAT.shape[1]
    H_ROM = np.zeros((n_out, len(freq)), dtype=complex)
    for id_freq, every_freq in enumerate(freq):
        omega = 2*np.pi*every_freq
        K_DYN = -omega*omega*MMAT + 1j*omega*DMAT + KMAT
        pdt = np.linalg.solve(K_DYN, f)

        H_ROM[:,id_freq] = pdt.flatten()

    return H_ROM