from re import A
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

### TRIANGULAR LATTICE
x_tri = np.diag([1, -.5, -.5]) # x operator
y_tri = np.diag([0, np.sqrt(3)/2, -np.sqrt(3)/2]) # y operator
h0_tri = -1 * (np.eye(3, 3, -2) + np.eye(3, 3, -1) + np.eye(3, 3, 1) + np.eye(3, 3, 2)) # transfer matrix
Lz_tri = 1j * (-np.eye(3, 3, -2) + np.eye(3, 3, -1) - np.eye(3, 3, 1) + np.eye(3, 3, 2)) / np.sqrt(3) # pseudo-angular momentum
q_tri = 3 # coordination number
H_MF_tri = lambda O, J, B : h0_tri + B * Lz_tri + .5 * J * q_tri * (O[0]**2 + O[1]**2) * np.identity(3) - J * q_tri * (O[0] * x_tri + O[1] * y_tri) # Mean field hamiltonian
Z_MF_tri = lambda O, J, B, T : np.trace(sp.linalg.expm(-H_MF_tri(O, J, B)/T)) # Mean field Partition function 
F_MF_tri = lambda O, J, B, T : np.real(-T * np.log(Z_MF_tri(O, J, B, T))) if T!=0 else np.real(np.min(np.linalg.eigvals(H_MF_tri(O, J, B)))) # Mean field free energy
E_MF_tri = lambda O, J, B : np.sort(np.linalg.eigvals(H_MF_tri(O, J, B))) # Mean field eigenvalues
P_MF_tri = lambda O, J, B, T : np.exp(-E_MF_tri(O, J, B)/T)/Z_MF_tri(O, J, B, T) if T!=0 else [1,0,0] # Mean field probabilities
def M_MF_tri(O, J, B, T):
    """Mean Field Magnetization"""

    # Eigenvalues
    vals, vecs = np.linalg.eig(H_MF_tri(O, J, B))

    # Sort lowest to largest (lets T=0 case work...)
    ids = vals.argsort()
    vals = vals[ids]
    vecs = vecs[:, ids]

    probs = np.exp(-1 * vals/T)/Z_MF_tri(O, J, B,T) if T!= 0 else [1, 0, 0]

    m = 0
    for i in range(3):
        m -= probs[i] * np.conj(vecs[:, i]).T @ Lz_tri @ vecs[:, i]

    return m
x0_tri = [1e-2, 1e-1]

### SQUARE LATTICE
x_squ = np.diag([1, 0, -1, 0]) # x operator
y_squ = np.diag([0, 1, 0, -1]) # y operator
h0_squ = -1 * (np.eye(4, 4, -3) + np.eye(4, 4, -1) + np.eye(4, 4, 1) + np.eye(4, 4, 3)) # transfer matrix
Lz_squ = -1j * (-np.eye(4, 4, -3) + np.eye(4, 4, -1) - np.eye(4, 4, 1) + np.eye(4, 4, 3))/2 # pseudo-angular momentum
q_squ = 4 # coordination number
H_MF_squ = lambda O, J, B : h0_squ + B * Lz_squ + .5 * J * q_squ * (O[0]**2 + O[1]**2) * np.identity(4) - J * q_squ * (O[0] * x_squ + O[1] * y_squ) # Mean field hamiltonian
Z_MF_squ = lambda O, J, B, T : np.trace(sp.linalg.expm(-H_MF_squ(O, J, B)/T)) # Mean field partition function
F_MF_squ = lambda O, J, B, T : np.real(-T * np.log(Z_MF_squ(O, J, B, T))) if T!=0 else np.real(np.min(np.linalg.eigvals(H_MF_squ(O, J, B)))) # Mean field free energy
E_MF_squ = lambda O, J, B : np.sort(np.linalg.eigvals(H_MF_squ(O, J, B))) # Mean field eigenvalues
P_MF_squ = lambda O, J, B, T : np.exp(-E_MF_squ(O, J, B)/T)/Z_MF_squ(O, J, B, T) if T!=0 else [1,0,0,0] # Mean field probabilities
x0_squ = [1e-2, 1e-1]
def M_MF_squ(O, J, B, T):
    """ Mean Field Magnetization"""
    # Eigenvalues
    vals, vecs = np.linalg.eig(H_MF_squ(O, J, B))

    # Sort lowest to largest (lets T=0 case work...)
    ids = vals.argsort()
    vals = vals[ids]
    vecs = vecs[:, ids]
    probs = np.exp(-1 * vals/T)/Z_MF_squ(O, J, B,T) if T!=0 else [1, 0, 0, 0]

    m = 0
    for i in range(4):
        m -= probs[i] * np.conj(vecs[:, i]).T @ Lz_squ @ vecs[:, i]

    return m

### CUBIC LATTICE
x_cub = np.diag([1, 0, 0, -1, 0, 0]) # x operator
y_cub = np.diag([0, 1, 0, 0, -1, 0]) # y operator
z_cub = np.diag([0, 0, 1, 0, 0, -1]) # z operator
h0_cub = -1 *(np.eye(6,6, -5) + np.eye(6,6, -4) + np.eye(6,6, -2) + np.eye(6,6, -1) + np.eye(6,6, 1) + np.eye(6,6, 2) + np.eye(6,6, 4) + np.eye(6,6, 5)) # transfer matrix
Lx_cub = np.insert(np.insert(Lz_squ, [0, 2], 0, axis = 1),[0,2],0,axis = 0) # x pseudo-angular momentum
Ly_cub = np.insert(np.insert(Lz_squ, [1, 3], 0, axis = 1),[1,3],0,axis = 0) # y pseudo-angular momentum
Lz_cub = np.insert(np.insert(Lz_squ, [2, 4], 0, axis = 1),[2,4],0,axis = 0) # z pseudo-angular momentum
q_cub = 6 # coordination number
H_MF_cub = lambda O, J, B : h0_cub + B[0] * Lx_cub + B[1] * Ly_cub + B[2] * Lz_cub + .5 * J * q_cub * (O[0]**2 + O[1]**2 + O[2]**2) * np.identity(6) - J * q_cub * (O[0] * x_cub + O[1] * y_cub + O[2] * z_cub) # mean field hamiltonian
Z_MF_cub = lambda O, J, B, T : np.real(np.trace(sp.linalg.expm(-H_MF_cub(O, J, B)/T))) # mean field partition function
F_MF_cub = lambda O, J, B, T : np.real(-T * np.log(Z_MF_cub(O, J, B, T))) if T!=0 else np.real(np.min(np.linalg.eigvals(H_MF_cub(O, J, B)))) # mean field free energy
E_MF_cub = lambda O, J, B : np.sort(np.linalg.eigvals(H_MF_cub(O, J, B))) # mean field eigenvalues
P_MF_cub = lambda O, J, B, T : np.exp(-E_MF_cub(O, J, B)/T)/Z_MF_cub(O, J, B, T) if T!=0 else [1,0,0,0,0,0] # mean field probabilities
x0_cub = [1e-2, 1e-1, 2e-1] 
def M_MF_cub(O, J, B, T):
    """ Mean field magnetization"""

    # Eigenvalues
    vals, vecs = np.linalg.eig(H_MF_cub(O, J, B))

    # Sort lowest to largest (lets T=0 case work...)
    ids = vals.argsort()
    vals = vals[ids]
    vecs = vecs[:, ids]
    probs = np.exp(-1 * vals/T)/Z_MF_cub(O, J, B,T) if T!=0 else [1, 0, 0, 0, 0, 0]

    m = 0
    for i in range(6):
        m -= probs[i] * np.conj(vecs[:, i]).T @ Lz_cub @ vecs[:, i]

    return m

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             