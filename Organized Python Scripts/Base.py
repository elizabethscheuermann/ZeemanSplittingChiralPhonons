from re import A
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Trigonal Lattice
x_tri = np.diag([1, -.5, -.5])
y_tri = np.diag([0, np.sqrt(3)/2, -np.sqrt(3)/2])
h0_tri = -1 * (np.eye(3, 3, -2) + np.eye(3, 3, -1) + np.eye(3, 3, 1) + np.eye(3, 3, 2))
Lz_tri = 1j * (-np.eye(3, 3, -2) + np.eye(3, 3, -1) - np.eye(3, 3, 1) + np.eye(3, 3, 2)) / np.sqrt(3)
q_tri = 3
H_MF_tri = lambda O, J, B : h0_tri + B * Lz_tri + .5 * J * q_tri * (O[0]**2 + O[1]**2) * np.identity(3) - J * q_tri * (O[0] * x_tri + O[1] * y_tri)
Z_MF_tri = lambda O, J, B, T : np.trace(sp.linalg.expm(-H_MF_tri(O, J, B)/T))
F_MF_tri = lambda O, J, B, T : np.real(-T * np.log(Z_MF_tri(O, J, B, T))) if T!=0 else np.real(np.min(np.linalg.eigvals(H_MF_tri(O, J, B))))
E_MF_tri = lambda O, J, B : np.sort(np.linalg.eigvals(H_MF_tri(O, J, B)))
P_MF_tri = lambda O, J, B, T : np.exp(-E_MF_tri(O, J, B)/T)/Z_MF_tri(O, J, B, T) if T!=0 else [1,0,0]
def M_MF_tri(O, J, B, T):
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

# Square Lattice
x_squ = np.diag([1, 0, -1, 0])
y_squ = np.diag([0, 1, 0, -1])
h0_squ = -1 * (np.eye(4, 4, -3) + np.eye(4, 4, -1) + np.eye(4, 4, 1) + np.eye(4, 4, 3))
Lz_squ = -1j * (-np.eye(4, 4, -3) + np.eye(4, 4, -1) - np.eye(4, 4, 1) + np.eye(4, 4, 3))/2
q_squ = 4
H_MF_squ = lambda O, J, B : h0_squ + B * Lz_squ + .5 * J * q_squ * (O[0]**2 + O[1]**2) * np.identity(4) - J * q_squ * (O[0] * x_squ + O[1] * y_squ)
Z_MF_squ = lambda O, J, B, T : np.trace(sp.linalg.expm(-H_MF_squ(O, J, B)/T))
F_MF_squ = lambda O, J, B, T : np.real(-T * np.log(Z_MF_squ(O, J, B, T))) if T!=0 else np.real(np.min(np.linalg.eigvals(H_MF_squ(O, J, B))))
E_MF_squ = lambda O, J, B : np.sort(np.linalg.eigvals(H_MF_squ(O, J, B)))
P_MF_squ = lambda O, J, B, T : np.exp(-E_MF_squ(O, J, B)/T)/Z_MF_squ(O, J, B, T) if T!=0 else [1,0,0,0]
x0_squ = [1e-2, 1e-1]
def M_MF_squ(O, J, B, T):
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

# Cubic Lattice
x_cub = np.diag([1, 0, 0, -1, 0, 0])
y_cub = np.diag([0, 1, 0, 0, -1, 0])
z_cub = np.diag([0, 0, 1, 0, 0, -1])
h0_cub = -1 *(np.eye(6,6, -5) + np.eye(6,6, -4) + np.eye(6,6, -2) + np.eye(6,6, -1) + np.eye(6,6, 1) + np.eye(6,6, 2) + np.eye(6,6, 4) + np.eye(6,6, 5))
Lx_cub = np.insert(np.insert(Lz_squ, [0, 2], 0, axis = 1),[0,2],0,axis = 0)
Ly_cub = np.insert(np.insert(Lz_squ, [1, 3], 0, axis = 1),[1,3],0,axis = 0)
Lz_cub = np.insert(np.insert(Lz_squ, [2, 4], 0, axis = 1),[2,4],0,axis = 0)
q_cub = 6
H_MF_cub = lambda O, J, B : h0_cub + B[0] * Lx_cub + B[1] * Ly_cub + B[2] * Lz_cub + .5 * J * q_cub * (O[0]**2 + O[1]**2 + O[2]**2) * np.identity(6) - J * q_cub * (O[0] * x_cub + O[1] * y_cub + O[2] * z_cub)
Z_MF_cub = lambda O, J, B, T : np.real(np.trace(sp.linalg.expm(-H_MF_cub(O, J, B)/T)))
F_MF_cub = lambda O, J, B, T : np.real(-T * np.log(Z_MF_cub(O, J, B, T))) if T!=0 else np.real(np.min(np.linalg.eigvals(H_MF_cub(O, J, B))))
E_MF_cub = lambda O, J, B : np.sort(np.linalg.eigvals(H_MF_cub(O, J, B)))
P_MF_cub = lambda O, J, B, T : np.exp(-E_MF_cub(O, J, B)/T)/Z_MF_cub(O, J, B, T) if T!=0 else [1,0,0,0,0,0]
x0_cub = [1e-2, 1e-1, 2e-1]
def M_MF_cub(O, J, B, T):
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

print(Lz_tri)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             