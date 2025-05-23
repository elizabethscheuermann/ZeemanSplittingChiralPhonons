from re import A
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

### GENERALIZED GELLMANN MATRICES
def GeneralizedGellMannMatrix(dim, coefs):
    if len(coefs) != dim**2 - 1:
        print(r"Must have $n^2 - 1$ coefficients")
    else:
        operator = np.zeros((dim,dim), dtype = complex)
        ix, iy = np.triu_indices(dim, k=1)
        K = int(.5 * dim * (dim - 1))
        operator[ix, iy] = coefs[0:K] + 1j * coefs[K:2*K]
        operator[iy,ix] = coefs[0:K] - 1j * coefs[K:2*K]

        for i in range(dim-1):
            operator[i+1,i+1] -= np.sqrt(2/((i+1)*(i+2)))*coefs[2*K + i]*(i+1)
            for j in range(i+1):
                operator[j][j] += np.sqrt(2/((i+1)*(i+2)))*coefs[2*K + i]

    return operator


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

# Phonon Dynamics Equation
def DE_MF_tri(operator, J, B, x_exp, y_exp, comm_x_op_exp, comm_y_op_exp, Q):
    # Get commutators
    h0_comm = h0_tri @ operator - operator @ h0_tri
    Lz_comm = Lz_tri @ operator - operator @ Lz_tri
    x_comm = x_tri @ operator - operator @ x_tri
    y_comm = y_tri @ operator - operator @ y_tri

    # Dynamics Equation
    d_eq = h0_comm + B * Lz_comm - J * q_tri * (x_exp * x_comm + y_exp * y_comm) - J * Q * (comm_x_op_exp * x_tri + comm_y_op_exp * y_tri)

    return d_eq

gmm_tri = [[[0, 1, 0], # x matrices
            [1, 0, 0],
            [0, 0, 0]
            ],
            [[0, 0, 1],
            [0, 0, 0],
            [1, 0, 0]
             ],
            [[0, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
             ],
            [[0, 1j, 0], # y matrices
            [-1j, 0, 0],
            [0, 0, 0]
             ],
            [[0, 0, 1j],
            [0, 0, 0],
            [-1j, 0, 0]
             ],
            [[0, 0, 0],
            [0, 0, 1j],
            [0, -1j, 0]
             ],
            [[1, 0, 0], # z matrices
            [0, -1, 0],
            [0, 0, 0]
             ],
            [[1/np.sqrt(3), 0, 0],
            [0, 1/np.sqrt(3), 0],
            [0, 0, -2/np.sqrt(3)]
             ]
        ]

# Phonon Dynamics Loop
def PD_MF_tri(O, J, B, T, Q):
    # Get states and probabilities
    vals, vecs = np.linalg.eigh(H_MF_tri(O, J, B))

    # Sort
    ids = vals.argsort()
    vals = vals[ids]
    vecs = vecs[:, ids]
    probs = np.exp(-1 * vals/T)/Z_MF_tri(O,J,B, T) if T!=0 else [1,0,0]

    # Evaluate <x>, <y>
    x_exp = 0
    y_exp = 0
    for i in range(3):
        x_exp += probs[i] * np.real(np.conj(vecs[:, i]).T @ x_tri @ vecs[:, i])
        y_exp += probs[i] * np.real(np.conj(vecs[:, i]).T @ y_tri @ vecs[:, i])

    # Dynamics Matrix
    D = np.zeros((8,8), dtype = complex)

    # Loop over matrices
    for i in range(8):
        op1 = gmm_tri[i]

        # Evaluate <[x, L]>, <[y,L]>
        comm_x_op_exp = 0
        comm_y_op_exp = 0
        for j in range(3):
            comm_x_op_exp += probs[j]*np.conj(vecs[:, j]).T @ (x_tri @ op1 - op1 @ x_tri) @ vecs[:,j]
            comm_y_op_exp += probs[j]*np.conj(vecs[:, j]).T @ (y_tri @ op1 - op1 @ y_tri) @ vecs[:,j]

        # Get Dynamics Equation
        d_eq = DE_MF_tri(op1, J, B, x_exp, y_exp, comm_x_op_exp, comm_y_op_exp, Q)

        # Fill Dynamics Matrix
        for j in range(8):
            op2 = gmm_tri[j]

            D[j,i] = np.trace(d_eq @ op2)/3

        # Solve system
        op_freqs, op_vecs = np.linalg.eig(D)

        # Get positive frequencies
        ids = op_freqs.argsort()[::-1]
        op_freqs = op_freqs[ids][:3]
        op_vecs = op_vecs[:, ids][:, :3]

        # Get norms to ground state
        op_norms = []
        for i in range(len(op_freqs)):
            op = np.zeros((3,3), dtype = complex)
            for j in range(8):
                op = op + op_vecs[j, i] * np.array(gmm_tri[j])
            
            op_norms.append(np.conjugate(op @ vecs[:,0]).T @ op @ vecs[:,0])

    return op_freqs, op_norms

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
x0_squ = [.1, 0]
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

# Phonon Dynamics Equation
def DE_MF_squ(operator, J, B, x_exp, y_exp, comm_x_op_exp, comm_y_op_exp, Q):
    # Get commutators
    h0_comm = h0_squ @ operator - operator @ h0_squ
    Lz_comm = Lz_squ @ operator - operator @ Lz_squ
    x_comm = x_squ @ operator - operator @ x_squ
    y_comm = y_squ @ operator - operator @ y_squ

    # Dynamics Equation
    d_eq = h0_comm + B * Lz_comm - J * q_squ * (x_exp * x_comm + y_exp * y_comm) - J * Q * (comm_x_op_exp * x_squ + comm_y_op_exp * y_squ)

    return d_eq

# GGMM
gmm_squ = [
            [[0, 1, 0, 0], # x matrices
             [1, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]
            ],
            [[0, 0, 1, 0],
             [0, 0, 0, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 0]
            ],
            [[0, 0, 0, 1],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [1, 0, 0, 0]
            ],
            [[0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0]
            ],
            [[0, 0, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 0, 0],
             [0, 1, 0, 0]
            ],
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 1],
             [0, 0, 1, 0]
            ],
            [[0, 1j, 0, 0], # y matrices
            [-1j, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
            ],
            [[0, 0, 1j, 0],
            [0, 0, 0, 0],
            [-1j, 0, 0, 0],
            [0, 0, 0, 0]
            ],
            [[0, 0, 0, 1j],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [-1j, 0, 0, 0]
            ],
            [[0, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, -1j, 0, 0],
            [0, 0, 0, 0]
            ],
            [[0, 0, 0, 0],
            [0, 0, 0, 1j],
            [0, 0, 0, 0],
            [0, -1j, 0, 0]
            ],
            [[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1j],
            [0, 0, -1j, 0]
            ],
            [[1, 0, 0, 0], # z matrices
            [0, -1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
            ],
            [[1/np.sqrt(3), 0, 0, 0],
            [0, 1/np.sqrt(3), 0, 0],
            [0, 0, -2/np.sqrt(3), 0],
            [0, 0, 0, 0]
            ],
            [[1/np.sqrt(6), 0, 0, 0],
            [0, 1/np.sqrt(6), 0, 0],
            [0, 0, 1/np.sqrt(6), 0],
            [0, 0, 0, -3/np.sqrt(6)]
            ]
 ]



# Phonon Dynamics Loop
def PD_MF_squ(O, J, B, T, Q):
    # Get states and probabilities
    vals, vecs = np.linalg.eigh(H_MF_squ(O, J, B))

    # Sort
    ids = vals.argsort()
    vals = vals[ids]
    vecs = vecs[:, ids]
    probs = np.exp(-1 * vals/T)/Z_MF_squ(O,J,B, T) if T!=0 else [1,0,0,0]

    # Evaluate <x>, <y>
    x_exp = 0
    y_exp = 0
    for i in range(4):
        x_exp += probs[i] * np.real(np.conj(vecs[:, i]).T @ x_squ @ vecs[:, i])
        y_exp += probs[i] * np.real(np.conj(vecs[:, i]).T @ y_squ @ vecs[:, i])

    # Dynamics Matrix
    D = np.zeros((15,15), dtype = complex)

    # Loop over matrices
    for i in range(15):
        op1 = gmm_squ[i]

        # Evaluate <[x, L]>, <[y,L]>
        comm_x_op_exp = 0
        comm_y_op_exp = 0
        for j in range(4):
            comm_x_op_exp += probs[j]*np.conj(vecs[:, j]).T @ (x_squ @ op1 - op1 @ x_squ) @ vecs[:,j]
            comm_y_op_exp += probs[j]*np.conj(vecs[:, j]).T @ (y_squ @ op1 - op1 @ y_squ) @ vecs[:,j]

        # Get Dynamics Equation
        d_eq = DE_MF_squ(op1, J, B, x_exp, y_exp, comm_x_op_exp, comm_y_op_exp, Q)

        # Fill Dynamics Matrix
        for j in range(15):
            op2 = gmm_squ[j]

            D[j,i] = np.trace(d_eq @ op2)/4

        # Solve system
        op_freqs, op_vecs = np.linalg.eig(D)

        # Get positive frequencies
        ids = op_freqs.argsort()[::-1]
        op_freqs = op_freqs[ids][:6]
        op_vecs = op_vecs[:, ids][:, :6]

        # Get norms to ground state
        op_norms = []
        for i in range(len(op_freqs)):
            op = np.zeros((4,4), dtype = complex)
            for j in range(15):
                op = op + op_vecs[j, i] * np.array(gmm_squ[j])
            
            op_norms.append(np.conjugate(op @ vecs[:,0]).T @ op @ vecs[:,0])

    return op_freqs, op_norms


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
