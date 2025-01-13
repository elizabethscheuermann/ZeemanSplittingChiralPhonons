import numpy as np
import scipy as sp


### DEFINE HAMILTONIAN MATRICES
H0 = np.array([
    [0, -1, -1],
    [-1, 0, -1],
    [-1, -1, 0]
])

x = np.array([
    [1, 0, 0],
    [0, -.5, 0],
    [0, 0, -.5]
])

y = np.array([
    [0, 0, 0],
    [0, np.sqrt(3)/2, 0],
    [0, 0, -np.sqrt(3)/2]
])

ml_1 =  np.array([np.exp(0 * (2*np.pi/3) * 1j), np.exp( 1 * (2*np.pi/3) * 1j), np.exp( 2 * (2*np.pi/3) * 1j)])/np.sqrt(3)
ml_n1 = np.array([np.exp(0 * (2*np.pi/3) * 1j), np.exp(-1 * (2*np.pi/3) * 1j), np.exp(-2 * (2*np.pi/3) * 1j)])/np.sqrt(3)
Lz = np.tensordot(ml_1, ml_n1, 0) - np.tensordot(ml_n1, ml_1, 0)

### DEFINE THERMODYNAMIC FUNCTIONS
q = 3 # Coordination Number
H_MF = lambda J, Ox, Oy, B : H0 + B * Lz + .5 * J * q * (Ox**2 + Oy**2) * np.identity(3) - J*q*(Ox * x + Oy * y) # Mean Field Hamiltonian
energies = lambda J, Ox, Oy, B : np.linalg.eigvals(H_MF(J, Ox, Oy, B))
Z = lambda T, J, Ox, Oy, B : np.trace(sp.linalg.expm(-H_MF(J, Ox, Oy, B)/T)) # Partition Function
F = lambda T, J, Ox, Oy, B : -T * np.log(Z(T, J, Ox, Oy, B)) if T!=0 else energies(J, Ox, Oy, B)[0] # Free Energy
F_opt = lambda O, J, B, T : -T * np.log(Z(T, J, O[0], O[1], B)) # Free Energy Optimization Function

print(Lz)
