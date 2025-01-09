import numpy as np
import scipy as sp


### DEFINE HAMILTONIAN MATRICES
H0 = np.array([
    [0, -1, -1, 0, -1, -1],
    [-1, 0, -1, -1, 0, -1],
    [-1, -1, 0, -1, -1, 0],
    [0, -1, -1, 0, -1, -1],
    [-1, 0, -1, -1, 0, -1],
    [-1, -1, 0, -1, -1, 0]
])

x = np.array([
    [1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])

y = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -1, 0],
    [0, 0, 0, 0, 0, 0]
])

z = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0 ,0 ,0],
    [0, 0, 0, 0, 0 ,0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, -1]
])


ml_x_1 = np.array([0,
                   np.exp(2*np.pi/4*0j),
                   np.exp(2*np.pi/4*1j),
                   0,
                   np.exp(2*np.pi/4*2j),
                   np.exp(2*np.pi/4*3j)])/np.sqrt(4)

ml_x_n1 = np.array([0,
                    np.exp(2*np.pi/4*0j),
                    np.exp(2*np.pi/4*-1j),
                    0,
                    np.exp(2*np.pi/4*-2j),
                    np.exp(2*np.pi/4*-3j)])/np.sqrt(4)

ml_y_1 = np.array([np.exp(2*np.pi/4*0j),
                   0,
                   np.exp(2*np.pi/4*1j),
                   np.exp(2*np.pi/4*2j),
                   0,
                   np.exp(2*np.pi/4*3j)])/np.sqrt(4)

ml_y_n1 = np.array([np.exp(2*np.pi/4*0j),
                    0,
                    np.exp(2*np.pi/4*-1j),
                    np.exp(2*np.pi/4*-2j),
                    0,
                    np.exp(2*np.pi/4*-3j)])/np.sqrt(4)

ml_z_1 = np.array([np.exp(2*np.pi/4*0j),
                   np.exp(2*np.pi/4*1j),
                   0,
                   np.exp(2*np.pi/4*2j),
                   np.exp(2*np.pi/4*3j),
                   0])/np.sqrt(4)

ml_z_n1 = np.array([np.exp(2*np.pi/4*0j),
                    np.exp(2*np.pi/4*-1j),
                    0,
                    np.exp(2*np.pi/4*-2j),
                    np.exp(2*np.pi/4*-3j),
                    0])/np.sqrt(4)

Lx = np.tensordot(ml_x_1, ml_x_n1, 0) - np.tensordot(ml_x_n1, ml_x_1, 0)
Ly = np.tensordot(ml_y_1, ml_y_n1, 0) - np.tensordot(ml_y_n1, ml_y_1, 0)
Lz = np.tensordot(ml_z_1, ml_z_n1, 0) - np.tensordot(ml_z_n1, ml_z_1, 0)

### DEFINE THERMODYNAMIC FUNCTIONS
q = 6 # Coordination Number
H_MF = lambda Ox, Oy, Oz, J, Bx, By, Bz : H0 + Bx * Lx + By * Ly + Bz * Lz + .5 * J * q * (Ox**2 + Oy**2 + Oz**2) * np.identity(6) - J*q*(Ox * x + Oy * y + Oz * z) # Mean Field Hamiltonian
Z = lambda T, Ox, Oy, Oz, J, Bx, By, Bz : np.trace(sp.linalg.expm(-H_MF(Ox, Oy, Oz, J, Bx, By, Bz)/T)) # Partition Function
F = lambda T, Ox, Oy, Oz, J, Bx, By, Bz : -T * np.log(Z(T, Ox, Oy, Oz, J, Bx, By, Bz)) # Free Energy
F_opt = lambda O, J, Bx, By, Bz, T : -T * np.log(Z(T, O[0], O[1], O[2], J, Bx, By, Bz)) # Free Energy Optimization Function

