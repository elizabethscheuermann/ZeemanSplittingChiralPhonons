from Base import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def GeneralizedGellMannOperator(dim, coefs):
    if len(coefs) != dim**2 - 1:
        print(r"Must have $n^2 -1$ coefficients")
    else:
        operator = np.zeros((dim, dim), dtype = complex)

        ix, iy = np.triu_indices(dim, k=1)
        K = int(.5*dim*(dim-1))
        operator[ix, iy] = coefs[0:K] + 1j*coefs[K:2*K]
        operator[iy, ix] = coefs[0:K] - 1j*coefs[K:2*K]
        for i in range(dim - 1):
            operator[i+1,i+1] -= np.sqrt(2/((i+1)*(i+2)))*coefs[2*K + i]*(i + 1)
            for j in range(i+1):
                operator[j][j] += np.sqrt(2/((i+1)*(i+2)))*coefs[2*K + i]

    return operator

   
def Commutator(op1, op2):
    return op1@op2 - op2@op1

def DE_squ(operator, J, B, x_exp, y_exp, comm_x_op_exp, comm_y_op_exp, Q):
    # Get commutators
    h0_comm = h0_squ @ operator - operator @ h0_squ
    Lz_comm = Lz_squ @ operator - operator @ Lz_squ
    x_comm = x_squ @ operator - operator @ x_squ
    y_comm = y_squ @ operator - operator @ y_squ

    # Dynamics equation
    d_eq = h0_comm + B * Lz_comm - J * q_squ * (x_exp * x_comm + y_exp * y_comm) - J * Q * (comm_x_op_exp * x_squ + comm_y_op_exp * y_squ)

    return d_eq

for j in range(15):
    coefs = np.zeros((15))
    coefs[j] = 1
    print(GeneralizedGellMannOperator(4, coefs))

### MAIN LOOP
B_min = 0.1; B_max = 5; B_n = 128; B_dom = np.linspace(B_min, B_max, B_n)
T0 = 0.1
J0 = .25
dim = 4

fig, ax = plt.subplots()
ax.set_ylim([0,3])
ax.set_xlabel("Magnetic Field Strength B/t")
ax.set_ylabel(r"Operator Eigenfrequency $\omega$")

Q = 4

### DATA ARRAYS
Ox_arr = []
x_exp_arr = []
eig_vals_arr = []
comm_x_op_exp_arr = []
comm_y_op_exp_arr = []


for b, B in enumerate(B_dom):
    ### OPTIMIZE FREE ENERGY
    O = sp.optimize.minimize(F_MF_squ, x0_squ, args = (J0, B, T0))
    #ax.scatter(B, np.max(O.x))

    ### GET STATES AND STATE PROBABILITIES 
    eig_vals, eig_vecs = np.linalg.eigh(H_MF_squ(O.x, J0, B))
    Z = np.sum(np.exp(-eig_vals/T0))
    probs = P_MF_squ(O.x, J0, B,T0)

    ### EVALUTATE <x>, <y>
    x_exp = 0
    y_exp = 0
    for i in range(4):      
        x_exp += probs[i] * np.abs(np.conj(eig_vecs[:,i]).T @ x_squ @ eig_vecs[:, i])
        y_exp += probs[i] * np.abs(np.conj(eig_vecs[:,i]).T @ y_squ @ eig_vecs[:, i])
   

    Ox_arr.append(np.max(O.x))
    x_exp_arr.append(x_exp)


    ### DYNAMICS MATRIX
    M = np.zeros((dim**2 - 1, dim**2 - 1), dtype = complex)

    comm_x_op_exp_arr.append([])
    comm_y_op_exp_arr.append([])

    ### LOOP OVER MATRICES
    for i in range(dim**2 - 1):
        ### GET OPERATOR
        coefs = np.zeros((dim**2 - 1))
        coefs[i] = 1
        operator = GeneralizedGellMannOperator(dim, coefs)

        ### EVALUATE <[x, Lambda]>, <[y, Lambda]>
        comm_x_op_exp = 0
        comm_y_op_exp = 0
        for j in range(4):
            comm_x_op_exp += probs[j] * np.conjugate(eig_vecs[:,j].T) @ (x_squ @ operator - operator @ x_squ) @ eig_vecs[:,j]
            comm_y_op_exp += probs[j] * np.conjugate(eig_vecs[:,j].T) @ (y_squ @ operator - operator @ y_squ) @ eig_vecs[:,j]
        comm_x_op_exp_arr[-1].append(comm_x_op_exp)
        comm_y_op_exp_arr[-1].append(comm_y_op_exp)

        ### GET DYNAMICS EQ
        d_eq = DE_squ(operator, J0, B, np.max(O.x), np.min(O.x), comm_x_op_exp, comm_y_op_exp, Q)
        ### LOOP OVER SECOND MATRIX
        for j in range(dim**2 - 1):
            ### GET OPERATOR
            coefs = np.zeros((dim**2 - 1))
            coefs[j] = 1
            operator = GeneralizedGellMannOperator(dim, coefs)

            ### GET MATRIX ELEMENT
            M[i,j] = .25 * np.trace(d_eq @ operator)


    ### GET EIGENFREQUENCIES
    vals = np.sort(np.linalg.eigvals(M))
    eig_vals_arr.append(vals)

eig_vals_arr = np.array(eig_vals_arr)
comm_x_op_exp_arr = np.abs(np.array(comm_x_op_exp_arr))
ax.plot(B_dom, Ox_arr, label = 'Ox')
ax.plot(B_dom, x_exp_arr, label = '<x>')
for i in range(15):
    ax.scatter(B_dom, eig_vals_arr[:,i], color = 'black')
ax.legend()
plt.show()
