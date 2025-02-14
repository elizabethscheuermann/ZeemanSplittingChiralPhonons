from Base import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

colors = mcolors.BASE_COLORS.values()

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


### MAIN LOOP
B_min = 0.1; B_max = 3; B_n = 128; B_dom = np.linspace(B_min, B_max, B_n)
T0 = 0.1
J0 = .25
dim = 4

# Eigen frequencies
fig2, ax2 = plt.subplots()
ax2.set_ylabel("Eigen frequencies")
ax2.set_xlabel("Magnetic Field Strength B/t")
ax2.set_ylim([-2.5, 2.5])

# Operator on ground state
fig3, ax3 = plt.subplots()
ax3.set_xlabel("Magnetic Field Strength B/t")
ax3.set_ylabel(r"$\langle 0 | \Lambda^\dagger \Lambda | 0 \rangle$")




Q = 4

### DATA ARRAYS
Ox_arr = []
eig_vals_arr = []
transition_elems = []
x_exp_arr = []
comm_x_op_exp_arr = []
comm_y_op_exp_arr = []

for b, B in enumerate(B_dom):
    ### OPTIMIZE FREE ENERGY
    O = sp.optimize.minimize(F_MF_squ, x0_squ, args = (J0, B, T0))
    #ax.scatter(B, np.max(O.x))

    ### GET STATES AND STATE PROBABILITIES 
    eig_vals, eig_vecs = np.linalg.eigh(H_MF_squ(O.x, J0, B))
    indices = eig_vals.argsort()
    eig_vals = eig_vals[indices]
    eig_vecs = eig_vecs[:, indices]

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
            M[j,i] = .25 * np.trace(d_eq @ operator)

    ### GET EIGENFREQUENCIES and EIGENVECS
    eig_freqs, coef_vecs = np.linalg.eig(M)
   
    ### SORT
    indices = eig_freqs.argsort()[::-1]
    eig_freqs = eig_freqs[indices]
    coef_vecs = coef_vecs[:, indices]

    ### GET VALUES CONNECTED TO GROUND
    eig_vals_arr.append(eig_freqs)
    transition_elems.append([])

    i = 0
    while i < len(eig_freqs):
        ### GET OPERATOR TRANSITION ELEMENT
        eigen_operator = GeneralizedGellMannOperator(dim, coef_vecs[:, i])
        operator_state = eigen_operator @ eig_vecs[:, 0]
        
        ### CHECK FOR DEGENERACY
        norm = np.conj(operator_state.T) @ operator_state
        j = 1
        if i < 14: 
            while (np.abs(eig_freqs[i] - eig_freqs[i+j]) < 1e-5): 
                eigen_operator = GeneralizedGellMannOperator(dim, coef_vecs[:,j])
                operator_state = eigen_operator @ eig_vecs[:, 0]
                norm += np.conj(operator_state.T) @ operator_state
                j+=1
        transition_elems[-1].append(norm)
        i+=j
eig_vals_arr = np.array(eig_vals_arr)
transition_elems = np.array(transition_elems)

for i in range(15):
    ax2.scatter(B_dom, eig_vals_arr[:, i], color = 'black')


for i in range(13):
    ax3.scatter(B_dom, transition_elems[:, i])


#eig_vals_arr = np.array(eig_vals_arr)
# comm_x_op_exp_arr = np.abs(np.array(comm_x_op_exp_arr))
plt.show()
