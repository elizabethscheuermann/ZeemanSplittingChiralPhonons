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
B_min = 0.1; B_max = 3; B_n = 64; B_dom = np.linspace(B_min, B_max, B_n)
T0 = 0.1
J0 = .25
dim = 4


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
   
    ### SORT AND SELECT POSITIVE MOST FREQUENCIES
    indices = eig_freqs.argsort()[::-1]
    eig_freqs = eig_freqs[indices[:6]]
    coef_vecs = coef_vecs[:, indices[:6]]

    #print(B,np.round(M,4))

    ### GET VALUES CONNECTED TO GROUND
    transition_elems.append([])
    eig_vals_arr.append([])
    
    i = 0
    while i < len(eig_freqs):
        ### GET OPERATOR AND NORM TO GROUND STATE
        eigen_operator = GeneralizedGellMannOperator(dim, coef_vecs[:, i])
        operator_state = eigen_operator @ eig_vecs[:, 0]
        norm = np.conjugate(operator_state.T) @ operator_state

        ### IF NORM PASSES THRESHHOLD ADD
        if norm > 1e-3:
            transition_elems[-1].append(norm)
            eig_vals_arr[-1].append(eig_freqs[i])

        i+=1
   
    print(len(eig_vals_arr[-1]))
    

    #s1 = GeneralizedGellMannOperator(dim, coef_vecs[:, 2]) @ eig_vecs[:, 0]
    #s2 = GeneralizedGellMannOperator(dim, coef_vecs[:, 3]) @ eig_vecs[:, 0]

   
### SWITCH TO NP ARRAYS
eig_vals_arr = np.array(eig_vals_arr)
transition_elems = np.array(transition_elems)

### PLOTTING
# Eigen frequencies plot
freq_fig, freq_ax = plt.subplots()
freq_ax.set_ylabel("Eigen frequencies")
freq_ax.set_xlabel("Magnetic Field Strength B/t")
freq_ax.set_ylim([0, 2.5])

# Operator Norms
op_norms_fig, op_norms_ax = plt.subplots()
op_norms_ax.set_xlabel("Magnetic Field Strength B/t")
op_norms_ax.set_ylabel(r"$\langle 0 | \Lambda^\dagger \Lambda | 0 \rangle$")

for i in range(3):
    freq_ax.plot(B_dom, eig_vals_arr[:, i], label = f"$\omega_{{{i}}}$")

freq_ax.legend()


for i in range(3):
    op_norms_ax.scatter(B_dom, transition_elems[:,i], label = "Op." + str(i))

#op_norms_ax.legend()


plt.show()
#np.savetxt("testing.csv", M)
