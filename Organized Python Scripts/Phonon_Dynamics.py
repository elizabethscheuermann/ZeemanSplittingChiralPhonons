from Base import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


### MAIN LOOP
T = .1
J = .25


Q = 4

### DATA ARRAYS
eig_vals_arr = []
norms_arr = []
def Test(J0, B, T0, Q, dim):
    ### OPTIMIZE FREE ENERGY
    O = sp.optimize.minimize(F_MF_squ, x0_squ, args = (J, B, T))

    ### GET STATES AND STATE PROBABILITIES 
    vals, vecs = np.linalg.eigh(H_MF_squ(O.x, J, B))
    ids = vals.argsort()
    vals = vals[ids]
    vecs = vecs[:, ids]

    probs = np.exp(-1 * vals/T0)/Z_MF_squ(O.x, J, B, T) if T0!=0 else [1,0,0,0]

    ### EVALUTATE <x>, <y>
    x_exp = 0
    y_exp = 0
    for i in range(4):      
        x_exp += probs[i] * np.real(np.conj(vecs[:,i]).T @ x_squ @ vecs[:, i])
        y_exp += probs[i] * np.real(np.conj(vecs[:,i]).T @ y_squ @ vecs[:, i])
   
    ### DYNAMICS MATRIX
    D = np.zeros((15, 15), dtype = complex)

    ### LOOP OVER MATRICES
    for i in range(15):
        ### GET OPERATOR
        op1 = gmm_squ[i]

        ### EVALUATE <[x, Lambda]>, <[y, Lambda]>
        comm_x_op_exp = 0
        comm_y_op_exp = 0
        for j in range(dim):
            comm_x_op_exp += probs[j] * np.conj(vecs[:,j].T) @ (x_squ @ op1 - op1 @ x_squ) @ vecs[:,j]
            comm_y_op_exp += probs[j] * np.conj(vecs[:,j].T) @ (y_squ @ op1 - op1 @ y_squ) @ vecs[:,j]

        ### GET DYNAMICS EQ
        d_eq = DE_MF_squ(op1, J, B, x_exp, y_exp, comm_x_op_exp, comm_y_op_exp, Q)
        
        ### LOOP OVER SECOND MATRIX
        for j in range(15):
            ### GET OPERATOR
            op2 = gmm_squ[j]
            ### GET MATRIX ELEMENT
            D[j,i] = np.trace(d_eq @ op2)/4

    ### GET EIGENFREQUENCIES and EIGENVECS
    op_freqs, op_vecs = np.linalg.eig(D)
    ### GET ONLY POSITIVE FREQUENCIES
    ids = op_freqs.argsort()[::-1]
    op_freqs = op_freqs[ids][:6]
    op_vecs = op_vecs[:, ids][:, :6]
   
    ### SORT BY TRANSITION AMPLITUDE TO GROUND
    norms = []
    
    for i in range(len(op_freqs)):
        ### GET OPERATOR AND NORM TO GROUND STATE
        op = GeneralizedGellMannMatrix(4, op_vecs[:, i])
        norms.append(np.conj(op @ vecs[:,0]).T @ op @ vecs[:,0])
    
    #norms_arr.append(norms)
    #norm_indices = np.array(norms).argsort()[::-1]
    #eig_freqs = eig_freqs[norm_indices]

            #operator = GeneralizedGellMannOperator(dim, coefs)
    #eig_vals_arr.append(eig_freqs)
    
    return op_freqs, norms
   
### SWITCH TO NP ARRAYS
#eig_vals_arr = np.array(eig_vals_arr)
#norms_arr = np.array(norms_arr)


#print(eig_vals_arr)

### PLOTTING
# Eigen frequencies plot
#fig, ax = plt.subplots(2)
#ax[0].set_ylabel("Eigen frequencies")
#ax[0].set_xlabel("Magnetic Field Strength B/t")
#
#
#for i in range(6):
#    ax[0].scatter(B_dom, eig_vals_arr[:, i], label = f"$\omega_{{{i}}}$")
#    ax[1].scatter(B_dom, norms_arr[:, i], label = i)
#ax[0].legend()
#ax[1].legend()




#plt.show()
