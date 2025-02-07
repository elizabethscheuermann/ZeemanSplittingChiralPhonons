import numpy as np
from Base import *


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

def DynamicsEquationMatrix(dim, J, B, x_exp, y_exp, comm_x_op_exp, comm_y_op_exp):
    # Handle Dimension
    if dim == 3:
        h0 = h0_tri
        Lz = Lz_tri
        x = x_tri
        y = y_tri
    elif dim == 4:
        h0 = h0_squ
        Lz = Lz_squ
        x = x_squ
        y = y_squ

    for i in range(dim**2 -1):
        for j in range(dim**2 - 1):
            # Get Operator
            coefs = np.zeros((dim**2 - 1))
            coefs[j] = 1
            Lambda_j = GeneralizedGellMannOperator(dim, coefs)
            
            # Get Commutators
            h0_comm = h0 @ Lambda_j - Lambda_j @ h0
            Lz_comm = Lz @ Lambda_j - Lambda_j @ Lz
            x_comm = x @ Lambda_j - Lambda_j @ x
            y_comm = y @ Lambda_j - Lambda_j @ y
            
            # Get Dynamics Equation
            d_eq = h0_comm + B * Lz_comm - J * q * (x_exp * x_comm + y_exp * y_comm) - J * Q * (comm_x_op_exp * x + comm_y_op_exp * y)

            # Get Second Operator
            coefs = np.zeros((dim**2 - 1))
            coefs[i] = 1
            Lambda_i = GeneralizedGellMannOperator(dim, coefs)

            # Get Matrix Element
            M[i,j] = np.trace(d_eq @ Lambda_i)
