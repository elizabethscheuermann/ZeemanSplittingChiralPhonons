import FourLevel_Base
import scipy as sp

### DEFINE GENERALIZED GELL MANN MATRICES

Lx12 = np.array([[0,1,0,0],
                 [1,0,0,0],
                 [0,0,0,0],
                 [0,0,0,0]])

Lx13 = np.array([[0,0,1,0],
                 [0,0,0,0],
                 [1,0,0,0],
                 [0,0,0,0]])

Lx14 = np.array([[0,0,0,1],
                 [0,0,0,0],
                 [0,0,0,0],
                 [1,0,0,0]])

Lx23 = np.array([[0,0,0,0],
                 [0,0,1,0],
                 [0,1,0,0],
                 [0,0,0,0]])

Lx24 = np.array([[0,0,0,0],
                 [0,0,0,1],
                 [0,0,0,0],
                 [0,1,0,0]])

Lx34 = np.array([[0,0,0,0],
                 [0,0,0,0],
                 [0,0,0,1],
                 [0,0,1,0]])

Ly12 = np.array([[0,1j,0,0],
               [-1j,0,0,0],
               [0,0,0,0],
               [0,0,0,0]])

Ly13 = np.array([[0,0,1j,0],
                 [0,0,0,0],
                [-1j,0,0,0],
                [0,0,0,0]])

Ly14 = np.array([[0,0,0,1j],
                 [0,0,0,0],
                [0,0,0,0],
                [-1j,0,0,0]])

Ly23 = np.array([[0,0,0,0],
                 [0,0,1j,0],
                [0,-1j,0,0],
                [0,0,0,0]])

Ly24 = np.array([0,0,0,0],
                [0,0,0,1j],
                [0,0,0,0],
                [0,-1j,0,0]])

Ly34 = np.array([[0,0,0,0],
                 [0,0,0,0],
                 [0,0,0,1j],
                [0,0,-1j,0])

Lz1 = np.array([[1,0,0,0],
               [0,-1,0,0],
               [0,0,0,0],
               [0,0,0,0]])

Lz2 = np.array([[1,0,0,0],
                [0,1,0,0],
               [0,0,-2,0],
               [0,0,0,0]])/np.sqrt(3)

Lz3 = np.array([[1,0,0,0],
                [0,1,0,0],
               [0,0,1,0],
               [0,0,0,-3]])/np.sqrt(6)

matrices = [Lx12, Lx13, Lx14, Lx23, Lx24, Lx34, Ly12, Ly13, Ly14, Ly23, Ly24, Ly34, Lz1, Lz2, Lz3]

def Operator(coefs):
    if len(coefs) != 16:
        print("There must be 16 coefficients")
        return
    O = np.zeros((4,4))
    for i in range(16):
        O = O + coefs[i] * matrices[i]

    return O

def Commutator(Op1, Op2):
    return Op1 @ Op2 - Op2 @ Op1


parser = argparse.ArgumentParser()
parser.add_argument("-Bmin", type = float)
parser.add_argument("-Bmax", type = float)
parser.add_argument("-Bn", type = int)
parser.add_argument("-T", type = float)
parser.add_argument("-J", type = float)


args = parser.parse_args()


def main():
    # Parameters
    J = args.J
    T = args.T
    Bmin = args.Bmin
    Bmax = args.Bmax
    Bn = args.Bn
    dB =(Bmax - Bmin)/Bn
    B_dom = np.linspace(Bmin, Bmax, Bn)


    ### SOLVE MF HAMILTONIAN
    for b in range(len(B_dom)):
        # Minimize Free Energy
        O = sp.optimize.minimize(F_opt, x0, args = (J, T, B_dom[b]))
        
        # Get Eigenstates
        eigenvals, eigenvecs = np.linalg.eigh(H_MF(J, O.x[0], O.x[1], B_dom[b]))

        # Get State Probabilities
        probs = []
        for v in range(len(eigenvals)):
            probs[v] = np.abs(np.exp(-eigenvals[v]/T))/Z(T, J,O.x[0], O.x[1], B_dom[b])


        # Get Position Expectation values
        exp_x = 0
        exp_y = 0
        for v in range(len(eigenvals)):
            exp_x = exp_x + probs[v] * np.conj(eigenvecs[:, v]).T @ x @ eigenvecs[:,v]
            exp_y = exp_y + probs[v] * np.conj(eigenvecs[:, v]).T @ y @ eigenvecs[:,v]
    
        # Get Commutator Expectation Values
        exp_x_comm = exp_x_comm + probs[v] *np.conj(eigenvecs[:,v]).T @ Commutator(x, )
