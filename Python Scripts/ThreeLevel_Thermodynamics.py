import numpy as np
import scipy as sp
import argparse

from ThreeLevel_Base import *

parser = argparse.ArgumentParser()
parser.add_argument('-Bmin', type = float)
parser.add_argument('-Bmax', type = float)
parser.add_argument('-Bn', type = int)

parser.add_argument('-Jmin', type = float)
parser.add_argument('-Jmax', type = float)
parser.add_argument('-Jn', type = int)

parser.add_argument('-Tmin', type = float)
parser.add_argument('-Tmax', type = float)
parser.add_argument('-Tn', type = int)

args = parser.parse_args()


### MAIN LOOP
def main():
    ### PARAMETER DOMAINS
    J_min = args.Jmin
    J_max = args.Jmax
    J_n = args.Jn
    dJ = (J_max - J_min)/J_n
    J_dom = np.linspace(J_min, J_max, J_n)

    B_min = args.Bmin
    B_max = args.Bmax
    B_n = args.Bn
    dB = (B_max - B_min)/B_n
    B_dom = np.linspace(B_min, B_max, B_n)

    T_min = args.Tmin
    T_max = args.Tmax
    T_n = args.Tn
    dT = (T_max - T_min)/T_n
    T_dom = np.linspace(T_min, T_max, T_n)

    ### DATA STORAGE ARRAYS
    Ox_Arr = np.zeros((J_n, B_n, T_n))
    Oy_Arr = np.zeros((J_n, B_n, T_n))
    probs_Arr = np.zeros((J_n, B_n, T_n, 3))
    M_Arr = np.zeros((J_n, B_n, T_n))
    chi_Arr = np.zeros((J_n, B_n, T_n))

    # Initial Minimization Guess
    x0 = [1.5, 0]

    ### MAIN LOOP
    for j in range(len(J_dom)):
        for b in range(len(B_dom)):
            for t in range(len(T_dom)):
                ### MINIMIZE FREE ENERGY W.R.T. ORDER PARAMETERS
                O = sp.optimize.minimize(F_opt, x0, args = (J_dom[j], B_dom[b], T_dom[t]))
                Ox_Arr[j, b, t] = np.max(np.abs(O.x))
                Oy_Arr[j, b, t] = np.min(np.abs(O.x))

                ### GET EIGENSTATES
                eigenvals, eigenvecs = np.linalg.eigh(H_MF(J_dom[j], O.x[0], O.x[1], B_dom[b]))

                ### GET STATE PROBABILITIES
                for v in range(len(eigenvals)):
                    probs_Arr[j, b, t, v] = np.abs(np.exp(-eigenvals[v]/T_dom[t])/Z(T_dom[t], J_dom[j], O.x[0], O.x[1], B_dom[b]))

                ### GET MAGNETIZATION
                m = 0
                for v in range(len(eigenvals)):
                    m += probs_Arr[j, b, t, v] * eigenvecs[:,v].conj().T @ Lz @ eigenvecs[:,v]
                
                M_Arr[j, b, t] = np.abs(m)

                ### GET MAGNETIC SUSCEPTIBILITY
                chi_Arr[j, b, t] = (M_Arr[j, b, t] - M_Arr[j, b-1, t])/dB

    ### FIX MAGNETIC SUSCEPTIBILITY BOUNDARY
    for j in range(len(J_dom)):
        for t in range(len(T_dom)):
            chi_Arr[j, 0, t] = (M_Arr[j][1][t]- M_Arr[j][0][t])/dB

    ### CREATE CSV
    with open("/home/twiggy/Research/Chiral_Phonon_Zeeman_Splitting/Data/ThreeLevel_Thermodynamics.csv", "w") as file:
        file.write("J, B, T, Ox, Oy, p1, p2, p3, m, chi")
        file.write('\n')
        for j in range(J_n):
            for b in range(B_n):
                for t in range(T_n):
                    text = str(J_dom[j]) + ", " + str(B_dom[b]) + ", " + str(T_dom[t]) + ", " 
                    text += str(Ox_Arr[j,b,t]) + ", " + str(Oy_Arr[j, b, t]) + ", "
                    text += str(probs_Arr[j, b, t, 0]) + ", " + str(probs_Arr[j, b, t, 1]) + ", " + str(probs_Arr[j, b, t, 2]) + ", "
                    text += str(M_Arr[j,b,t]) + ", " + str(chi_Arr[j,b,t])
                    file.write(text)
                    file.write('\n')


main()


