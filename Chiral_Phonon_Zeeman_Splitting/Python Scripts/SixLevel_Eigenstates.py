import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from SixLevel_Base import *

### MAIN LOOP
def main():
    ### PARAMETER DOMAINS
    J0 = .1
    B_min = 0; B_max = 5; B_n = 32; dB = (B_max - B_min)/B_n; B_dom = np.linspace(B_min, B_max, B_n);
    theta = 0
    phi = 0
    T0 = .01


    ### DATA STORAGE ARRAYS
    energies = []

    # Initial Minimization Guess
    x0 = [1e-2, 0, 0]

    ### MAIN LOOP
    for b in range(len(B_dom)):
        ### MINIMIZE FREE ENERGY W.R.T. ORDER PARAMETERS
        O = sp.optimize.minimize(F_opt, x0, args = (J0, 0, 0, B_dom[b], T0))

        ### GET EIGENVALUES
        eigenvals, eigenvecs = np.linalg.eigh(H_MF(O.x[0], O.x[1], O.x[2], J0, B_dom[b]*np.sin(theta), B_dom[b]*np.sin(theta) * np.sin(phi), B_dom[b]))
        energies.append(eigenvals)

    fig, ax = plt.subplots()
    ax.set_xlabel("Magnetic Field Strength [B]")
    ax.set_ylabel("Eigenstate Energy")

    energies = np.array(energies)
    
    print(energies.shape)
    for j in range(energies.shape[1]):
        ax.plot(B_dom, energies[:,j], color = 'black')

    plt.show()

main()


