import numpy as np
import matplotlib.pyplot as plt 
from FourLevel_Base import *




def main():
    Jmin = 1e-1
    Jmax = 2
    Jn = 2
    Jdom = np.linspace(Jmin, Jmax, Jn)

    Bmin = 0 
    Bmax = 5
    Bn = 2
    Bdom = np.linspace(Bmin, Bmax, Bn)

    Tmin = .1
    Tmax = 2
    Tn = 4
    Tdom = np.linspace(Tmin, Tmax, Tn)


    boundaries = {}
    low_lim = 1e-3
    up_lim = 5e-1
    x0 = [1.5, 0]
    for t in range(Tn):
        print(t)
        boundaries[t] = []

        for b in range(Bn):
            row = []
            for j in range(Jn):
                O = sp.optimize.minimize(F_opt, x0, args = (Jdom[j], Bdom[b], Tdom[t]))
                row.append(O.x[0])
            grad = np.gradient(row)
            index = np.argmax(grad)
            boundaries[t].append([Jdom[index], Bdom[b]])
            

    fig, ax = plt.subplots()
    ax.set_xlabel("Interaction Strength [J]")
    ax.set_ylabel("Magnetic Field Strength [B]")

    for t in range(Tn):
        data = np.array(boundaries[t])
        data = data[data[:, 1].argsort()]
        ax.scatter(data[:,0], data[:,1], marker = '.', label = "T = " + str(Tdom[t]))

    ax.legend()
    plt.show()


    J = .1
    T = 1e-1
    Bdom = np.linspace(0, 5, 128)
    eigval_arr = []
    for b in Bdom:
        O = sp.optimize.minimize(F_opt, x0, args = (J, b, T))

        eigenvals, eigenvecs = np.linalg.eigh(H_MF(J, O.x[0], O.x[1], b))
        eigval_arr.append(eigenvals)

    fig2, ax2 = plt.subplots()
    ax2.set_xlabel("Magnetic Field Strength[B]")
    ax2.set_ylabel("Energy [E]")
    eigval_arr = np.array(eigval_arr)
    ax2.plot(Bdom, eigval_arr[:, 0], color = 'black')
    ax2.plot(Bdom, eigval_arr[:, 1], color = 'black')
    ax2.plot(Bdom, eigval_arr[:, 2], color = 'black')
    ax2.plot(Bdom, eigval_arr[:, 3], color = 'black')

    plt.show()
        


main()
