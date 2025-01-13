import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as mpl_cols
import scipy as sp
import ThreeLevel_Base as tri
import FourLevel_Base as squ
import SixLevel_Base as cub


def PlotChange(dJ_dom, dOx_Arr_3, dOx_Arr_4):
    fig, ax = plt.subplots()
    ax.scatter(dJ_dom, dOx_Arr_3, label = "3 Level System")
    ax.scatter(dJ_dom, dOx_Arr_4, label = "4 Level System")
    ax.legend()
    ax.set_xlabel("Change in interaction strength dJ")
    ax.set_ylabel("Change in order parameter d<x>")

    plt.show()

def PlotOx(J_dom_3, J_dom_4, Ox_Arr_3, Ox_Arr_4):
    fig, ax = plt.subplots(2)
    ax[0].scatter(J_dom_3, Ox_Arr_3, label = "3 Level System")
    ax[0].set_xlabel("Interaction Strength J")
    ax[0].set_ylabel("Order Parameter <x>")
    ax[0].set_title("3 Level System")


    ax[1].scatter(J_dom_4, Ox_Arr_4, label = "4 Level System")
    ax[1].set_xlabel("Interaction Strength J")
    ax[1].set_ylabel("Order Parameter <x>")
    ax[1].set_title("4 Level System")

    plt.show()

def main():
    # Parameters
    B = .25
    T = 1e-2

    # Interaction Strength Domain

    J_min = 1e-2; J_max = 1; J_n = 16;
    J3_dom = np.linspace(J_min, J_max, J_n)
    J4_dom = np.linspace(J_min, J_max, J_n)
    J6_dom = np.linspace(J_min, J_max, J_n)

    num_iterations = 20
    # Data Arrays
    Ox_Arr_3 = np.zeros((J_n))
    Ox_Arr_4 = np.zeros((J_n))
    Ox_Arr_6 = np.zeros((J_n))

    dOx3_Arr = []
    dOx4_Arr = []
    dOx6_Arr = []
    dJ3_dom = []
    dJ4_dom = []
    dJ6_dom = []

    # Initial Minimizatoin Guess
    x0 = [1.5, 0]
    x03 = [1.5,0,0]
    for i in range(num_iterations):
        print(i)
        for j in range(J_n):
            # 3 Level System
            Ox_3 = sp.optimize.minimize(tri.F_opt, x0, args = (J3_dom[j], B, T))
            Ox_Arr_3[j] = np.abs(Ox_3.x[0])
        
            # 4 Level System
            Ox_4 = sp.optimize.minimize(squ.F_opt, x0, args = (J4_dom[j], B, T))
            Ox_Arr_4[j] = np.abs(Ox_4.x[0])

            # 6 Level System
            Ox_6 = sp.optimize.minimize(cub.F_opt, x03, args = (J6_dom[j], 0, 0, B, T))
            Ox_Arr_6[j] = np.abs(Ox_6.x[0])
        
        # Get maximum gradients
        Ox_3_grad = np.gradient(Ox_Arr_3)
        Ox_4_grad = np.gradient(Ox_Arr_4)
        Ox_6_grad = np.gradient(Ox_Arr_6)
        
        i3 = Ox_3_grad.argmax()
        i4 = Ox_4_grad.argmax()
        i6 = Ox_6_grad.argmax()
    

        # New Domains
        J3_dom = np.linspace(.5*(J3_dom[i3] + J3_dom[0]), .5*(J3_dom[-1] + J3_dom[i3]), J_n)
        J4_dom = np.linspace(.5*(J4_dom[i4] + J4_dom[0]), .5*(J4_dom[-1] + J4_dom[i4]), J_n)
        J6_dom = np.linspace(.5*(J6_dom[i6] + J6_dom[0]), .5*(J6_dom[-1] + J6_dom[i6]), J_n)

        # Save data
        dOx3_Arr.append(Ox_3_grad[i3])
        dOx4_Arr.append(Ox_4_grad[i4])
        dOx6_Arr.append(Ox_6_grad[i6])

        dJ3_dom.append(J3_dom[-1] - J3_dom[0])
        dJ4_dom.append(J4_dom[-1] - J4_dom[0])
        dJ6_dom.append(J6_dom[-1] - J6_dom[0])


    fig, ax = plt.subplots()

    ax.set_xlabel(r"Change in Interaction Strength $J_C - \delta J$")
    ax.set_ylabel(r"Change in Order Parameter $\delta \langle x \rangle$")
    ax.loglog(dJ3_dom, dOx3_Arr, label = "Triagonal")
    ax.loglog(dJ4_dom, dOx4_Arr, label = "Square")
    ax.loglog(dJ6_dom, dOx6_Arr, label = "Cubic")
    ax.legend()
    plt.show()

main()
