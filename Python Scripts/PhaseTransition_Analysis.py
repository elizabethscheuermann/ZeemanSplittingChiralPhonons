import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import ThreeLevel_Base as three
import FourLevel_Base as four
import SixLevel_Base as six

def main():
    ### ZERO T FREE ENERGY
    def F3(O, J, B):
        return np.min(np.linalg.eigvals(three.H_MF(J, O[0], O[1], B)))
    
    def F4(O, J, B):
        return np.min(np.linalg.eigvals(four.H_MF(J, O[0], O[1], B)))
    
    def F6(O, J, B):
        return np.min(np.linalg.eigvals(six.H_MF(J, O[0], O[1], O[2], B)))
    
    ### PARAMETERS
    B0 = 0
    J_min = 1e-2; J_max = 2; J_n = 32; dJ = (J_max - J_min)/J_n; J_dom = np.linspace(J_min, J_max, J_n);

    ### DYNAMIC DOMAINS
    J3_dom = np.linspace(J_min, J_max, J_n)
    J4_dom = np.linspace(J_min, J_max, J_n)
    J6_dom = np.linspace(J_min, J_max, J_n)

    ### DATA ARRAYS
    O3_Arr = []
    O4_Arr = []
    O6_Arr = []

    dO3_max_Arr = []
    dO4_max_Arr = []
    dO6_max_Arr = []

    # Initial Minimization Geusses
    x03 = [1.5, 0]
    x04 = [1.5, 0]
    x06 = [3, 0, 0]

    ### MAIN LOOP
    iterations = 100
    bounds_3dom = []
    bounds_4dom = []
    bounds_6dom = []


    fig, ax = plt.subplots()
    threshhold = 1e-4
    step_size = .05
    for i in range(iterations):
        ### MINIMIZE FREE ENERGY
        for j in range(J_n):
            O3 = sp.optimize.minimize(F3, x03, args = (J3_dom[j], B0))
            O3_Arr.append(np.max(np.real(O3.x)))

            O4 = sp.optimize.minimize(F4, x04, args = (J4_dom[j], B0))
            O4_Arr.append(np.max(np.real(O4.x)))

            O6 = sp.optimize.minimize(F6, x06, args = (J6_dom[j], B0))
            O6_Arr.append(np.max(np.real(O6.x)))


        ax.legend()

        plt.pause(.1)
        ### TAKE DERIVATIVE
        dO3 = np.gradient(np.real(np.array(O3_Arr)))
        dO4 = np.gradient(np.real(np.array(O4_Arr)))
        dO6 = np.gradient(np.real(np.array(O6_Arr)))

        ### GET INDEX OF THRESHHOLD PASS
        i3 = [i for i in range(len(O3_Arr)) if O3_Arr[i] <= threshhold][-1]
        i4 = [i for i in range(len(O4_Arr)) if O4_Arr[i] <= threshhold][-1]
        i6 = [i for i in range(len(O6_Arr)) if O6_Arr[i] <= threshhold][-1]

        ax.cla()
        ax.plot(J3_dom, O3_Arr, label = '3')
        ax.plot(J4_dom, O4_Arr, label = '4')
        ax.plot(J6_dom, O6_Arr, label = '6')
        ax.plot([J_min, J_max], [threshhold, threshhold], linestyle = '--')

        J3_dom = np.linspace(.5 * (J3_dom[i3] + J3_dom[0]), .5 * (J3_dom[i3] + J3_dom[-1]), J_n)
        J4_dom = np.linspace(.5 * (J4_dom[i4] + J4_dom[0]), .5 * (J4_dom[i4] + J4_dom[-1]), J_n)
        J6_dom = np.linspace(.5 * (J6_dom[i6] + J6_dom[0]), .5 * (J6_dom[i6] + J6_dom[-1]), J_n)

        ### CLEAR OLD DOMAINS
        O3_Arr = []
        O4_Arr = []
        O6_Arr = []

        print(i3, i4, i6)

    ### PLOTTING
    fig, ax = plt.subplots()
    plt.xscale("log")
    plt.yscale("log")
    ax.set_xlabel(r"$\delta J$")

    ax.plot(bounds_3dom, dO3_max_Arr, label = r"$O_3$")
    ax.plot(bounds_4dom, dO4_max_Arr, label = r"$O_4$")
    ax.plot(bounds_6dom, dO6_max_Arr, label = r"$O_6$")

    ax.legend()

    plt.show()

main()
