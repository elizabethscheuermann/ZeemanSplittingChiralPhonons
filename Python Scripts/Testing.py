import numpy as np
import matplotlib.pyplot as plt
import SixLevel_Base as six
import ThreeLevel_Base as three
import FourLevel_Base as four
import scipy as sp

J_n = 100
J_dom = np.linspace(.962, .963, J_n)

### DATA ARRAYS
O3_Arr = []
O4_Arr = []
O6_Arr = []

# Initial Minimization Geusses
x03 = [1.5, 0]
x04 = [1.5, 0]
x06 = [1.5, 0, 0]

### PARAMETERS
B0 = 0

N = 25
bound = 2
x = np.linspace(-bound, bound, N)
y = np.linspace(-bound, bound, N)
z = np.linspace(-bound, bound, N)

B0 = .1

### ZERO T FREE ENERGY
def F3(O, J, B):
    return np.min(np.linalg.eigvals(three.H_MF(J, O[0], O[1], B)))

def F4(O, J, B):
    return np.min(np.linalg.eigvals(four.H_MF(J, O[0], O[1], B)))

def F6(O, J, B):
    return np.min(np.linalg.eigvals(six.H_MF(J, O[0], O[1], O[2], B)))

w3 = []
w4 = []
w6 = []
x03 = [1.5, 0]
x04 = [1.5, 0]
x06 = [1.5, 0,0]
O3_Arr = []
O4_Arr = []
O6_Arr = []
fig, ax = plt.subplots(1, 2)

for j in range(J_n):
    O3 = sp.optimize.minimize(F3, x03, args = (J_dom[j], B0))
    O3_Arr.append(np.max(np.real(O3.x)))

    O4 = sp.optimize.minimize(F4, x04, args = (J_dom[j], B0))
    O4_Arr.append(np.max(np.real(O4.x)))

    O6 = sp.optimize.minimize(F6, x06, args = (J_dom[j], B0))
    O6_Arr.append(np.max(np.real(O6.x)))

    for i in range(N):
        w3.append(F3([x[i], 1e-7, 0], J_dom[j], B0))
        w4.append(F4([x[i], 1e-7, 0], J_dom[j], B0))
        w6.append(F6([x[i], 1e-7, 0], J_dom[j], B0))


    ax[0].cla()
    ax[0].plot(x, w3, label = r"$F_3(O_x)$")
    ax[0].plot(x, w4, label = r"$F_4(O_x)$")
    ax[0].plot(x, w6, label = r"$F_6(O_x)$")
    ax[0].scatter(O3.x[0], F3(O3.x, J_dom[j], B0))
    ax[0].scatter(O4.x[0], F4(O4.x, J_dom[j], B0))
    ax[0].scatter(O6.x[0], F6(O6.x, J_dom[j], B0))
    ax[0].legend()

    ax[1].cla()
    ax[1].set_ylim(0,1e-3)
    ax[1].plot(O3_Arr)
    ax[1].plot(O4_Arr)
    ax[1].plot(O6_Arr)
    plt.pause(.1)
    w3 = []
    w4 = []
    w6 = []




plt.show()