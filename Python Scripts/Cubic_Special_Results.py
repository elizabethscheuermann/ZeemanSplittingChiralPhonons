from SixLevel_Base import *
import matplotlib.pyplot as plt
import numpy as np

def main():
    J = 5
    T = .1
    B = .1

    phi_min = 0
    phi_max = 2*np.pi
    phi_n = 2043
    phi_dom = np.linspace(phi_min, phi_max, phi_n)
    
    theta = np.pi/4

    theta_dom = np.linspace(0, 2*np.pi, 100)
    
    phi = np.pi/3
    Fxs = []
    Fys = []
    Fzs = []
    for theta in theta_dom:
        Bx = B * np.sin(theta) * np.cos(phi)
        By = B * np.sin(theta) * np.sin(phi)
        Bz = B * np.cos(theta)
        F_x = F(T, 2, 0, 0, J, Bx, By, Bz) 
        F_y = F(T, 0, 2, 0, J, Bx, By, Bz)
        F_z = F(T, 0, 0, 2, J, Bx, By, Bz)
    
        Fxs.append(F_x)
        Fys.append(F_y)
        Fzs.append(F_z)

    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\theta [rad]$")
    ax.set_ylabel(r"Free Energy Minimum")
    ax.set_title(r"$\phi = \pi/3$") 
    

    ax.plot(theta_dom, Fxs, label = r"$F_x$")
    ax.plot(theta_dom, Fys, label = r"$F_y$")
    ax.plot(theta_dom, Fzs, label = r"$F_z$")


    ax.legend()
    plt.show()

    J = .25
    T = 1
    Bdom = np.linspace(0, 10,100)
    dOx = 1e-2

    dF_Arr = []
    coefs = [7/240, -2/5, 169/60, -122/15, 91/8, -122/15, 169/60, -2/5, 7/240]
    for B in Bdom:
        dF = 0
        for c in range(len(coefs)):
            dF += coefs[c] * F(T, (c - 4)*dOx, 0, 0, J, 0, 0, B)
            
        dF_Arr.append((dF)/(dOx**4))

    fig2, ax2 = plt.subplots()
    ax2.spines['left'].set_position('zero')
    ax2.spines['bottom'].set_position('zero')
    ax2.set_xlabel("Magnetic Field [B]")
    ax2.set_ylabel(r"$\gamma(B,T)$")
    ax2.plot(Bdom, dF_Arr)
    plt.show()
main()
