from FourLevel_Base import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

### READ DATA FILE
data = pd.read_csv("/home/twiggy/Research/Chiral_Phonon_Zeeman_Splitting/Data/FourLevel_Thermodynamics.csv", delimiter = ', ')

### GET DOMAINS
J_dom = np.unique(data['J'])
B_dom = np.unique(data['B'])
T_dom = np.unique(data['T'])


def PlotOx_JvsB(t_index):
    # Get Slice
    z = np.array(data[data['T'] == T_dom[t_index]]['Ox'])

    # Reshape data
    J_mesh, B_mesh = np.meshgrid(J_dom, B_dom)
    z = z.reshape((len(B_dom), len(J_dom))).T

    # Plot
    fig, ax = plt.subplots()
    ax.set_xlabel("Interaction Strength [J]")
    ax.set_ylabel("Magnetic Field Strength [B]")

    p = ax.contourf(J_mesh, B_mesh, z)
    cbar = fig.colorbar(p)

    plt.show()


def PlotM_JvsB(t_index):
    # Get Slice
    z = np.array(data[data['T'] == T_dom[t_index]]['m'])

    # Reshape data
    J_mesh, B_mesh = np.meshgrid(J_dom, B_dom)
    z = z.reshape((len(B_dom), len(J_dom))).T

    # Plot
    fig, ax = plt.subplots()
    ax.set_xlabel("Interaction Strength [J]")
    ax.set_ylabel("Magnetic Field Strength [B]")
    p = ax.contourf(J_mesh, B_mesh, z)
    cbar = fig.colorbar(p)

    plt.show()

def PlotChi_JvsB(t_index):
    # Get Slice
    z = np.array(data[data['T'] == T_dom[t_index]]['chi'])

    # Reshape data
    J_mesh, B_mesh = np.meshgrid(J_dom, B_dom)
    z = z.reshape((len(B_dom), len(J_dom))).T

    # Plot
    fig, ax = plt.subplots()
    ax.set_xlabel("Interaction Strength [J]")
    ax.set_ylabel("Magnetic Field Strength [B]")
    p = ax.contourf(J_mesh, B_mesh, z)
    cbar = fig.colorbar(p)

    plt.show()
   
def PlotChivsJ(t_index, B_index):
    # Get Slice
    z = data[data['T'] == T_dom[t_index]]
    z = np.array(z[z['B'] == B_dom[B_index]]['chi'])

    fig, ax = plt.subplots()
    ax.set_xlabel("Interaction Strength [J]")
    ax.set_ylabel("Magnetic Susceptibility")
    ax.plot(J_dom, z)

    plt.show()
    print(z)
    
def Plot_FreeEnergy(J0, B0, T0):
    N = 100
    bound = 1
    Ox_dom = np.linspace(-bound, bound, N)
    Oy_dom = np.linspace(-bound, bound, N)
    X, Y = np.meshgrid(Ox_dom, Oy_dom)

    f = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            f[i, j] = F(T0, J0, Ox_dom[i], Oy_dom[j], B0)

    fig, ax = plt.subplots()
    ax.contourf(X, Y, f)

    plt.show()
    
PlotOx_JvsB(0)
