import numpy as np
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
from Base import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-Jmin", "--Jmin")
parser.add_argument("-Jmax", "--Jmax")
parser.add_argument("-Jn", "--Jn")
parser.add_argument("-Bmin", "--Bmin")
parser.add_argument("-Bmax", "--Bmax")
parser.add_argument("-Bn", "--Bn")
parser.add_argument("-Tmin", "--Tmin")
parser.add_argument("-Tmax", "--Tmax")
parser.add_argument("-Tn", "--Tn")
parser.add_argument("-J0index", "--J0index")
parser.add_argument("-B0index", "--B0index")
parser.add_argument("-T0index", "--T0index")



### GET ARGS
args = parser.parse_args()

folder_path = "DataRuns/"

folder_path += "J=[" + str(args.Jmin) + "," + str(args.Jmax) + "," + str(args.Jn) + "],"
folder_path += "B=[" + str(args.Bmin) + "," + str(args.Bmax) + "," + str(args.Bn) + "],"
folder_path += "T=[" + str(args.Tmin) + "," + str(args.Tmax) + "," + str(args.Tn) + "]/"


J0_index = int(args.J0index)
B0_index = int(args.B0index)
T0_index = int(args.T0index)

### GET PARAMETERS
with open(folder_path + "parameters.json", "r") as params_file:
    params = json.load(params_file)


J_dom = np.linspace(params["J_min"], params["J_max"], params["J_n"])
B_dom = np.linspace(params["B_min"], params["B_max"], params["B_n"])
T_dom = np.linspace(params["T_min"], params["T_max"], params["T_n"])
ox_dom = np.linspace(-1.5, 1.5, 100)
oy_dom = np.linspace(-1.5, 1.5, 100)

J_mesh, B_mesh = np.meshgrid(J_dom, B_dom)


if not (os.path.isdir(folder_path +"/Plots/" )):
    os.mkdir(folder_path + "/Plots")


### GET DATA
# Triangular Lattice
tri_data = np.loadtxt(folder_path + "TriangularLattice_Data.csv", delimiter = ",", skiprows = 1)
ox_tri_arr = np.reshape(tri_data.T[3], (params["J_n"], params["B_n"], params["T_n"]))
m_tri_arr = np.reshape(tri_data.T[4], (params["J_n"], params["B_n"], params["T_n"]))
chi_tri_arr = np.reshape(tri_data.T[5], (params["J_n"], params["B_n"], params["T_n"]))

E0_tri_arr = np.reshape(tri_data.T[6], (params["J_n"], params["B_n"], params["T_n"]))
E1_tri_arr = np.reshape(tri_data.T[7], (params["J_n"], params["B_n"], params["T_n"]))
E2_tri_arr = np.reshape(tri_data.T[8], (params["J_n"], params["B_n"], params["T_n"]))

p0_tri_arr = np.reshape(tri_data.T[9], (params["J_n"], params["B_n"], params["T_n"]))
p1_tri_arr = np.reshape(tri_data.T[10], (params["J_n"], params["B_n"], params["T_n"]))
p2_tri_arr = np.reshape(tri_data.T[11], (params["J_n"], params["B_n"], params["T_n"]))

w1_tri_arr = np.reshape(tri_data.T[12], (params["J_n"], params["B_n"], params["T_n"]))
w2_tri_arr = np.reshape(tri_data.T[13], (params["J_n"], params["B_n"], params["T_n"]))

tri_boundary_arr = np.reshape(tri_data.T[14], (params["J_n"], params["B_n"], params["T_n"]))

# Square Lattice
squ_data = np.loadtxt(folder_path + "SquareLattice_Data.csv", delimiter = ",", skiprows = 1)
ox_squ_arr = np.reshape(squ_data.T[3], (params["J_n"], params["B_n"], params["T_n"]))
m_squ_arr = np.reshape(squ_data.T[4], (params["J_n"], params["B_n"], params["T_n"]))
chi_squ_arr = np.reshape(squ_data.T[5], (params["J_n"], params["B_n"], params["T_n"]))

E0_squ_arr = np.reshape(squ_data.T[6], (params["J_n"], params["B_n"], params["T_n"]))
E1_squ_arr = np.reshape(squ_data.T[7], (params["J_n"], params["B_n"], params["T_n"]))
E2_squ_arr = np.reshape(squ_data.T[8], (params["J_n"], params["B_n"], params["T_n"]))
E3_squ_arr = np.reshape(squ_data.T[9], (params["J_n"], params["B_n"], params["T_n"]))

p0_squ_arr = np.reshape(squ_data.T[10], (params["J_n"], params["B_n"], params["T_n"]))
p1_squ_arr = np.reshape(squ_data.T[11], (params["J_n"], params["B_n"], params["T_n"]))
p2_squ_arr = np.reshape(squ_data.T[12], (params["J_n"], params["B_n"], params["T_n"]))
p3_squ_arr = np.reshape(squ_data.T[13], (params["J_n"], params["B_n"], params["T_n"]))


w1_squ_arr = np.reshape(squ_data.T[14], (params["J_n"], params["B_n"], params["T_n"]))
w2_squ_arr = np.reshape(squ_data.T[15], (params["J_n"], params["B_n"], params["T_n"]))
w3_squ_arr = np.reshape(squ_data.T[16], (params["J_n"], params["B_n"], params["T_n"]))
#w4_squ_arr = np.reshape(squ_data.T[17], (params["J_n"], params["B_n"], params["T_n"]))
#w5_squ_arr = np.reshape(squ_data.T[18], (params["J_n"], params["B_n"], params["T_n"]))
#w6_squ_arr = np.reshape(squ_data.T[19], (params["J_n"], params["B_n"], params["T_n"]))

squ_boundary_arr = np.reshape(squ_data.T[17], (params["J_n"], params["B_n"], params["T_n"]))

# Cubic Lattice
cub_data = np.loadtxt(folder_path + "CubicLattice_Data.csv", delimiter = ",", skiprows = 1)
ox_cub_arr = np.reshape(cub_data.T[3], (params["J_n"], params["B_n"], params["T_n"]))
m_cub_arr = np.reshape(cub_data.T[4], (params["J_n"], params["B_n"], params["T_n"]))
chi_cub_arr = np.reshape(cub_data.T[5], (params["J_n"], params["B_n"], params["T_n"]))

E0_cub_arr = np.reshape(cub_data.T[6], (params["J_n"], params["B_n"], params["T_n"]))
E1_cub_arr = np.reshape(cub_data.T[7], (params["J_n"], params["B_n"], params["T_n"]))
E2_cub_arr = np.reshape(cub_data.T[8], (params["J_n"], params["B_n"], params["T_n"]))
E3_cub_arr = np.reshape(cub_data.T[9], (params["J_n"], params["B_n"], params["T_n"]))
E4_cub_arr = np.reshape(cub_data.T[10], (params["J_n"], params["B_n"], params["T_n"]))
E5_cub_arr = np.reshape(cub_data.T[11], (params["J_n"], params["B_n"], params["T_n"]))

p0_cub_arr = np.reshape(cub_data.T[12], (params["J_n"], params["B_n"], params["T_n"]))
p1_cub_arr = np.reshape(cub_data.T[13], (params["J_n"], params["B_n"], params["T_n"]))
p2_cub_arr = np.reshape(cub_data.T[14], (params["J_n"], params["B_n"], params["T_n"]))

p0_cub_arr = np.reshape(cub_data.T[12], (params["J_n"], params["B_n"], params["T_n"]))
p1_cub_arr = np.reshape(cub_data.T[13], (params["J_n"], params["B_n"], params["T_n"]))
p2_cub_arr = np.reshape(cub_data.T[14], (params["J_n"], params["B_n"], params["T_n"]))
E4_cub_arr = np.reshape(cub_data.T[10], (params["J_n"], params["B_n"], params["T_n"]))
E5_cub_arr = np.reshape(cub_data.T[11], (params["J_n"], params["B_n"], params["T_n"]))

p0_cub_arr = np.reshape(cub_data.T[12], (params["J_n"], params["B_n"], params["T_n"]))
p1_cub_arr = np.reshape(cub_data.T[13], (params["J_n"], params["B_n"], params["T_n"]))
p2_cub_arr = np.reshape(cub_data.T[14], (params["J_n"], params["B_n"], params["T_n"]))
p3_cub_arr = np.reshape(cub_data.T[15], (params["J_n"], params["B_n"], params["T_n"]))
p4_cub_arr = np.reshape(cub_data.T[16], (params["J_n"], params["B_n"], params["T_n"]))
p5_cub_arr = np.reshape(cub_data.T[17], (params["J_n"], params["B_n"], params["T_n"]))


cub_boundary_arr = np.reshape(cub_data.T[18], (params["J_n"], params["B_n"], params["T_n"]))

### PLOTTING

### COMMON STRINGS
B_LABEL = r"Magnetic Field Strength H/t"
J_LABEL = r"Interaction Strength J/t"
T_LABEL = r"Temperature T/t"
OX_LABEL = r"$\langle x \rangle$"
OY_LABEL = r"$\langle y \rangle$"
FONTSIZE = 16

### PLOTTING FUNCTIONS
def Plot1D(X, Y, datasets, labels, path, title = "", xlim = ["", ""], ylim = ["", ""]):
    fig, ax = plt.subplots()
    if X == "B":
        ax.set_xlabel(B_LABEL, fontsize = FONTSIZE)
        dom = B_dom
    elif X == "J":
        ax.set_xlabel(J_LABEL, fontsize = FONTSIZE)
        dom = J_dom

    ax.set_ylabel(Y,fontsize = FONTSIZE)
    if labels != []:
        for d, dataset in enumerate(datasets):
            ax.plot(dom, dataset, label = labels[d])
        ax.legend()
    else:
        for d, dataset in enumerate(datasets):
            ax.plot(dom, dataset)

    if xlim[0] != "" and xlim[1] != "":
        ax.set_xlim(xlim)

    if ylim[0] != "" and ylim[1] != "":
        ax.set_ylim(ylim)
    ax.set_title(title, fontsize = FONTSIZE)
    fig.tight_layout()
    fig.savefig(folder_path + "/Plots/" + path)
            
    plt.close(fig)

def Plot2D(X, Y, dataset, path, title = ""):
    fig, ax = plt.subplots()
    if X == "B":
        ax.set_xlabel(B_LABEL, fontsize = FONTSIZE)
        x_dom = B_dom
    elif X == "J":
        ax.set_xlabel(J_LABEL, fontsize = FONTSIZE)
        x_dom = J_dom
    elif X == "T":
        ax.set_xlabel(T_LABEL, fontsize = FONTSIZE)
        x_dom = T_dom
    elif X == "Ox":
        ax.set_xlabel(OX_LABEL, fontsize = FONTSIZE)
        x_dom = ox_dom
    elif X == "Oy":
        ax.set_xlabel(OY_LABEL, fontsize = FONTSIZE)
        x_dom = oy_dom
    elif X == "x":
        ax.set_xlabel(r"$x$", fontsize = FONTSIZE)
        x_dom = ox_dom

    if Y == "B":
        ax.set_ylabel(B_LABEL, fontsize = FONTSIZE)
        y_dom = B_dom
    elif Y == "J":
        ax.set_ylabel(J_LABEL, fontsize = FONTSIZE)
        y_dom = J_dom
    elif Y == "T":
        ax.set_ylabel(T_LABEL, fontsize = FONTSIZE)
        y_dom = T_dom
    elif Y == "Ox":
        ax.set_ylabel(OX_LABEL, fontsize = FONTSIZE)
        y_dom = ox_dom
    elif Y == "Oy":
        ax.set_ylabel(OY_LABEL, fontsize = FONTSIZE)
        y_dom = oy_dom
    elif Y == "y":
        ax.set_ylabel(r"$y$", fontsize = FONTSIZE)
        y_dom = oy_dom

    X_mesh, Y_mesh = np.meshgrid(x_dom, y_dom)

    ax.contourf(X_mesh, Y_mesh, dataset, levels = 40)
    ax.set_title(title, fontsize = FONTSIZE)
    fig.tight_layout()
    fig.savefig(folder_path + "/Plots/" + path)

    plt.close(fig)
### ENERGY LEVELS 
# Triangular Lattice
Plot1D("B", "Energy", [E0_tri_arr[J0_index, :, T0_index],
                       E1_tri_arr[J0_index, :, T0_index],
                       E2_tri_arr[J0_index, :, T0_index]],
       [], f"TriangularLattice_Energies_J0={J_dom[J0_index]:.2f}_T0={T_dom[T0_index]:.2f}.jpg")
# Square Lattice
Plot1D("B", "Energy", [E0_squ_arr[J0_index, :, T0_index],
                       E1_squ_arr[J0_index, :, T0_index],
                       E2_squ_arr[J0_index, :, T0_index],
                       E3_squ_arr[J0_index, :, T0_index]],
       [], f"SquareLattice_Energies_J0={J_dom[J0_index]:.2f}_T0={T_dom[T0_index]:.2f}.jpg")
# Cubic Lattice
Plot1D("B", "Energy", [E0_cub_arr[J0_index, :, T0_index],
                       E1_cub_arr[J0_index, :, T0_index],
                       E2_cub_arr[J0_index, :, T0_index],
                       E3_cub_arr[J0_index, :, T0_index],
                       E4_cub_arr[J0_index, :, T0_index],
                       E5_cub_arr[J0_index, :, T0_index]],
       [], f"CubicLattice_Energies_J0={J_dom[J0_index]:.2f}_T0={T_dom[T0_index]:.2f}.jpg")

### PROBABILITIES
# Triangular Lattice
Plot1D("B", "Probability", [p0_tri_arr[J0_index, :, T0_index],
                            p1_tri_arr[J0_index, :, T0_index],
                            p2_tri_arr[J0_index, :, T0_index]],
       [], f"TriangularLattice_Probabilities_J0={J_dom[J0_index]:.2f}_T0={T_dom[T0_index]:.2f}.jpg")
# Square Lattice
Plot1D("B", "Probability", [p0_squ_arr[J0_index, :, T0_index],
                            p1_squ_arr[J0_index, :, T0_index],
                            p2_squ_arr[J0_index, :, T0_index],
                            p3_squ_arr[J0_index, :, T0_index]],
       [], f"SquareLattice_Probabilities_J0={J_dom[J0_index]:.2f}_T0={T_dom[T0_index]:.2f}.jpg")
# Cubic Lattice
Plot1D("B", "Probability", [p0_cub_arr[J0_index, :, T0_index],
                            p1_cub_arr[J0_index, :, T0_index],
                            p2_cub_arr[J0_index, :, T0_index],
                            p3_cub_arr[J0_index, :, T0_index],
                            p4_cub_arr[J0_index, :, T0_index],
                            p5_cub_arr[J0_index, :, T0_index]],
       [], f"CubicLattice_Probabilities_J0={J_dom[J0_index]:.2f}_T0={T_dom[T0_index]:.2f}.jpg")

### MAGNETIZATIONS
# Combined
Plot1D("B", r"Magnetization $m$", [m_tri_arr[J0_index, :, T0_index],
                                  m_squ_arr[J0_index, :, T0_index],
                                  m_cub_arr[J0_index, :, T0_index]],
       ["Triangular", "Square", "Cubic"], f"Magnetizations_J0={J_dom[J0_index]:.2f}_T0={T_dom[T0_index]:.2f}.jpg")
# Separate
Plot1D("B", r"Magnetization $m$", [m_tri_arr[J0_index, :, T0_index]],
       [], f"TriangularLattice_Magnetization_J0={J_dom[J0_index]:.2f}_T0={T_dom[T0_index]:.2f}.jpg", f"Magnetization m, $J={{{J_dom[J0_index]}}}$, $T={{{T_dom[T0_index]}}}$")
Plot1D("B", r"Magnetization $m$", [m_squ_arr[J0_index, :, T0_index]],
       [], f"SquareLattice_Magnetization_J0={J_dom[J0_index]:.2f}_T0={T_dom[T0_index]:.2f}.jpg", f"Magnetization m, $J={{{J_dom[J0_index]}}}$, T=${{{T_dom[T0_index]}}}$")
Plot1D("B", r"Magnetization $m$", [m_cub_arr[J0_index, :, T0_index]],
       [], f"CubicLattice_Magnetization_J0={J_dom[J0_index]:.2f}_T0={T_dom[T0_index]:.2f}.jpg", f"Magnetization m, $J={{{J_dom[J0_index]}}}$, $T={{{T_dom[T0_index]}}}$")

### MAGNETIC SUSCEPTIBILITY 1D
# Combined
Plot1D("J", r"Magnetic Susceptibility $\chi$", 
		[chi_tri_arr[:, B0_index, T0_index],
		 chi_squ_arr[:, B0_index, T0_index]],
["Triangular", "Square"], f"MagneticSusceptibilities_B0={B_dom[B0_index]:.2f}_T0={T_dom[T0_index]:.2f}.jpg", r"Magnetic Susceptibility $\chi = \frac{\partial m}{\partial B}\vert_{B=0}$ at " + f"$T = {{{T_dom[T0_index]}}}$")


# Separate
Plot1D("J", r"Magnetic Susceptibility $\chi$", [chi_tri_arr[:, B0_index, T0_index]],
       [], f"TriangularLattice_MagneticSusceptibility_B0={B_dom[B0_index]:.2f}_T0={T_dom[T0_index]:.2f}.jpg",
       r"Magnetic Susceptibility $\chi = \frac{\partial m}{\partial B}\vert_{B=0}$" + f"$T = {{{T_dom[T0_index]}}}$")
Plot1D("J", r"Magnetic Susceptibility $\chi$", [chi_squ_arr[:, B0_index, T0_index]],
       [], f"SquareLattice_MagneticSusceptibility_B0={B_dom[B0_index]:.2f}_T0={T_dom[T0_index]:.2f}.jpg",
       r"Magnetic Susceptibility $\chi = \frac{\partial m}{\partial B}\vert_{B=0}$" + f"$T = {{{T_dom[T0_index]}}}$")
Plot1D("J", r"Magnetic Susceptibility $\chi$", [chi_cub_arr[:, B0_index, T0_index]],
       [], f"CubicLattice_MagneticSusceptibility_B0={B_dom[B0_index]:.2f}_T0={T_dom[T0_index]:.2f}.jpg",
       r"Magnetic Susceptibility $\chi = \frac{\partial m}{\partial B}\vert_{B=0}$" + f"$T = {{{T_dom[T0_index]}}}$")

### PHONON DYNAMICS 1D
# Triangular Lattice
Plot1D("B", r"Phonon Frequency $\omega$", [w1_tri_arr[J0_index, :, T0_index],
					   w2_tri_arr[J0_index, :, T0_index]],
       [r"$\omega_{0,-1}$", r"$\omega_{0,1}$"], f"TriangularLattice_PhononDynamics_J0={J_dom[J0_index]:.2f}_T0={T_dom[T0_index]:.2f}.jpg", f"$J/t = {{{J_dom[J0_index]:.2f}}}$, $T/t = {{{T_dom[T0_index]:.2f}}}$")


# Square Lattice
Plot1D("B", r"Phonon Frequency $\omega$", [w1_squ_arr[J0_index, :, T0_index],
                                           w2_squ_arr[J0_index, :, T0_index],
                                           w3_squ_arr[J0_index, :, T0_index]],
                                           #w4_squ_arr[J0_index, :, T0_index],
                                           #w5_squ_arr[J0_index, :, T0_index],
                                           #w6_squ_arr[J0_index, :, T0_index]],
       [r"$\omega_{0,-1}$", r"$\omega_{0,1}$", r"$\omega_{0,2}$"], f"SquareLattice_PhononDynamics_J0={J_dom[J0_index]:.2f}_T0={T_dom[T0_index]:.2f}.jpg", f"$J/t = {{{J_dom[J0_index]:.2f}}}$, $T/t = {{{T_dom[T0_index]:.2f}}}$")

### 2D PLOTS

a = -16
b = -a/2

# Potential Energy
U_tri_Arr = np.zeros((len(ox_dom), len(oy_dom)))
U_squ_Arr = np.zeros((len(ox_dom), len(oy_dom)))
for i in range(len(ox_dom)):
	for j in range(len(oy_dom)):
		r = np.sqrt(ox_dom[i]**2 + oy_dom[j]**2)
		theta = np.arctan2(oy_dom[j], ox_dom[i])

		U_tri_Arr[i,j] = a * r**2 + b * r**4 * (2 - np.cos(3*theta)) if r < 1.5 else np.nan
		
		U_squ_Arr[i,j] = a * r**2 + b * r**4 * (2 - np.cos(4*theta)) if r < 1.5 else np.nan


Plot2D("x", "y", U_tri_Arr, f"TriangularLattice_PotentialEnergy_J0={J_dom[J0_index]:.2f}_B0={B_dom[B0_index]:.2f}.jpg")
Plot2D("x", "y", U_squ_Arr, f"SquareLattice_PotentialEnergy_J0={J_dom[J0_index]:.2f}_B0={B_dom[B0_index]:.2f}.jpg")


# Free Energy
F_tri_Arr = np.zeros((len(ox_dom), len(oy_dom)))
F_squ_Arr = np.zeros((len(ox_dom), len(oy_dom)))

for i in range(len(ox_dom)):
	for j in range(len(oy_dom)):
		F_tri_Arr[i,j] = F_MF_tri([ox_dom[i], oy_dom[j]], J_dom[J0_index], B_dom[B0_index], T_dom[T0_index])
		F_squ_Arr[i,j] = F_MF_squ([ox_dom[i], oy_dom[j]], J_dom[J0_index], B_dom[B0_index], T_dom[T0_index])

Plot2D("Ox", "Oy", F_tri_Arr, f"TriangularLattice_FreeEnergy_J0={J_dom[J0_index]:.2f}_B0={B_dom[B0_index]:.2f}_T0={T_dom[T0_index]:.2f}.jpg")
Plot2D("Ox", "Oy", F_squ_Arr, f"SquareLattice_FreeEnergy_J0={J_dom[J0_index]:.2f}_B0={B_dom[B0_index]:.2f}_T0={T_dom[T0_index]:.2f}.jpg")


# Magnetization
Plot2D("J", "B", m_tri_arr[:, :, T0_index].T, f"TriangularLattice_2DMagnetization_T0={T_dom[T0_index]:.2f}.jpg")
Plot2D("J", "B", m_squ_arr[:, :, T0_index].T, f"SquareLattice_2DMagnetization_T0={T_dom[T0_index]:.2f}.jpg")
Plot2D("J", "B", m_cub_arr[:, :, T0_index].T, f"CubicLattice_2DMagnetization_T0={T_dom[T0_index]:.2f}.jpg")

# Magnetic Susceptibility
Plot2D("J", "B", chi_tri_arr[:, :, T0_index].T, f"TriangularLattice_2DMagneticSusceptibility_T0={T_dom[T0_index]:.2f}.jpg")
Plot2D("J", "B", chi_squ_arr[:, :, T0_index].T, f"SquareLattice_2DMagneticSusceptibility_T0={T_dom[T0_index]:.2f}.jpg")
Plot2D("J", "B", chi_cub_arr[:, :, T0_index].T, f"CubicLattice_2DMagneticSusceptibility_T0={T_dom[T0_index]:.2f}.jpg")

# Order Parameter
Plot2D("J", "B", ox_tri_arr[:, :, T0_index].T, f"TriangularLattice_OrderParameter_T0={T_dom[T0_index]:.2f}.jpg")
Plot2D("J", "B", ox_squ_arr[:, :, T0_index].T, f"SquareLattice_OrderParameter_T0={T_dom[T0_index]:.2f}.jpg")
Plot2D("J", "B", ox_cub_arr[:, :, T0_index].T, f"CubicLattice_OrderParameter_T0={T_dom[T0_index]:.2f}.jpg")



### BOUNDARY PLOTS
# Triangular
tri_boundary_fig, tri_boundary_ax = plt.subplots()
indices = np.where(tri_boundary_arr == 1)

tri_boundary_ax.set_xlabel(J_LABEL, fontsize = FONTSIZE)
tri_boundary_ax.set_ylabel(B_LABEL, fontsize = FONTSIZE)

for t, T in enumerate(T_dom): 
    J_pts = [indices[0][i] for i in range(len(indices[0])) if indices[2][i] == t]
    B_pts = [indices[1][i] for i in range(len(indices[0])) if indices[2][i] == t]

    tri_boundary_ax.scatter(J_dom[J_pts], B_dom[B_pts], label = f"$T/t = {{{T}}}$")

    tri_boundary_ax.legend()

    tri_boundary_fig.tight_layout()
    tri_boundary_fig.savefig(folder_path + "/Plots/" + f"TriangularBoundary_Figure{t}")


# Square
squ_boundary_fig, squ_boundary_ax = plt.subplots()
indices = np.where(squ_boundary_arr == 1)

squ_boundary_ax.set_xlabel(J_LABEL, fontsize = FONTSIZE)
squ_boundary_ax.set_ylabel(B_LABEL, fontsize = FONTSIZE)

for t, T in enumerate(T_dom):
    
    J_pts = [indices[0][i] for i in range(len(indices[0])) if indices[2][i] == t]
    B_pts = [indices[1][i] for i in range(len(indices[0])) if indices[2][i] == t]

    squ_boundary_ax.scatter(J_dom[J_pts], B_dom[B_pts], label = f"$T/t = {{{T}}}$")
    
    squ_boundary_ax.legend()
    
    squ_boundary_fig.tight_layout()
    squ_boundary_fig.savefig(folder_path + "/Plots/" + f"SquareBoundary_Figure{t}")


# Cubic
cub_boundary_fig, cub_boundary_ax = plt.subplots()
indices = np.where(cub_boundary_arr == 1)

cub_boundary_ax.set_xlabel(J_LABEL, fontsize = FONTSIZE)
cub_boundary_ax.set_ylabel(B_LABEL, fontsize = FONTSIZE)

for t, T in enumerate(T_dom):
    
    J_pts = [indices[0][i] for i in range(len(indices[0])) if indices[2][i] == t]
    B_pts = [indices[1][i] for i in range(len(indices[0])) if indices[2][i] == t]

    cub_boundary_ax.scatter(J_dom[J_pts], B_dom[B_pts], label = f"$T/t = {{{T}}}$")

    cub_boundary_ax.legend()

    cub_boundary_fig.tight_layout()
    cub_boundary_fig.savefig(folder_path + "/Plots/" + f"CubicBoundary_Figure{t}")


# Cubic Tuning
angles = [[0,0], [0,np.pi/2], [np.pi/4,np.pi/2]]



for angle in angles:
	B0_vec = B_dom[B0_index]*np.array([np.cos(angle[0])*np.sin(angle[1]), np.sin(angle[0])*np.sin(angle[1]), np.cos(angle[1])])


	F_vals = np.zeros((len(ox_dom), len(oy_dom)))

	for i in range(len(ox_dom)):
    		for j in range(len(oy_dom)):
        		o_vec = np.array([ox_dom[i], oy_dom[j], 0])
        		F_vals[i,j] = F_MF_cub(o_vec, J_dom[J0_index], B0_vec, T_dom[T0_index])


	fig, ax = plt.subplots()

	ax.contourf(ox_dom, oy_dom, F_vals, levels = 20)

	ax.quiver(0, 0, B0_vec[0], B0_vec[1], color = "black", scale = 5, label = "B Field Direction")
		
	ax.set_xlabel(OX_LABEL, fontsize = FONTSIZE)
	ax.set_ylabel(OY_LABEL, fontsize = FONTSIZE)
	
	ax.set_title(f"J/t = {J_dom[J0_index]:.2f}, B/t = ({B0_vec[0]:.2f}, {B0_vec[1]:.2f}, {B0_vec[2]:.2f})", fontsize = FONTSIZE)

	fig.tight_layout()	
	fig.savefig(folder_path + "/Plots/" + f"CubicLattice_FreeEnergy_J0={J_dom[J0_index]:.2f}_B0={B_dom[B0_index]:.2f}_T0={T_dom[T0_index]:.2f}_theta={angle[0]:.2f}_phi={angle[1]:.2f}.jpg")
