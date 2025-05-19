import numpy as np
import json
import matplotlib.pyplot as plt
import os
from Base import *

folder_path = "Data Runs/T=[0,2,5], B=[0,5,64], J=[0.1,2,64]/"

if not (os.path.isdir(folder_path +"/Plots")):
    os.mkdir(folder_path + "/Plots")


### GET PARAMETERS
with open(folder_path + "parameters.json", "r") as params_file:
    params = json.load(params_file)


J_dom = np.linspace(params["J_min"], params["J_max"], params["J_n"])
B_dom = np.linspace(params["B_min"], params["B_max"], params["B_n"])
T_dom = np.linspace(params["T_min"], params["T_max"], params["T_n"])

J_mesh, B_mesh = np.meshgrid(J_dom, B_dom)

           
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

tri_boundary_arr = np.reshape(tri_data.T[12], (params["J_n"], params["B_n"], params["T_n"]))

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

squ_boundary_arr = np.reshape(squ_data.T[14], (params["J_n"], params["B_n"], params["T_n"]))

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
J0_index = 3
B0_index = 1
T0_index = 0

### COMMON STRINGS
B_LABEL = r"Magnetic Field Strength H/t"
J_LABEL = r"Interaction Strength J/t"
FONTSIZE = 14

### PLOTTING FUNCTIONS
def Plot1D(X, Y, datasets, labels, path, title = ""):
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

    ax.set_title(title)
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

    if Y == "B":
        ax.set_ylabel(B_LABEL, fontsize = FONTSIZE)
        y_dom = B_dom
    elif Y == "J":
        ax.set_ylabel(J_LABEL, fontsize = FONTSIZE)
        y_dom = J_dom

    X_mesh, Y_mesh = np.meshgrid(x_dom, y_dom)

    ax.contourf(X_mesh, Y_mesh, dataset)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(folder_path + "/Plots/" + path)

    plt.close(fig)
### ENERGY LEVELS 
J0_index = 0
T0_index = 0
# Triangular Lattice
Plot1D("B", "Energy", [E0_tri_arr[J0_index, :, T0_index],
                       E1_tri_arr[J0_index, :, T0_index],
                       E2_tri_arr[J0_index, :, T0_index]],
       [], "TriangularLattice_Energies.jpg")
# Square Lattice
Plot1D("B", "Energy", [E0_squ_arr[J0_index, :, T0_index],
                       E1_squ_arr[J0_index, :, T0_index],
                       E2_squ_arr[J0_index, :, T0_index],
                       E3_squ_arr[J0_index, :, T0_index]],
       [], "SquareLattice_Energies.jpg")
# Cubic Lattice
Plot1D("B", "Energy", [E0_cub_arr[J0_index, :, T0_index],
                       E1_cub_arr[J0_index, :, T0_index],
                       E2_cub_arr[J0_index, :, T0_index],
                       E3_cub_arr[J0_index, :, T0_index],
                       E4_cub_arr[J0_index, :, T0_index],
                       E5_cub_arr[J0_index, :, T0_index]],
       [], "CubicLattice_Energies.jpg")

### PROBABILITIES
J0_index = 0
T0_index = 0
# Triangular Lattice
Plot1D("B", "Probability", [p0_tri_arr[J0_index, :, T0_index],
                            p1_tri_arr[J0_index, :, T0_index],
                            p2_tri_arr[J0_index, :, T0_index]],
       [], "TriangularLattice_Probabilities.jpg")
# Square Lattice
Plot1D("B", "Probability", [p0_squ_arr[J0_index, :, T0_index],
                            p1_squ_arr[J0_index, :, T0_index],
                            p2_squ_arr[J0_index, :, T0_index],
                            p3_squ_arr[J0_index, :, T0_index]],
       [], "SquareLattice_Probabilities.jpg")
# Cubic Lattice
Plot1D("B", "Probability", [p0_cub_arr[J0_index, :, T0_index],
                            p1_cub_arr[J0_index, :, T0_index],
                            p2_cub_arr[J0_index, :, T0_index],
                            p3_cub_arr[J0_index, :, T0_index],
                            p4_cub_arr[J0_index, :, T0_index],
                            p5_cub_arr[J0_index, :, T0_index]],
       [], "CubicLattice_Probabilities.jpg")

### MAGNETIZATIONS
# Combined
Plot1D("B", r"Magnetization $m$", [m_tri_arr[J0_index, :, T0_index],
                                  m_squ_arr[J0_index, :, T0_index],
                                  m_cub_arr[J0_index, :, T0_index]],
       ["Triangular", "Square", "Cubic"], "Magnetizations.jpg")
# Separate
Plot1D("B", r"Magnetization $m$", [m_tri_arr[J0_index, :, T0_index]],
       [], "TriangularLattice_Magnetization.jpg", f"Magnetization m, $J={{{J_dom[J0_index]}}}$, $T={{{T_dom[T0_index]}}}$")
Plot1D("B", r"Magnetization $m$", [m_squ_arr[J0_index, :, T0_index]],
       [], "SquareLattice_Magnetization.jpg", f"Magnetization m, $J={{{J_dom[J0_index]}}}$, T=${{{T_dom[T0_index]}}}$")
Plot1D("B", r"Magnetization $m$", [m_cub_arr[J0_index, :, T0_index]],
       [], "CubicLattice_Magnetization.jpg", f"Magnetization m, $J={{{J_dom[J0_index]}}}$, $T={{{T_dom[T0_index]}}}$")

### MAGNETIC SUSCEPTIBILITY 1D
B0_index = 0
# Separate
Plot1D("J", r"Magnetic Susceptibility $\chi$", [chi_tri_arr[:, B0_index, T0_index]],
       [], "TriangularLattice_MagneticSusceptibility.jpg",
       r"Magnetic Susceptibility $\chi = \frac{\partial m}{\partial B}\vert_{B=0}$" + f"$T = {{{T_dom[T0_index]}}}$")
Plot1D("J", r"Magnetic Susceptibility $\chi$", [chi_squ_arr[:, B0_index, T0_index]],
       [], "SquareLattice_MagneticSusceptibility.jpg",
       r"Magnetic Susceptibility $\chi = \frac{\partial m}{\partial B}\vert_{B=0}$" + f"$T = {{{T_dom[T0_index]}}}$")
Plot1D("J", r"Magnetic Susceptibility $\chi$", [chi_cub_arr[:, B0_index, T0_index]],
       [], "CubicLattice_MagneticSusceptibility.jpg",
       r"Magnetic Susceptibility $\chi = \frac{\partial m}{\partial B}\vert_{B=0}$" + f"$T = {{{T_dom[T0_index]}}}$")

### 2D PLOTS
# Magnetization
Plot2D("J", "B", m_tri_arr[:, :, T0_index].T, "TriangularLattice_2DMagnetization.jpg")
Plot2D("J", "B", m_squ_arr[:, :, T0_index].T, "SquareLattice_2DMagnetization.jpg")
Plot2D("J", "B", m_cub_arr[:, :, T0_index].T, "CubicLattice_2DMagnetization.jpg")

# Magnetic Susceptibility
Plot2D("J", "B", chi_tri_arr[:, :, T0_index].T, "TriangularLattice_2DMagneticSusceptibility.jpg")
Plot2D("J", "B", chi_squ_arr[:, :, T0_index].T, "SquareLattice_2DMagneticSusceptibility.jpg")
Plot2D("J", "B", chi_cub_arr[:, :, T0_index].T, "CubicLattice_2DMagneticSusceptibility.jpg")

# Order Parameter
Plot2D("J", "B", ox_tri_arr[:, :, T0_index].T, "TriangularLattice_OrderParameter.jpg")
Plot2D("J", "B", ox_squ_arr[:, :, T0_index].T, "SquareLattice_OrderParameter.jpg")
Plot2D("J", "B", ox_cub_arr[:, :, T0_index].T, "CubicLattice_OrderParameter.jpg")

### BOUNDARY PLOTS
# Triangular
tri_boundary_fig, tri_boundary_ax = plt.subplots()
indices = np.where(tri_boundary_arr == 1)

tri_boundary_ax.set_xlabel(J_LABEL)
tri_boundary_ax.set_ylabel(B_LABEL)

for t, T in enumerate(T_dom): 
    J_pts = [indices[0][i] for i in range(len(indices[0])) if indices[2][i] == t]
    B_pts = [indices[1][i] for i in range(len(indices[0])) if indices[2][i] == t]

    tri_boundary_ax.scatter(J_dom[J_pts], B_dom[B_pts], label = f"$T/t = {{{T}}}$")

    tri_boundary_ax.legend()

    tri_boundary_fig.tight_layout()
    tri_boundary_fig.savefig(folder_path + f"/Plots/TriangularBoundary_Figure{t}")


# Square
squ_boundary_fig, squ_boundary_ax = plt.subplots()
indices = np.where(squ_boundary_arr == 1)

squ_boundary_ax.set_xlabel(J_LABEL)
squ_boundary_ax.set_ylabel(B_LABEL)

for t, T in enumerate(T_dom):
    
    J_pts = [indices[0][i] for i in range(len(indices[0])) if indices[2][i] == t]
    B_pts = [indices[1][i] for i in range(len(indices[0])) if indices[2][i] == t]

    squ_boundary_ax.scatter(J_dom[J_pts], B_dom[B_pts], label = f"$T/t = {{{T}}}$")
    
    squ_boundary_ax.legend()
    
    squ_boundary_fig.tight_layout()
    squ_boundary_fig.savefig(folder_path + f"/Plots/SquareBoundary_Figure{t}")


# Cubic
cub_boundary_fig, cub_boundary_ax = plt.subplots()
indices = np.where(cub_boundary_arr == 1)

cub_boundary_ax.set_xlabel(J_LABEL)
cub_boundary_ax.set_ylabel(B_LABEL)

for t, T in enumerate(T_dom):
    
    J_pts = [indices[0][i] for i in range(len(indices[0])) if indices[2][i] == t]
    B_pts = [indices[1][i] for i in range(len(indices[0])) if indices[2][i] == t]

    cub_boundary_ax.scatter(J_dom[J_pts], B_dom[B_pts], label = f"$T/t = {{{T}}}$")

    cub_boundary_ax.legend()

    cub_boundary_fig.tight_layout()
    cub_boundary_fig.savefig(folder_path + f"/Plots/CubicBoundary_Figure{t}")
