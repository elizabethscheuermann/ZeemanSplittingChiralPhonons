import numpy as np
import json
import matplotlib.pyplot as plt
import os
from Base import *

folder_path = "Data Runs/T=[0,2,5], B=[0,5,16], J=[0.1,2,16]/"

### GET PARAMETERS
with open(folder_path + "parameters.json", "r") as params_file:
    params = json.load(params_file)


J_dom = np.linspace(params["J_min"], params["J_max"], params["J_n"])
B_dom = np.linspace(params["B_min"], params["B_max"], params["B_n"])
T_dom = np.linspace(params["T_min"], params["T_max"], params["T_n"])

J_mesh, B_mesh = np.meshgrid(J_dom, B_dom)

# Make folder
if not (os.path.isdir(folder_path + "/Plots")):
    os.mkdir(folder_path + "/Plots")

plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
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
J_LABEL = r"Interactoin Strength J/t"

### PLOTTING FUNCTIONS
def Plot1D(X, Y, datasets, labels, path, title = ""):
    fig, ax = plt.subplots()
    if X == "B":
        ax.set_xlabel(B_LABEL)
        dom = B_dom
    elif X == "J":
        ax.set_xlabel(J_LABEL)
        dom = J_dom

    ax.set_ylabel(Y)
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

### ENERGY LEVELS 
J0_index = 0
T0_index = 0
# Triangular Lattice
Plot1D("B", "Energy", [E0_tri_arr[J0_index, :, T0_index],
                       E1_tri_arr[J0_index, :, T0_index],
                       E2_tri_arr[J0_index, :, T0_index]],
       ["A", "B", "C"], "TriangularLattice_Energies.jpg")
# Square Lattice
Plot1D("B", "Energy", [E0_squ_arr[J0_index, :, T0_index],
                       E1_squ_arr[J0_index, :, T0_index],
                       E2_squ_arr[J0_index, :, T0_index],
                       E3_squ_arr[J0_index, :, T0_index]],
       ["A", "B", "C", "D"], "SquareLattice_Energies.jpg")
# Cubic Lattice
Plot1D("B", "Energy", [E0_cub_arr[J0_index, :, T0_index],
                       E1_cub_arr[J0_index, :, T0_index],
                       E2_cub_arr[J0_index, :, T0_index],
                       E3_cub_arr[J0_index, :, T0_index],
                       E4_cub_arr[J0_index, :, T0_index],
                       E5_cub_arr[J0_index, :, T0_index]],
       ["A", "B", "C", "D", "E", "F"], "CubicLattice_Energies.jpg")

### PROBABILITIES
J0_index = 0
T0_index = 0
# Triangular Lattice
Plot1D("B", "Probability", [p0_tri_arr[J0_index, :, T0_index],
                            p1_tri_arr[J0_index, :, T0_index],
                            p2_tri_arr[J0_index, :, T0_index]],
       ["A", "B", "C"], "TriangularLattice_Probabilities.jpg")
# Square Lattice
Plot1D("B", "Probability", [p0_squ_arr[J0_index, :, T0_index],
                            p1_squ_arr[J0_index, :, T0_index],
                            p2_squ_arr[J0_index, :, T0_index],
                            p3_squ_arr[J0_index, :, T0_index]],
       ["A", "B", "C", "D"], "SquareLattice_Probabilities.jpg")
# Cubic Lattice
Plot1D("B", "Probability", [p0_cub_arr[J0_index, :, T0_index],
                            p1_cub_arr[J0_index, :, T0_index],
                            p2_cub_arr[J0_index, :, T0_index],
                            p3_cub_arr[J0_index, :, T0_index],
                            p4_cub_arr[J0_index, :, T0_index],
                            p5_cub_arr[J0_index, :, T0_index]],
       ["A", "B", "C", "D", "E", "F"], "CubicLattice_Probabilities.jpg")

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

# Magnetic Susceptibility J vs B
chi_fig2, chi_ax2 = plt.subplots(1, 3)
chi_ax2[0].contourf(J_mesh, B_mesh, chi_tri_arr[:, :, T0_index].T)
chi_ax2[1].contourf(J_mesh, B_mesh, chi_squ_arr[:, :, T0_index].T)
chi_ax2[2].contourf(J_mesh, B_mesh, chi_cub_arr[:, :, T0_index].T)


# Order Parameters
order_param_fig, order_param_ax = plt.subplots(1, 3)
order_param_ax[0].contourf(J_mesh, B_mesh, ox_tri_arr[:, :, T0_index].T)
order_param_ax[1].contourf(J_mesh, B_mesh, ox_squ_arr[:, :, T0_index].T)
order_param_ax[2].contourf(J_mesh, B_mesh, ox_cub_arr[:, :, T0_index].T)

order_param_ax[0].set_xlabel(J_LABEL)
order_param_ax[0].set_ylabel(B_LABEL)
order_param_ax[0].set_title(r"Triangular Lattice")

order_param_ax[1].set_xlabel(J_LABEL)
order_param_ax[1].set_ylabel(B_LABEL)
order_param_ax[1].set_title(r"Square Lattice")

order_param_ax[2].set_xlabel(J_LABEL)
order_param_ax[2].set_ylabel(B_LABEL)
order_param_ax[2].set_title(r"Cubic Lattice")

order_param_fig.suptitle(r"Mean Field Order Parameter, $T = $" + str(T_dom[T0_index]))

tri_order_param_fig, tri_order_param_ax = plt.subplots()
tri_order_param_ax.set_xlabel(J_LABEL)
tri_order_param_ax.set_ylabel(B_LABEL)
tri_order_param_ax.set_title(r"Triangular Lattice Order Parameter $\langle x \rangle$ at " + f"$T = {{{T_dom[T0_index]}}}$")

tri_im = tri_order_param_ax.contourf(J_mesh, B_mesh, np.round(ox_tri_arr[:, :, T0_index].T,4))
tri_order_param_fig.colorbar(tri_im, location = 'right')

squ_order_param_fig, squ_order_param_ax = plt.subplots()
squ_order_param_ax.set_xlabel(J_LABEL)
squ_order_param_ax.set_ylabel(B_LABEL)
squ_order_param_ax.set_title(r"Square Lattice Order Parameter $\langle x \rangle$ at " + f"$T = {{{T_dom[T0_index]}}}$")

squ_im = squ_order_param_ax.contourf(J_mesh, B_mesh, np.round(ox_squ_arr[:, :, T0_index].T,4))
squ_order_param_fig.colorbar(squ_im, location = 'right')

cub_order_param_fig, cub_order_param_ax = plt.subplots()
cub_order_param_ax.set_xlabel(J_LABEL)
cub_order_param_ax.set_ylabel(B_LABEL)
cub_order_param_ax.set_title(r"Cubic Lattice Order Paramater $\langle x \rangle$ at " + f"$T = {{{T_dom[T0_index]}}}$")

cub_im = cub_order_param_ax.contourf(J_mesh, B_mesh, np.round(ox_cub_arr[:, :, T0_index].T,4))
cub_order_param_fig.colorbar(cub_im, location = 'right')

# Cubic Tuning Plots
B0 = 20
phi_1 = 0
theta_1 = 0
B1 = [B0 * np.cos(phi_1)*np.sin(theta_1), B0 * np.sin(phi_1) * np.sin(theta_1), B0 * np.cos(theta_1)]

phi_2 = 0
theta_2 = np.pi/2
B2 = [B0 * np.cos(phi_2)*np.sin(theta_2), B0 * np.sin(phi_2) * np.sin(theta_2), B0 * np.cos(theta_2)]

phi_3 = np.pi/4
theta_3 = np.pi/4
B3 = [B0 * np.cos(phi_3)*np.sin(theta_3), B0 * np.sin(phi_3) * np.sin(theta_3), B0 * np.cos(theta_3)]

O_bound = 2
O_N = 32
Ox_dom = np.linspace(-O_bound, O_bound, O_N)
Oy_dom = np.linspace(-O_bound, O_bound, O_N)
Ox_mesh, Oy_mesh = np.meshgrid(Ox_dom, Oy_dom)
J0 = 5
T0 = 0.1

F1_arr = np.zeros((O_N, O_N))
F2_arr = np.zeros((O_N, O_N))
F3_arr = np.zeros((O_N, O_N))
for x, Ox in enumerate(Ox_dom):
    for y, Oy in enumerate(Oy_dom):
        F1_arr[x,y] = F_MF_cub([Ox, Oy, 0.1], J0, B1, T0)
        F2_arr[x,y] = F_MF_cub([Ox, Oy, 0.1], J0, B2, T0)
        F3_arr[x,y] = F_MF_cub([Ox, Oy, 0.1], J0, B3, T0)

cubic_tuning_fig, cubic_tuning_ax = plt.subplots(1, 3)
cubic_tuning_ax[0].contourf(Ox_mesh, Oy_mesh, F1_arr.T)
cubic_tuning_ax[1].contourf(Ox_mesh, Oy_mesh, F2_arr.T)
cubic_tuning_ax[2].contourf(Ox_mesh, Oy_mesh, F3_arr.T)

cubic_tuning_ax[0].set_xlabel(r"x Order Parameter $\langle x \rangle$")
cubic_tuning_ax[0].set_ylabel(r"y Order Parameter $\langle y \rangle$")
cubic_tuning_ax[0].set_title(r"$\phi = 0, \theta = 0$")

cubic_tuning_ax[1].set_xlabel(r"x Order Parameter $\langle x \rangle$")
cubic_tuning_ax[1].set_ylabel(r"y Order Parameter $\langle y \rangle$")
cubic_tuning_ax[1].set_title(r"$\phi = 0, \theta = \pi/2$")

cubic_tuning_ax[2].set_xlabel(r"x Order Parameter $\langle x \rangle$")
cubic_tuning_ax[2].set_ylabel(r"y Order Parameter $\langle y \rangle$")
cubic_tuning_ax[2].set_title(r"$\phi = \pi/4, \theta = \pi/4$")

# Boundary plots
# Triangular
tri_boundary_fig, tri_boundary_ax = plt.subplots()
indices = np.where(tri_boundary_arr == 1)

tri_boundary_ax.set_xlabel(J_LABEL)
tri_boundary_ax.set_ylabel(B_LABEL)

for t, T in enumerate(T_dom): 
    J_pts = [indices[0][i] for i in range(len(indices[0])) if indices[2][i] == t]
    B_pts = [indices[1][i] for i in range(len(indices[0])) if indices[2][i] == t]

    tri_boundary_ax.scatter(J_dom[J_pts], B_dom[B_pts], label = f"$T = {{{T}}}$")

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

    squ_boundary_ax.scatter(J_dom[J_pts], B_dom[B_pts], label = f"$T = {{{T}}}$")
    
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

    cub_boundary_ax.scatter(J_dom[J_pts], B_dom[B_pts], label = f"$T = {{{T}}}$")

    cub_boundary_ax.legend()

    cub_boundary_fig.tight_layout()
    cub_boundary_fig.savefig(folder_path + f"/Plots/CubicBoundary_Figure{t}")




### SAVE FIGURES

order_param_fig.set_size_inches(9,4)
order_param_fig.tight_layout()
order_param_fig.savefig(folder_path + "/Plots/CombinedOrderParameter_Figure.jpg")

tri_order_param_fig.tight_layout()
tri_order_param_fig.savefig(folder_path + "/Plots/TriangularOrderParameter_Figure.jpg")

squ_order_param_fig.tight_layout()
squ_order_param_fig.savefig(folder_path + "/Plots/SquareOrderParameter_Figure.jpg")

cub_order_param_fig.tight_layout()
cub_order_param_fig.savefig(folder_path + "/Plots/CubicOrderParameter_Figure.jpg")

cubic_tuning_fig.set_size_inches(9, 4)
cubic_tuning_fig.tight_layout()
cubic_tuning_fig.savefig(folder_path + "/Plots/CubicTuning_Figure.jpg")

