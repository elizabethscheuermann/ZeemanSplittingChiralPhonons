import numpy as np
import json
import matplotlib.pyplot as plt
import os

folder_path = "Data Runs/T=0, B=[0,5,64], J=[0.1,2,16]/"

### GET PARAMETERS
with open(folder_path + "parameters.json", "r") as params_file:
    params = json.load(params_file)


J_dom = np.linspace(params["J_min"], params["J_max"], params["J_n"])
B_dom = np.linspace(params["B_min"], params["B_max"], params["B_n"])
T_dom = np.linspace(params["T_min"], params["T_max"], params["T_n"])

J_mesh, B_mesh = np.meshgrid(J_dom, B_dom)


### GET DATA
# Triangular Lattice
tri_data = np.loadtxt(folder_path + "TriangularLattice.csv", delimiter = ",", skiprows = 1)
ox_tri_arr = np.reshape(tri_data.T[3], (params["J_n"], params["B_n"], params["T_n"]))
m_tri_arr = np.reshape(tri_data.T[4], (params["J_n"], params["B_n"], params["T_n"]))
chi_tri_arr = np.reshape(tri_data.T[5], (params["J_n"], params["B_n"], params["T_n"]))

E0_tri_arr = np.reshape(tri_data.T[6], (params["J_n"], params["B_n"], params["T_n"]))
E1_tri_arr = np.reshape(tri_data.T[7], (params["J_n"], params["B_n"], params["T_n"]))
E2_tri_arr = np.reshape(tri_data.T[8], (params["J_n"], params["B_n"], params["T_n"]))

p0_tri_arr = np.reshape(tri_data.T[9], (params["J_n"], params["B_n"], params["T_n"]))
p1_tri_arr = np.reshape(tri_data.T[10], (params["J_n"], params["B_n"], params["T_n"]))
p2_tri_arr = np.reshape(tri_data.T[11], (params["J_n"], params["B_n"], params["T_n"]))

# Square Lattice
squ_data = np.loadtxt(folder_path + "SquareLattice.csv", delimiter = ",", skiprows = 1)
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

# Cubic Lattice
cub_data = np.loadtxt(folder_path + "CubicLattice.csv", delimiter = ",", skiprows = 1)
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


### PLOTTING
J0_index = 3
B0_index = 1
T0_index = 0

# Energy Levels
tri_energies_fig, tri_energies_ax = plt.subplots()
tri_energies_ax.plot(B_dom, E0_tri_arr[J0_index, :, T0_index], label = r'$|0\rangle$')
tri_energies_ax.plot(B_dom, E1_tri_arr[J0_index, :, T0_index], label = r'$|-1\rangle$')
tri_energies_ax.plot(B_dom, E2_tri_arr[J0_index, :, T0_index], label = r'$|1\rangle$')

tri_energies_ax.set_xlabel(r"Magnetic Field Strength B/t")
tri_energies_ax.set_ylabel(r"Energy")
tri_energies_ax.set_title(r"Triangular Lattice Energy Levels, $J = $" + str(J_dom[J0_index]) + r", $T = $" + str(T_dom[T0_index]))
tri_energies_ax.legend()

squ_energies_fig, squ_energies_ax = plt.subplots()
squ_energies_ax.plot(B_dom, E0_squ_arr[J0_index, :, T0_index], label = r'$|0\rangle$')
squ_energies_ax.plot(B_dom, E1_squ_arr[J0_index, :, T0_index], label = r'$|-1\rangle$')
squ_energies_ax.plot(B_dom, E2_squ_arr[J0_index, :, T0_index], label = r'$|1\rangle$')
squ_energies_ax.plot(B_dom, E3_squ_arr[J0_index, :, T0_index], label = r'$|2\rangle + |-2\rangle$')

squ_energies_ax.set_xlabel(r"Magnetic Field Strength B/t")
squ_energies_ax.set_ylabel(r"Energy")
squ_energies_ax.set_title(r"Square Lattice Energy Levels, $J = $" + str(J_dom[J0_index]) + r", $T = $" + str(T_dom[T0_index]))
squ_energies_ax.legend()

cub_energies_fig, cub_energies_ax = plt.subplots()
cub_energies_ax.plot(B_dom, E0_cub_arr[J0_index, :, T0_index], label = r'$|0\rangle$')
cub_energies_ax.plot(B_dom, E1_cub_arr[J0_index, :, T0_index], label = r'$|-1\rangle$')
cub_energies_ax.plot(B_dom, E2_cub_arr[J0_index, :, T0_index], label = r'$|1\rangle$')
cub_energies_ax.plot(B_dom, E3_cub_arr[J0_index, :, T0_index], label = r'$|2\rangle + |-2\rangle$')

cub_energies_ax.plot(B_dom, E4_cub_arr[J0_index, :, T0_index], label = r'$|2\rangle + |-2\rangle$')
cub_energies_ax.plot(B_dom, E5_cub_arr[J0_index, :, T0_index], label = r'$|2\rangle + |-2\rangle$')
### LABELS INCORRECT

cub_energies_ax.set_xlabel(r"Magnetic Field Strength B/t")
cub_energies_ax.set_ylabel(r"Energy")
cub_energies_ax.set_title(r"Cubic Lattice Energy Levels, $J = $" + str(J_dom[J0_index]) + r", $T = $" + str(T_dom[T0_index]))
cub_energies_ax.legend()

# Probabilities
probs_fig, probs_ax = plt.subplots()
probs_ax.plot(B_dom, p0_squ_arr[J0_index, :, T0_index], label = r'$|0\rangle$')
probs_ax.plot(B_dom, p1_squ_arr[J0_index, :, T0_index], label = r'$|-1\rangle$')
probs_ax.plot(B_dom, p2_squ_arr[J0_index, :, T0_index], label = r'$|1\rangle$')
probs_ax.plot(B_dom, p3_squ_arr[J0_index, :, T0_index], label = r'$|2\rangle + |-2\rangle$')

probs_ax.set_xlabel(r"Magnetic Field Strength B/t")
probs_ax.set_ylabel(r"Probability")
probs_ax.set_title(r"Square State Probabilities, $J = $" + str(J_dom[J0_index]) + r", $T = $" + str(T_dom[T0_index]))
probs_ax.legend()

# Magnetization
magn_fig, magn_ax = plt.subplots()
magn_ax.plot(B_dom, m_tri_arr[J0_index, :, T0_index], label = r"Triangular")
magn_ax.plot(B_dom, m_squ_arr[J0_index, :, T0_index], label = r"Square")
magn_ax.plot(B_dom, m_cub_arr[J0_index, :, T0_index], label = r"Cubic")

magn_ax.set_xlabel(r"Magnetic Field Strength B/t")
magn_ax.set_ylabel(r"Magnetization m")
magn_ax.set_title(r"Model Magnetizations, $J = $" + str(J_dom[J0_index]) + ", $T = $" + str(T_dom[T0_index]))
magn_ax.legend()

# Magnetic Susceptibility vs. J
chi_fig, chi_ax = plt.subplots()
chi_ax.plot(J_dom, chi_tri_arr[:, B0_index, T0_index], label = r"Triangular")
chi_ax.plot(J_dom, chi_squ_arr[:, B0_index, T0_index], label = r"Square")
# chi_ax.plot(J_dom, chi_cub_arr[:, B0_index, T0_index], label = r"Cubic") ### EXCLUDING CUBIC BECAUSE ITS NOT WORKING

chi_ax.set_xlabel(r"Interaction Strength J/t")
chi_ax.set_ylabel(r"Magnetic Susceptibility $\chi$")
chi_ax.set_title(r"Model Magnetic Susceptibilities, $B = $" + str(B_dom[B0_index]) + ", $T = $" + str(T_dom[T0_index]))
chi_ax.legend()

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

order_param_ax[0].set_xlabel(r"Interaction Strength J/t")
order_param_ax[0].set_ylabel(r"Magnetic Field Strength B/t")
order_param_ax[0].set_title(r"Triangular Lattice")

order_param_ax[1].set_xlabel(r"Interaction Strength J/t")
order_param_ax[1].set_ylabel(r"Magnetic Field Strength B/t")
order_param_ax[1].set_title(r"Square Lattice")

order_param_ax[2].set_xlabel(r"Interaction Strength J/t")
order_param_ax[2].set_ylabel(r"Magnetic Field Strength B/t")
order_param_ax[2].set_title(r"Cubic Lattice")

order_param_fig.suptitle(r"Mean Field Order Parameter, $T = $" + str(T_dom[T0_index]))


### SAVE FIGURES
if not (os.path.isdir(folder_path + "/Plots")):
    os.mkdir(folder_path + "/Plots")

tri_energies_fig.tight_layout()
tri_energies_fig.savefig(folder_path + "/Plots/TriangularEnergies_Figure.jpg")

squ_energies_fig.tight_layout()
squ_energies_fig.savefig(folder_path + "/Plots/SquareEnergies_Figure.jpg")

cub_energies_fig.tight_layout()
cub_energies_fig.savefig(folder_path + "/Plots/CubicEnergies_Figure.jpg")


probs_fig.tight_layout()
probs_fig.savefig(folder_path + "/Plots/SquareProbabilities_Figure.jpg")

magn_fig.tight_layout()
magn_fig.savefig(folder_path + "/Plots/Magnetization_Figure.jpg")

chi_fig.tight_layout()
chi_fig.savefig(folder_path + "/Plots/Susceptibility_Figure.jpg")

order_param_fig.tight_layout()
order_param_fig.savefig(folder_path + "/Plots/OrderParameter_Figure.jpg")


