import numpy as np
import json
import matplotlib.pyplot as plt

folder_path = "Data Runs/T=0, B=[0,5,17], J=[0.1,2,16]/"

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
fig, ax = plt.subplots()
ax.plot(B_dom, E0_squ_arr[10, :, 0])
ax.plot(B_dom, E1_squ_arr[10, :, 0])
ax.plot(B_dom, E2_squ_arr[10, :, 0])
ax.plot(B_dom, E3_squ_arr[10, :, 0])


plt.show()
### PLOTTING
# fig, ax = plt.subplots(1, 3)

# ax[0].contourf(J_mesh, B_mesh, chi_tri_Arr.T)
# ax[1].contourf(J_mesh, B_mesh, chi_squ_Arr.T)
# ax[2].contourf(J_mesh, B_mesh, chi_cub_Arr.T)

# ### AXIS LABELS
# ax[0].set_xlabel("Interaction Strength J//t")
# ax[1].set_xlabel("Interaction Strength J//t")
# ax[2].set_xlabel("Interaction Strength J//t")

# ax[0].set_ylabel("Magnetic Field Strength B//t")
# ax[1].set_ylabel("Magnetic Field Strength B//t")
# ax[2].set_ylabel("Magnetic Field Strength B//t")


# fig2, ax2 = plt.subplots(1, 3)

# ax2[0].contourf(J_mesh, B_mesh, O_tri_Arr.T)
# ax2[1].contourf(J_mesh, B_mesh, O_squ_Arr.T)
# ax2[2].contourf(J_mesh, B_mesh, O_cub_Arr.T)

# ### AXIS LABELS
# ax2[0].set_xlabel("Interaction Strength J//t")
# ax2[1].set_xlabel("Interaction Strength J//t")
# ax2[2].set_xlabel("Interaction Strength J//t")

# ax2[0].set_ylabel("Magnetic Field Strength B//t")
# ax2[1].set_ylabel("Magnetic Field Strength B//t")
# ax2[2].set_ylabel("Magnetic Field Strength B//t")


# plt.show()