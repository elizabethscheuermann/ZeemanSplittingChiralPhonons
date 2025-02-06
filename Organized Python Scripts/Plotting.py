import numpy as np

tri_data = np.loadtxt("Data Runs/T=0, B=[0,5,4], J=[0.1,2,4]/CubicLattice.csv", delimiter=",", skiprows = 1)


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