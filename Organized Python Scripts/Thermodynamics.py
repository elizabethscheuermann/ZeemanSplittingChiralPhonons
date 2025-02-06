from Base import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

### CALCULATIONS
# Parameters
B_min, B_max, B_n = 0, 5, 16
B_dom = np.linspace(B_min, B_max, B_n)

J_min, J_max, J_n = 1e-1, 2, 16
J_dom = np.linspace(J_min, J_max, J_n)

J_mesh, B_mesh = np.meshgrid(J_dom, B_dom)

T = 0

# Data Arrays
O_tri_Arr = np.zeros((J_n, B_n))
O_squ_Arr = np.zeros((J_n, B_n))
O_cub_Arr = np.zeros((J_n, B_n))

E_tri_Arr = np.zeros((J_n, B_n, 3))
E_squ_Arr = np.zeros((J_n, B_n, 4))
E_cub_Arr = np.zeros((J_n, B_n, 6))

p_tri_Arr = np.zeros((J_n, B_n, 3))
p_squ_Arr = np.zeros((J_n, B_n, 4))
p_cub_Arr = np.zeros((J_n, B_n, 6))

m_tri_Arr = np.zeros((J_n, B_n))
m_squ_Arr = np.zeros((J_n, B_n))
m_cub_Arr = np.zeros((J_n, B_n))

chi_tri_Arr = np.zeros((J_n, B_n))
chi_squ_Arr = np.zeros((J_n, B_n))
chi_cub_Arr = np.zeros((J_n, B_n))

for j, J in enumerate(J_dom):
    for b, B in enumerate(B_dom):
        # Order parameters
        O_tri = sp.optimize.minimize(F_MF_tri, x0_tri, args = (J, B, T))
        O_squ = sp.optimize.minimize(F_MF_squ, x0_squ, args = (J, B, T))
        O_cub = sp.optimize.minimize(F_MF_cub, x0_cub, args = (J, [0, 0, B], T))

        O_tri_Arr[j, b] = np.max(O_tri.x)
        O_squ_Arr[j, b] = np.max(O_squ.x)
        O_cub_Arr[j, b] = np.max(O_cub.x)

        # Energy Levels
        E_tri_Arr[j, b] = E_MF_tri(O_tri.x, J, B)
        E_squ_Arr[j, b] = E_MF_squ(O_squ.x, J, B)
        E_cub_Arr[j, b] = E_MF_cub(O_cub.x, J, [0, 0, B])
       
        # Probabilities
        p_tri_Arr[j, b] = P_MF_tri(O_tri.x, J, B, T)
        p_squ_Arr[j, b] = P_MF_squ(O_squ.x, J, B, T)
        p_cub_Arr[j, b] = P_MF_cub(O_cub.x, J, [0, 0, B], T)

        # Magnetization
        m_tri_Arr[j, b] = M_MF_tri(O_tri.x, J, B, T)
        m_squ_Arr[j, b] = M_MF_squ(O_squ.x, J, B, T)
        m_cub_Arr[j, b] = M_MF_cub(O_cub.x, J, [0, 0, B], T)

    # Magnetic Susceptibility
    chi_tri_Arr[j] = np.gradient(m_tri_Arr[j], B_dom[1] - B_dom[0])
    chi_squ_Arr[j] = np.gradient(m_squ_Arr[j], B_dom[1] - B_dom[0])
    chi_cub_Arr[j] = np.gradient(m_cub_Arr[j], B_dom[1] - B_dom[0])

### PLOTTING
fig, ax = plt.subplots(1, 3)

ax[0].contourf(J_mesh, B_mesh, m_tri_Arr.T)
ax[1].contourf(J_mesh, B_mesh, m_squ_Arr.T)
ax[2].contourf(J_mesh, B_mesh, m_cub_Arr.T)

### AXIS LABELS
ax[0].set_xlabel("Interaction Strength J/t")
ax[1].set_xlabel("Interaction Strength J/t")
ax[2].set_xlabel("Interaction Strength J/t")

ax[0].set_ylabel("Magnetic Field Strength B/t")
ax[1].set_ylabel("Magnetic Field Strength B/t")
ax[2].set_ylabel("Magnetic Field Strength B/t")


fig2, ax2 = plt.subplots(1, 3)

ax2[0].contourf(J_mesh, B_mesh, O_tri_Arr.T)
ax2[1].contourf(J_mesh, B_mesh, O_squ_Arr.T)
ax2[2].contourf(J_mesh, B_mesh, O_cub_Arr.T)

### AXIS LABELS
ax2[0].set_xlabel("Interaction Strength J/t")
ax2[1].set_xlabel("Interaction Strength J/t")
ax2[2].set_xlabel("Interaction Strength J/t")

ax2[0].set_ylabel("Magnetic Field Strength B/t")
ax2[1].set_ylabel("Magnetic Field Strength B/t")
ax2[2].set_ylabel("Magnetic Field Strength B/t")


plt.show()