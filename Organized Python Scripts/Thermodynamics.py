from Base import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import json


### CALCULATIONS
# Parameters
B_min, B_max, B_n = 0, 5, 4
B_dom = np.linspace(B_min, B_max, B_n)

J_min, J_max, J_n = 1e-1, 2, 4
J_dom = np.linspace(J_min, J_max, J_n)

T_min, T_max, T_n = 0, 0, 1
T_dom = np.linspace(T_min, T_max, T_n)

# Data Arrays
O_tri_Arr = np.zeros((J_n, B_n, T_n))
O_squ_Arr = np.zeros((J_n, B_n, T_n))
O_cub_Arr = np.zeros((J_n, B_n, T_n))

E_tri_Arr = np.zeros((J_n, B_n, T_n, 3))
E_squ_Arr = np.zeros((J_n, B_n, T_n, 4))
E_cub_Arr = np.zeros((J_n, B_n, T_n, 6))

p_tri_Arr = np.zeros((J_n, B_n, T_n, 3))
p_squ_Arr = np.zeros((J_n, B_n, T_n, 4))
p_cub_Arr = np.zeros((J_n, B_n, T_n, 6))

m_tri_Arr = np.zeros((J_n, B_n, T_n))
m_squ_Arr = np.zeros((J_n, B_n, T_n))
m_cub_Arr = np.zeros((J_n, B_n, T_n))

chi_tri_Arr = np.zeros((J_n, B_n, T_n))
chi_squ_Arr = np.zeros((J_n, B_n, T_n))
chi_cub_Arr = np.zeros((J_n, B_n, T_n))

for j, J in enumerate(J_dom):
    for b, B in enumerate(B_dom):
        for t, T in enumerate(T_dom):
            # Order parameters
            O_tri = sp.optimize.minimize(F_MF_tri, x0_tri, args = (J, B, T))
            O_squ = sp.optimize.minimize(F_MF_squ, x0_squ, args = (J, B, T))
            O_cub = sp.optimize.minimize(F_MF_cub, x0_cub, args = (J, [0, 0, B], T))

            O_tri_Arr[j, b, t] = np.max(O_tri.x)
            O_squ_Arr[j, b, t] = np.max(O_squ.x)
            O_cub_Arr[j, b, t] = np.max(O_cub.x)

            # Energy Levels
            E_tri_Arr[j, b, t] = E_MF_tri(O_tri.x, J, B)
            E_squ_Arr[j, b, t] = E_MF_squ(O_squ.x, J, B)
            E_cub_Arr[j, b, t] = E_MF_cub(O_cub.x, J, [0, 0, B])
       
            # Probabilities
            p_tri_Arr[j, b, t] = P_MF_tri(O_tri.x, J, B, T)
            p_squ_Arr[j, b, t] = P_MF_squ(O_squ.x, J, B, T)
            p_cub_Arr[j, b, t] = P_MF_cub(O_cub.x, J, [0, 0, B], T)

            # Magnetization
            m_tri_Arr[j, b, t] = M_MF_tri(O_tri.x, J, B, T)
            m_squ_Arr[j, b, t] = M_MF_squ(O_squ.x, J, B, T)
            m_cub_Arr[j, b, t] = M_MF_cub(O_cub.x, J, [0, 0, B], T)

    # Magnetic Susceptibility
    for t, T in enumerate(T_dom):
        chi_tri_Arr[j, :, t] = np.gradient(m_tri_Arr[j, :, t], B_dom[1] - B_dom[0])
        chi_squ_Arr[j, :, t] = np.gradient(m_squ_Arr[j, :, t], B_dom[1] - B_dom[0])
        chi_cub_Arr[j, :, t] = np.gradient(m_cub_Arr[j, :, t], B_dom[1] - B_dom[0])


### WRITE TO FILES
# Create Filepaths

# Folder
folder_path = "T=[" + str(T_min)+","+str(T_max)+","+str(T_n) +"], " if T_n != 1 else "T=" + str(T_min) + ", "
folder_path += "B=["+str(B_min)+","+str(B_max)+","+str(B_n)+"], " if B_n != 1 else "B=" + str(B_min) + ", "
folder_path += "J=["+str(J_min)+","+str(J_max)+","+str(J_n)+"]" if J_n != 1 else "J=" + str(J_min) + ", "

if not (os.path.isdir("Data Runs/" + folder_path)):
    os.mkdir("Data Runs/" + folder_path)

# Lattices
tri_data_filepath = "Data Runs/" + folder_path + "/TriangularLattice.csv"
squ_data_filepath = "Data Runs/" + folder_path + "/SquareLattice.csv"
cub_data_filepath = "Data Runs/" + folder_path + "/CubicLattice.csv"

# Triangular Write Loop
with open(tri_data_filepath, 'w') as tri:
    tri.write("J, B, T, O, m, chi, E0, E1, E2, p0, p1, p2 \n")
    for j, J in enumerate(J_dom):
        for b, B in enumerate(B_dom):
            for t, T in enumerate(T_dom):
                line = str(J) + ", " + str(B) + ", " + str(T) + ", " + str(O_tri_Arr[j,b,t])
                line += ", " + str(m_tri_Arr[j,b,t]) + ", " + str(chi_tri_Arr[j,b,t]) 
                line += ", " + str(E_tri_Arr[j,b,t][0]) + ", " + str(E_tri_Arr[j,b,t][1]) + ", " + str(E_tri_Arr[j,b,t][2])
                line += ", " + str(p_tri_Arr[j,b,t][0]) + ", " + str(p_tri_Arr[j,b,t][1]) + ", " + str(p_tri_Arr[j,b,t][2]) + "\n"
                tri.write(line)

# Square Write Loop
with open(squ_data_filepath, 'w') as squ:
    squ.write("J, B, T, O, m, chi, E0, E1, E2, E3, p0, p1, p2, p3 \n")
    for j, J in enumerate(J_dom):
        for b, B in enumerate(B_dom):
            for t, T in enumerate(T_dom):
                line = str(J) + ", " + str(B) + ", " + str(T) + ", " + str(O_squ_Arr[j ,b])
                line += ", " + str(m_squ_Arr[j,b,t]) + ", " + str(chi_squ_Arr[j,b,t]) 
                line += ", " + str(E_squ_Arr[j,b,t][0]) + ", " + str(E_squ_Arr[j,b,t][1]) + ", " + str(E_squ_Arr[j,b,t][2]) + ", " + str(E_squ_Arr[j,b,t][3])
                line += ", " + str(p_squ_Arr[j,b,t][0]) + ", " + str(p_squ_Arr[j,b,t][1]) + ", " + str(p_squ_Arr[j,b,t][2]) + ", " + str(p_squ_Arr[j,b,t][3]) + "\n"
                squ.write(line)

# Cubic Write Loop
with open(cub_data_filepath, 'w') as cub:
    cub.write("J, B, T, O, m, chi, E0, E1, E2, E3, E4, E5, p0, p1, p2, p3, p4, p5 \n")
    for j, J in enumerate(J_dom):
        for b, B in enumerate(B_dom):
            for t, T in enumerate(T_dom):
                line = str(J) + ", " + str(B) + ", " + str(T) + ", " + str(O_cub_Arr[j ,b])
                line += ", " + str(m_cub_Arr[j,b,t]) + ", " + str(chi_cub_Arr[j,b,t]) 
                line += ", " + str(E_cub_Arr[j,b,t][0]) + ", " + str(E_cub_Arr[j,b,t][1]) + ", " + str(E_cub_Arr[j,b,t][2]) + ", " + str(E_cub_Arr[j,b,t][3]) + ", " + str(E_cub_Arr[j,b,t][4]) + ", " + str(E_cub_Arr[j,b,t][5])
                line += ", " + str(p_cub_Arr[j,b,t][0]) + ", " + str(p_cub_Arr[j,b,t][1]) + ", " + str(p_cub_Arr[j,b,t][2]) + ", " + str(p_cub_Arr[j,b,t][3]) + ", " + str(p_cub_Arr[j,b,t][4]) + ", " + str(p_cub_Arr[j,b,t][5]) + "\n"
                cub.write(line)

# Create .json summary file
with open("Data Runs/" + folder_path + "/parameters.json", "w") as json_file:
    json_data = {"T_min": T_min, "T_max": T_max, "T_n": T_n,  "B_min": B_min, "B_max": B_max, "B_n": B_n, "J_min": J_min, "J_max": J_max, "J_n": J_n}
    json.dump(json_data, json_file)
