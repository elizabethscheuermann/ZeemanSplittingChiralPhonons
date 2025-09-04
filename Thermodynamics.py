from Base import *
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import json
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

args = parser.parse_args()

### CALCULATIONS
# Parameters
B_min, B_max, B_n = float(args.Bmin), float(args.Bmax), int(args.Bn)
B_dom = np.linspace(B_min, B_max, B_n)

J_min, J_max, J_n = float(args.Jmin), float(args.Jmax), int(args.Jn)
J_dom = np.linspace(J_min, J_max, J_n)

T_min, T_max, T_n = float(args.Tmin), float(args.Tmax), int(args.Tn)
T_dom = np.linspace(T_min, T_max, T_n)

def PrintProgress(j, b, t,n_=50):
        os.system("clear")
        percent = (j*len(B_dom)*len(T_dom) + b * len(T_dom) + t)/(len(J_dom) * len(B_dom) * len(T_dom))
        hash_str = int(n_ *percent)*"#"
        blank_str = int(n_ * (1-percent)) * "-"
        print("[", hash_str, blank_str, "]", str(100*percent) + "%")


Q_tri = 1
Q_squ = 4

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

freq_tri_Arr = np.zeros((J_n, B_n, T_n, 2))
freq_squ_Arr = np.zeros((J_n, B_n, T_n, 3))


squ_phase_bound = []
tri_phase_bound = []
cub_phase_bound = []

for j, J in enumerate(J_dom):
    for b, B in enumerate(B_dom):
        for t, T in enumerate(T_dom):
            PrintProgress(j,b,t)
            # Order parameters
            O_tri = sp.optimize.minimize(F_MF_tri, x0_tri, args = (J, B, T))
            O_squ = sp.optimize.minimize(F_MF_squ, x0_squ, args = (J, B, T))
            O_cub = sp.optimize.minimize(F_MF_cub, x0_cub, args = (J, [0, 0, B], T))

            O_tri_Arr[j, b, t] = np.max(O_tri.x)
            O_squ_Arr[j, b, t] = np.max(O_squ.x)
            O_cub_Arr[j, b, t] = np.max(O_cub.x)

            # Energy Levels
            E_tri_Arr[j, b, t] = np.real(E_MF_tri(O_tri.x, J, B))
            E_squ_Arr[j, b, t] = np.real(E_MF_squ(O_squ.x, J, B))
            E_cub_Arr[j, b, t] = np.real(E_MF_cub(O_cub.x, J, [0, 0, B]))
       
            # Probabilities
            p_tri_Arr[j, b, t] = np.real(P_MF_tri(O_tri.x, J, B, T))
            p_squ_Arr[j, b, t] = np.real(P_MF_squ(O_squ.x, J, B, T))
            p_cub_Arr[j, b, t] = np.real(P_MF_cub(O_cub.x, J, [0, 0, B], T))

            # Magnetization
            m_tri_Arr[j, b, t] = np.real(M_MF_tri(O_tri.x, J, B, T))
            m_squ_Arr[j, b, t] = np.real(M_MF_squ(O_squ.x, J, B, T))
            m_cub_Arr[j, b, t] = np.real(M_MF_cub(O_cub.x, J, [0, 0, B], T))
        
            # Phonon Dynamics
            freq_tri_Arr[j, b, t, :] = np.real(PD_MF_tri(O_tri.x, J, B, T, Q_tri))
            freq_squ_Arr[j, b, t, :] = np.real(PD_MF_squ(O_squ.x, J, B, T, Q_squ))

    for t, T in enumerate(T_dom):
        # Magnetic Susceptibility
        chi_tri_Arr[j, :, t] = np.gradient(m_tri_Arr[j, :, t], B_dom[1] - B_dom[0])
        chi_squ_Arr[j, :, t] = np.gradient(m_squ_Arr[j, :, t], B_dom[1] - B_dom[0])
        chi_cub_Arr[j, :, t] = np.gradient(m_cub_Arr[j, :, t], B_dom[1] - B_dom[0])


for b, B in enumerate(B_dom):
    for t, T in enumerate(T_dom):
        # Phase Boundary
        tri_j = np.argmax(np.gradient(O_tri_Arr[:, b, t], J_dom[1] - J_dom[0]))
        tri_phase_bound.append([tri_j, b, t])

        squ_j = np.argmax(np.gradient(O_squ_Arr[:, b, t], J_dom[1] - J_dom[0]))
        squ_phase_bound.append([squ_j, b, t])

        cub_j = np.argmax(np.gradient(O_cub_Arr[:, b, t], J_dom[1] - J_dom[0]))
        cub_phase_bound.append([cub_j,b,t])


### WRITE TO FILES
# Create Filepaths

# Folder

folder_path = "J=["+str(J_min)+","+str(J_max)+","+str(J_n)+"]," if J_n != 1 else "J=[" + str(J_min) + "],"
folder_path += "B=["+str(B_min)+","+str(B_max)+","+str(B_n)+"]," if B_n != 1 else "B=[" + str(B_min) + "],"
folder_path += "T=[" + str(T_min)+","+str(T_max)+","+str(T_n) +"]" if T_n != 1 else "T=[" + str(T_min) + "]"

if not (os.path.isdir("DataRuns/" + folder_path)):
    os.mkdir("DataRuns/" + folder_path)

# Lattices
tri_data_filepath = "DataRuns/" + folder_path + "/TriangularLattice_Data.csv"
squ_data_filepath = "DataRuns/" + folder_path + "/SquareLattice_Data.csv"
cub_data_filepath = "DataRuns/" + folder_path + "/CubicLattice_Data.csv"


# Triangular Write Loop
with open(tri_data_filepath, 'w') as tri:
    tri.write("J, B, T, O, m, chi, E0, E1, E2, p0, p1, p2, w1, w2, boundary \n")
    for j, J in enumerate(J_dom):
        for b, B in enumerate(B_dom):
            for t, T in enumerate(T_dom):
                line = str(J) + ", " + str(B) + ", " + str(T) + ", " + str(O_tri_Arr[j,b,t])
                line += ", " + str(m_tri_Arr[j,b,t]) + ", " + str(chi_tri_Arr[j,b,t]) 
                line += ", " + str(E_tri_Arr[j,b,t][0]) + ", " + str(E_tri_Arr[j,b,t][1]) + ", " + str(E_tri_Arr[j,b,t][2])
                line += ", " + str(p_tri_Arr[j,b,t][0]) + ", " + str(p_tri_Arr[j,b,t][1]) + ", " + str(p_tri_Arr[j,b,t][2])
                line += ", " + str(freq_tri_Arr[j,b,t][0]) + ", " + str(freq_tri_Arr[j,b,t][1])
                if [j, b, t] in tri_phase_bound:
                    line+= ", 1 \n"
                else:
                    line+= ", 0 \n"

                tri.write(line)

# Square Write Loop
with open(squ_data_filepath, 'w') as squ:
    squ.write("J, B, T, O, m, chi, E0, E1, E2, E3, p0, p1, p2, p3, w1, w2, w3, w4, w5, w6, boundary \n")
    for j, J in enumerate(J_dom):
        for b, B in enumerate(B_dom):
            for t, T in enumerate(T_dom):
                line = str(J) + ", " + str(B) + ", " + str(T) + ", " + str(O_squ_Arr[j,b,t])
                line += ", " + str(m_squ_Arr[j,b,t]) + ", " + str(chi_squ_Arr[j,b,t]) 
                line += ", " + str(E_squ_Arr[j,b,t][0]) + ", " + str(E_squ_Arr[j,b,t][1]) + ", " + str(E_squ_Arr[j,b,t][2]) + ", " + str(E_squ_Arr[j,b,t][3])
                line += ", " + str(p_squ_Arr[j,b,t][0]) + ", " + str(p_squ_Arr[j,b,t][1]) + ", " + str(p_squ_Arr[j,b,t][2]) + ", " + str(p_squ_Arr[j,b,t][3])
                line += ", " + str(freq_squ_Arr[j,b,t][0]) + ", " + str(freq_squ_Arr[j,b,t][1]) +", " + str(freq_squ_Arr[j,b,t][2])
                if [j, b, t] in squ_phase_bound:
                    line+= ", 1 \n"
                else:
                    line+= ", 0 \n"
                squ.write(line)

                
# Cubic Write Loop
with open(cub_data_filepath, 'w') as cub:
    cub.write("J, B, T, O, m, chi, E0, E1, E2, E3, E4, E5, p0, p1, p2, p3, p4, p5, boundary \n")
    for j, J in enumerate(J_dom):
        for b, B in enumerate(B_dom):
            for t, T in enumerate(T_dom):
                line = str(J) + ", " + str(B) + ", " + str(T) + ", " + str(O_cub_Arr[j,b,t])
                line += ", " + str(m_cub_Arr[j,b,t]) + ", " + str(chi_cub_Arr[j,b,t]) 
                line += ", " + str(E_cub_Arr[j,b,t][0]) + ", " + str(E_cub_Arr[j,b,t][1]) + ", " + str(E_cub_Arr[j,b,t][2]) + ", " + str(E_cub_Arr[j,b,t][3]) + ", " + str(E_cub_Arr[j,b,t][4]) + ", " + str(E_cub_Arr[j,b,t][5])
                line += ", " + str(p_cub_Arr[j,b,t][0]) + ", " + str(p_cub_Arr[j,b,t][1]) + ", " + str(p_cub_Arr[j,b,t][2]) + ", " + str(p_cub_Arr[j,b,t][3]) + ", " + str(p_cub_Arr[j,b,t][4]) + ", " + str(p_cub_Arr[j,b,t][5])
                if [j, b, t] in cub_phase_bound:
                    line+= ", 1 \n"
                else:
                    line+= ", 0 \n"
                cub.write(line)



# Create .json summary file
with open("DataRuns/" + folder_path + "/parameters.json", "w") as json_file:
    json_data = {"T_min": T_min, "T_max": T_max, "T_n": T_n,  "B_min": B_min, "B_max": B_max, "B_n": B_n, "J_min": J_min, "J_max": J_max, "J_n": J_n}
    json.dump(json_data, json_file)
