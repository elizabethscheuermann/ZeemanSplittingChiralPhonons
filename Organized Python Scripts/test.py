from Base import *
import scipy as sp
import matplotlib.pyplot as plt
import Phonon_Dynamics

J = .5
B_dom = np.linspace(0, 5, 64)
T = 0.1
Q = 4

norms_arr = []
norms2_arr = []
freqs_arr = []
freqs2_arr = []
for B in B_dom:
    O_tri = sp.optimize.minimize(F_MF_tri, x0_tri, args = (J, B, T))
    O_squ = sp.optimize.minimize(F_MF_squ, x0_squ, args = (J, B, T))
    freqs, norms = PD_MF_tri(O_tri.x, J, B, T, Q)
    freqs2, norms2 = PD_MF_squ(O_squ.x, J, B, T, Q)

    # Sort by norms
    #norms = np.real(np.array(norms))
    #ids = norms.argsort()[::-1]
    #freqs = freqs[ids][:3]
    
    # Then by frequency again
    ids = freqs.argsort()

    #freqs = freqs[ids]

    norms_arr.append(norms)
    freqs_arr.append(freqs)

    norms2_arr.append(norms2)
    freqs2_arr.append(freqs2)

norms_arr = np.abs(np.array(norms_arr))
freqs_arr = np.real(np.array(freqs_arr))

norms2_arr = np.abs(np.array(norms2_arr))
freqs2_arr = np.abs(np.array(freqs2_arr))

fig, ax = plt.subplots(2,2)
for i in range(3):
    ax[0][0].plot(B_dom, freqs_arr[:, i], label = i)
    ax[1][0].plot(B_dom, freqs2_arr[:, i], label = str(i) + "'")
    ax[0][1].plot(B_dom, norms_arr[:, i], label = i)
    ax[1][1].plot(B_dom, norms2_arr[:,i], label = str(i) + "'")

ax[0][0].set_title("tri_freqs")
ax[1][0].set_title("squ_freqs")
ax[0][1].set_title("tri_norms")
ax[1][1].set_title("squ_norms")
ax[0][0].legend()
ax[1][0].legend()
ax[0][1].legend()
ax[1][1].legend()
plt.show()

