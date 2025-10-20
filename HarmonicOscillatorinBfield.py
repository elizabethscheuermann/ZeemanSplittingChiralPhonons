import numpy as np
import matplotlib.pyplot as plt

def E(n, m, omega0, omegaB):
	return np.sqrt(omega0**2 + omegaB**2)*n + omegaB * m

omegaB_dom = np.linspace(0, 1, 100)

fig, ax = plt.subplots()

for n in range(2):
	for m in range(-n, n+1):
		E_nm = E(n, m, 1, omegaB_dom)
		ax.plot(omegaB_dom, E_nm, label=f'n={n}, m={m}')

ax.set_xlabel(r'$\omega_B$')
ax.set_ylabel('Energy Levels E(n,m)')
ax.legend()

plt.show()
