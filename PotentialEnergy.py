import numpy as np
import matplotlib.pyplot as plt

def potential_energy(x,y, k=1, alpha = 1):
    return 0.5 * k * (x**2 + y**2)*np.cos(4*np.arctan2(y,x)) + alpha*(x**2 + y**2)**2

k = -5
alpha = 1
rmax = 1.5

# Create a grid of points
x = np.linspace(-rmax*np.sqrt(-k/(4*alpha)), rmax*np.sqrt(-k/(4*alpha)), 400)
y = np.linspace(-rmax*np.sqrt(-k/(4*alpha)), rmax*np.sqrt(-k/(4*alpha)), 400)
X, Y = np.meshgrid(x, y)

# Calculate the potential energy at each point
Z = potential_energy(X, Y, k = k, alpha = alpha)
# Plot the potential energy surface
fig, ax = plt.subplots()

fig.set_size_inches(8, 6)

ax.set_xlabel('x', fontsize = 16)
ax.set_ylabel('y', fontsize = 16)
ax.set_title(r'$V(r,\theta)$', fontsize = 20)

plot = ax.contourf(X, Y, Z, levels = 30, cmap='viridis')
cbar = fig.colorbar(plot)

# Save figure
plt.savefig('PotentialEnergy.png', dpi=300)

