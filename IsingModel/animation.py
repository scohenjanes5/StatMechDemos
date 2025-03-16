import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Ising import runMC

# Define temperature
temp = 0.1

# Run the simulation
mag, iterations = runMC(1 / temp, 20, 0, 10000, plot=True)

# Function to update the grid for each frame in the animation
def update(frame):
    mat.set_data(iterations[frame])
    line.set_data(np.arange(frame + 1), mags[: frame + 1])
    return [mat, line]


# Verify the length of iterations and magnitudes
mags = [np.sum(config) / len(iterations) for config in iterations]

# Create the figure and axes
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

# Set up the initial plots
mat = ax1.matshow(iterations[0], cmap="gray")
(line,) = ax2.plot([], [], lw=2)
ax2.set_xlim(0, len(iterations))
ax2.set_ylim(min(mags) - 0.1, max(mags) + 0.1)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Magnetization")

# Create the animation
ani = FuncAnimation(fig, update, frames=len(iterations), interval=10, blit=True)

# Show the animation
plt.show()
