import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from rich.progress import track, Progress
from concurrent.futures import ProcessPoolExecutor

# 2D Ising model simulation


def initialstate(N):
    """generates a random spin configuration for initial condition"""
    state = 2 * np.random.randint(2, size=(N, N)) - 1
    return state


def mcmove(config, beta, N):
    """Monte Carlo move using Metropolis algorithm"""
    # choose a random spin
    a = np.random.randint(0, N)
    b = np.random.randint(0, N)
    s = config[a, b]
    # calculate the interactions betwen the selected spin and neighbors. Modulus is used for periodic boundary conditions.
    neighbors = (
        config[(a + 1) % N, b]
        + config[a, (b + 1) % N]
        + config[(a - 1) % N, b]
        + config[a, (b - 1) % N]
    )
    cost = 2 * s * neighbors
    # flip the spin based on the probability of the transition.
    # If unfavorable interaction, s and neighbors have opposite signs, so cost is negative.
    flip = False
    if cost < 0 or rand() < np.exp(-beta * cost):
        config[a, b] *= -1
        flip = True
    return config, flip


def calcMag(config):
    """Magnetization of a given configuration"""
    mag = np.sum(config)
    return mag


def runMC(beta, N, eqSteps, mcSteps, plot=False):
    """Run Monte Carlo simulation for a given beta and lattice size"""
    config = initialstate(N)
    iterations = [config.copy()]
    mag = 0

    for i in range(eqSteps):
        config, flip = mcmove(config, beta, N)
        if flip and plot:
            iterations.append(config.copy())

    for i in range(mcSteps):
        config, flip = mcmove(config, beta, N)
        if flip and plot:
            iterations.append(config.copy())
        mag += calcMag(config)

    avg_mag = mag / (mcSteps * N * N)
    if plot:
        return avg_mag, iterations
    else:
        return avg_mag


if __name__ == "__main__":
    ## change these parameters for a smaller (faster) simulation
    nt = 80  #  number of temperature points
    N = 20  #  size of the lattice, N x N
    eqSteps = 400000  #  number of MC sweeps for equilibration
    mcSteps = 150000  #  number of MC sweeps for calculation

    T = np.linspace(1.50, 3.30, nt)
    # T = np.random.normal(2.3, 0.5, nt)
    M = np.zeros(nt)

    with Progress() as progress:
        MC_progress = progress.add_task("[blue]MC simulation", total=nt)
        with ProcessPoolExecutor() as executor:
            MC_futures = []
            for tt in range(nt):
                beta = 1.0 / T[tt]
                # reduce the number of steps needed for equilibration at low temperatures
                if beta > 0.5:
                    stepModifier = 0.40
                else:
                    stepModifier = 1.45

                MC_futures.append(
                    executor.submit(
                        runMC,
                        beta,
                        N,
                        int(stepModifier * eqSteps),
                        int(stepModifier * mcSteps),
                    )
                )

            while (n_finished := sum(future.done() for future in MC_futures)) <= nt:
                progress.update(MC_progress, completed=n_finished, total=nt)
                if n_finished == nt:
                    break
        M = np.array([future.result() for future in MC_futures])

    plt.plot(T, abs(M), "o", color="RoyalBlue", label="Simulation")
    plt.xlabel("Temperature (K)", fontsize=20)
    plt.ylabel("Magnetization ", fontsize=20)
    plt.axis("tight")
    plt.show()
