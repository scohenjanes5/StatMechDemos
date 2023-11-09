import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
from rich.progress import track, Progress
from concurrent.futures import ProcessPoolExecutor

#2D Ising model simulation

def initialstate(N):   
    ''' generates a random spin configuration for initial condition'''
    state = 2*np.random.randint(2, size=(N,N))-1
    return state


def mcmove(config, beta):
    '''Monte Carlo move using Metropolis algorithm '''
    for i in range(N):
        for j in range(N):
                #choose a random spin
                a = np.random.randint(0, N)
                b = np.random.randint(0, N)
                s =  config[a, b]
                #calculate the energy change if we flip this spin. Modulus is used for periodic boundary conditions.
                neighbors = config[(a+1)%N,b] + config[a,(b+1)%N] + config[(a-1)%N,b] + config[a,(b-1)%N]
                cost = 2*s*neighbors
                #flip the spin based on the probability of the transition
                if rand() < np.exp(-beta*cost):
                    s *= -1
                config[a, b] = s
    return config

def calcMag(config):
    '''Magnetization of a given configuration'''
    mag = np.sum(config)
    return mag

def runMC(beta, N, eqSteps, mcSteps):
    '''Run Monte Carlo simulation for a given beta and lattice size'''
    config = initialstate(N)
    mag = 0
    #equilibrate
    for i in range(eqSteps):
        config = mcmove(config, beta)
    #start recording
    for i in range(mcSteps):
        config = mcmove(config, beta)
        mag += calcMag(config)
    return mag/(mcSteps*N*N)

## change these parameters for a smaller (faster) simulation 
nt      = 88         #  number of temperature points
N       = 12        #  size of the lattice, N x N
eqSteps = 1000       #  number of MC sweeps for equilibration
mcSteps = 700        #  number of MC sweeps for calculation

T = np.linspace(1.53, 3.28, nt); 
M = np.zeros(nt)

#----------------------------------------------------------------------
#  MAIN PART OF THE CODE
#----------------------------------------------------------------------
with Progress() as progress:
    MC_progress = progress.add_task("[red]MC simulation", total=nt)
    with ProcessPoolExecutor() as executor:
        MC_futures = []
        for tt in range(nt):
            beta=1.0/T[tt]
            #reduce the number of steps needed for equilibration at low temperatures
            if beta > 0.5:
                stepModifier = 0.6
            else:
                stepModifier = 1

            MC_futures.append(executor.submit(runMC, beta, N, int(stepModifier * eqSteps), int(stepModifier * mcSteps)))

        while (n_finished := sum(future.done() for future in MC_futures)) < nt:
            progress.update(MC_progress, completed=n_finished, total=nt)
            if n_finished == nt:
                break
    M = np.array([future.result() for future in MC_futures])

plt.plot(T, abs(M), 'o', color='RoyalBlue', label='Simulation')
plt.xlabel("Temperature (T)", fontsize=20); 
plt.ylabel("Magnetization ", fontsize=20);   plt.axis('tight');

plt.show()

