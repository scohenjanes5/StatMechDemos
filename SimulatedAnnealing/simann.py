#Simulated Annealing of Argon Atoms.
#Start with 2 atoms, so we only deal with one coordinate.

import numpy as np
import matplotlib.pyplot as plt

#Function to calculate the Lennard-Jones potential:
def LJ(r):
    return 4*eps*((sig/r)**12 - (sig/r)**6)

#Function to choose new r
def newr(r,T):
    pururbation = np.random.uniform(-nudge,nudge)
    return r + pururbation

#Constants:
#Radius of Argon atom:
rad = 1
#Lennard-Jones potential parameters:
eps = 1
sig = 1
#Initial temperature:
T = 5
nudge = .05
cooling_rate = 0.99

#Initial r is random:
r = np.random.uniform(rad,10)

Energies = []
Radii = []
failures = 0
#failure_array = []

while failures < 300:
    #Calculate energy of current r:
    E = LJ(r)
    #Choose new r:
    rnew = newr(r,T)
    #Calculate energy of new r:
    Enew = LJ(rnew)
    #Calculate change in energy:
    dE = Enew - E
    #If dE < 0, accept new r:
    if dE < 0:
        r = rnew
        E = Enew
        failures = 0
    #If dE > 0, accept new r with probability exp(-dE/T):
    elif np.random.uniform(0,1) < np.exp(-dE/T):
        r = rnew
        E = Enew
        failures = 0
    else:
        failures += 1
    #Append energy to list:
    Energies.append(E)
    Radii.append(r)
    #Decrease temperature:
    T *= cooling_rate
#    failure_array.append(failures)

fig, (ax1, ax2) = plt.subplots(2, 1)
x = np.linspace(0.95,max(r+2,5),1000)
ax1.plot(x,LJ(x))
ax1.scatter(r,LJ(r))
ax1.annotate('Final Position', xy=(r,LJ(r)), textcoords='data', xytext=(2,-0.5), arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
ax2.plot(Energies)
ax2.plot(Radii)
ax2.legend(['Energy','Radius'])
#ax3.plot(failure_array)
#ax3.set_xlabel('Iteration')
#ax3.set_ylabel('Number of Failures')
plt.show()
