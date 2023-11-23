#Simulated Annealing of Argon Atoms.
#Start with 2 atoms, so we only deal with one coordinate.

import numpy as np
import matplotlib.pyplot as plt

#Function to calculate the Lennard-Jones potential:
def LJ(r):
    return 4*eps*((sig/r)**12 - (sig/r)**6)

#Function to choose new r
def newr(r,T):
    pururbation = np.random.uniform(-Scaling_Factor,Scaling_Factor)
    return r + pururbation

#Constants:
#Radius of Argon atom:
rad = 1
#Lennard-Jones potential parameters:
eps = 1
sig = 1
#Initial temperature:
T = 5
Scaling_Factor = .05

#Initial r is random:
r = np.random.uniform(rad,2)

Energies = []
Radii = []
dEs = [1]
Ts = [T]

#while dEs[-1] > -1e-9:
for i in range(10000):
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
    #If dE > 0, accept new r with probability exp(-dE/T):
    else:
        if np.random.uniform(0,1) < np.exp(-dE/T):
            r = rnew
            E = Enew
    #Append energy to list:
    Energies.append(E)
    Radii.append(r)
    #Decrease temperature:
    T *= 0.99
    Ts.append(T)
    dEs.append(dE)

fig, (ax1, ax2) = plt.subplots(2, 1)
x = np.linspace(0.95,max(r+2,5),1000)
ax1.plot(x,LJ(x))
ax1.scatter(r,LJ(r))
ax1.annotate('Final Position', xy=(r,LJ(r)), textcoords='data', xytext=(2,-0.5), arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
ax2.plot(Energies)
ax2.plot(Radii)
ax2.legend(['Energy','Radius'])
plt.show()
