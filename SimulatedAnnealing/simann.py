#Simulated Annealing of Argon Atoms.
#Start with 2 atoms, so we only deal with one coordinate.

import numpy as np
import matplotlib.pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Simulated Annealing of Argon Atoms.')
    parser.add_argument('-r', '--radius', type=float, metavar='', help='Radius of Argon atom.', default=1)
    parser.add_argument('-e', '--epsilon', type=float, metavar='', help='Lennard-Jones potential parameter.', default=1)
    parser.add_argument('-s', '--sigma', type=float, metavar='', help='Lennard-Jones potential parameter.', default=1)
    parser.add_argument('-t', '--temperature', type=float, metavar='', help='Initial temperature.', default=5)
    parser.add_argument('-n', '--nudge', type=float, metavar='', help='Nudge factor.', default=0.05)
    parser.add_argument('-c', '--cooling_rate', type=float, metavar='', help='Cooling rate.', default=0.99)
    parser.add_argument('-i', '--initial_bounds', help='Initial bounds of r.', nargs='+', type=float, metavar='', default=[1,10]) 
    return parser.parse_args()

#Function to calculate the Lennard-Jones potential:
def LJ(r):
    return 4*eps*((sig/r)**12 - (sig/r)**6)

#Function to choose new r
def newr(r,T):
    pururbation = np.random.uniform(-nudge,nudge)
    return r + pururbation


args = get_args()
#Constants:
#Radius of Argon atom:
rad = args.radius
#Lennard-Jones potential parameters:
eps = args.epsilon
sig = args.sigma
#Initial temperature:
T = args.temperature
nudge = args.nudge
cooling_rate = args.cooling_rate

#Initial r is random:
if len(args.initial_bounds) == 1:
    r = args.initial_bounds[0]
elif len(args.initial_bounds) == 2:
    r = np.random.uniform(args.initial_bounds[0],args.initial_bounds[1])
else:
    raise ValueError('Initial bounds must be a list of length 1 or 2.')

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
