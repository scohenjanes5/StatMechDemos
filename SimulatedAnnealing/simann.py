#Simulated Annealing of Argon Atoms.
#Start with 2 atoms, so we only deal with one coordinate.

import numpy as np
import matplotlib.pyplot as plt
import argparse

class atom:
    def __init__(self, x, y, z, rad):
        self.coords = np.array([x,y,z])
        self.rad = rad

class config:
    def __init__(self, n, bounds, rad):
        self.n = n
        self.bounds = bounds
        self.atoms = self.random_start(rad)
        self.E = self.config_E() 

    def random_start(self, rad):
        atoms = [atom(0,0,0,rad)]
        for i in range(1,self.n):
            #pick random r, theta, phi:
            r = np.random.uniform(max(self.bounds[0],rad), self.bounds[1])

            if self.n > 2:
                theta = np.random.uniform(0,2*np.pi)
            else:
                theta = 0 #dimer aligned along x-axis

            if self.n > 3:
                phi = np.random.uniform(0,np.pi)
            else:
                phi = np.pi/2 #dimer and trimer in x-y plane

            #convert to cartesian coordinates:
            x = r*np.sin(phi)*np.cos(theta)
            y = r*np.sin(phi)*np.sin(theta)
            z = r*np.cos(phi)
            atoms.append(atom(x,y,z,rad))
        return atoms

    def LJ(self, r):
        return 4*eps*((sig/r)**12 - (sig/r)**6)

    def config_E(self):
        E = 0
        #Loop over all pairs of atoms:
        for i in range(self.n):
            for j in range(i+1,self.n):
                #Calculate distance between atoms:
                r = np.linalg.norm(self.atoms[i].coords - self.atoms[j].coords)
                #Add to energy:
                E += self.LJ(r)
        return E

    def field(self, point):
        #Calculate field at point:
        E = 0
        for atom in self.atoms:
            #Calculate distance between atom and point:
            r = np.linalg.norm(atom.coords - point)
            #Add to field:
            E += self.LJ(r)
        return E

    def plot_PE_surface(self):
        #Create grid of points:
        x = np.linspace(-self.bounds[1], self.bounds[1],100)
        y = np.linspace(-self.bounds[1], self.bounds[1],100)
        X, Y = np.meshgrid(x, y)
        #Calculate potential energy at each point:
        Z = np.zeros((100,100))
        for i in range(100):
            for j in range(100):
                Z[i,j] = self.field(np.array([X[i,j],Y[i,j],0]))
        #Plot:
        plt.scatter(X,Y,c=np.log(abs(Z)),cmap='viridis')
        plt.colorbar()
        plt.show()
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #ax.plot_surface(X, Y, Z)
        #plt.show()

def get_args():
    parser = argparse.ArgumentParser(description='Simulated Annealing of Argon Atoms.')
    parser.add_argument('-r', '--radius', type=float, metavar='', help='Radius of Argon atom.', default=1)
    parser.add_argument('-e', '--epsilon', type=float, metavar='', help='Lennard-Jones potential parameter.', default=1)
    parser.add_argument('-s', '--sigma', type=float, metavar='', help='Lennard-Jones potential parameter.', default=1)
    parser.add_argument('-t', '--temperature', type=float, metavar='', help='Initial temperature.', default=5)
    parser.add_argument('-n', '--nudge', type=float, metavar='', help='Nudge factor.', default=0.05)
    parser.add_argument('-c', '--cooling_rate', type=float, metavar='', help='Cooling rate.', default=0.99)
    parser.add_argument('-i', '--initial_bounds', help='Initial bounds of r.', nargs='+', type=float, metavar='', default=[1,10])
    parser.add_argument('-N', '--number_of_atoms', type=int, metavar='', help='Number of atoms.', default=2)
    return parser.parse_args()


#Function to choose new r
def newr(r,T):
    pururbation = np.random.uniform(-nudge,nudge)
    return r + pururbation

args = get_args()
#Constants:
rad = args.radius
T = args.temperature
nudge = args.nudge
cooling_rate = args.cooling_rate
N = args.number_of_atoms
#Lennard-Jones potential parameters:
eps = args.epsilon
sig = args.sigma

#Initial r is random:
if len(args.initial_bounds) == 1:
    args.initial_bounds.append(args.initial_bounds[0])
elif len(args.initial_bounds) > 2:
    raise ValueError('Initial bounds must be a list of length 1 or 2.')

config = config(N, args.initial_bounds, rad)
config.plot_PE_surface()
quit()





Energies = []
Radii = []
failures = 0

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
plt.show()
