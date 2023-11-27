#Simulated Annealing of Argon Atoms.
#Start with 2 atoms, so we only deal with one coordinate.

import numpy as np
import matplotlib.pyplot as plt
import argparse

class atom:
    def __init__(self, coords, rad, cartesian=True):
        if cartesian == True:
            self.coords = np.array(coords)
        else:
            self.coords = self.polar_to_cartesian(coords)
        self.rad = rad

    def polar_to_cartesian(self, coords):
        #Converts polar coordinates to cartesian coordinates.
        r = coords[0]
        theta = coords[1]
        phi = coords[2]

        x = r*np.sin(phi)*np.cos(theta)
        y = r*np.sin(phi)*np.sin(theta)
        z = r*np.cos(phi)
        return np.array([x,y,z])

    def nudge(self, nudge, angles):
        #Nudges the atom by a random amount.
        theta, phi = angles
        purt_vector = self.polar_to_cartesian([nudge, theta, phi])
        self.coords = purt_vector + self.coords

class config:
    def __init__(self, n, bounds, rad, T):
        self.n = n
        self.bounds = bounds
        self.atoms = self.random_start(rad)
        self.E = self.config_E()
        self.T = T
        self.Energy_array = []

    def random_angles(self):
        if self.n > 2:
            theta = np.random.uniform(0,2*np.pi)
        else:
            theta = 0 #dimer aligned along x-axis

        if self.n > 3:
            phi = np.random.uniform(0,np.pi)
        else:
            phi = np.pi/2 #dimer and trimer in x-y plane
        return theta, phi

    def random_start(self, rad):
        atoms = [atom([0,0,0], rad)]
        for i in range(1,self.n):
            r = np.random.uniform(max(self.bounds[0],rad), self.bounds[1])
            theta, phi = self.random_angles()
            coords = [r, theta, phi]
            atoms.append(atom(coords, rad, cartesian=False))
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

    def anneal(self):
        failures = 0
        Energy = []
        while failures < self.n*250:
            failures = self.step(failures)
            Energy.append(self.E)
        self.Energy_array = Energy
        self.center_on_origin()

    def step(self, failures):
        for atom in self.atoms:
            #Calculate energy of current configuration:
            E = self.config_E()
            current_coords = atom.coords
            #Nudge atom:
            angles = self.random_angles()
            atom.nudge(nudge, angles)
            #Calculate energy of new configuration:
            Enew = self.config_E()
            #Calculate change in energy:
            dE = Enew - E
            #If dE < 0, accept new r:
            if dE < 0:
                failures = 0
            #If dE > 0, accept new r with probability exp(-dE/T):
            elif np.random.uniform(0,1) < np.exp(-dE/self.T):
                failures = 0
            else:
                #If new r is rejected, revert to old r:
                atom.coords = current_coords
                failures += 1
        #Decrease temperature:
        self.T *= cooling_rate
        return failures

    def center_on_origin(self):
        #Center configuration on origin:
        center = np.zeros(3)
        for atom in self.atoms:
            center += atom.coords
        center /= self.n
        for atom in self.atoms:
            atom.coords -= center

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
        plt.scatter(X,Y,c=np.log(abs(Z)),cmap='viridis')
        plt.colorbar()
        plt.show()

    def plot_3D(self):
        #Plot 3D configuration:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for atom in self.atoms:
            ax.scatter(atom.coords[0], atom.coords[1], atom.coords[2], c='b', s=400)
        plt.show()

    def create_xyz_file(self):
        #Create xyz file of current configuration:
        with open('config.xyz', 'w') as f:
            f.write(str(self.n) + '\n')
            f.write('Atoms. T = ' + str(self.T) + '\n')
            for atom in self.atoms:
                f.write('Ar ' + str(atom.coords[0]) + ' ' + str(atom.coords[1]) + ' ' + str(atom.coords[2]) + '\n')

    
def get_args():
    parser = argparse.ArgumentParser(description='Simulated Annealing of Argon Atoms.')
    parser.add_argument('-r', '--radius', type=float, metavar='', help='Radius of Argon atom.', default=1)
    parser.add_argument('-e', '--epsilon', type=float, metavar='', help='Lennard-Jones potential parameter.', default=1)
    parser.add_argument('-s', '--sigma', type=float, metavar='', help='Lennard-Jones potential parameter.', default=1)
    parser.add_argument('-t', '--temperature', type=float, metavar='', help='Initial temperature.', default=5)
    parser.add_argument('-n', '--nudge', type=float, metavar='', help='Nudge factor.', default=0.05)
    parser.add_argument('-c', '--cooling_rate', type=float, metavar='', help='Cooling rate.', default=0.99)
    parser.add_argument('-i', '--initial_bounds', help='Initial bounds of r.', nargs='+', type=float, metavar='', default=[1,3])
    parser.add_argument('-N', '--number_of_atoms', type=int, metavar='', help='Number of atoms.', default=2)
    return parser.parse_args()

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

config = config(N, args.initial_bounds, rad, T)
config.anneal()
#config.plot_PE_surface()
config.plot_3D()
config.create_xyz_file()

