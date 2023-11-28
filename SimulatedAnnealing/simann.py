#Simulated Annealing of Argon Atoms.
#Start with 2 atoms, so we only deal with one coordinate.

import numpy as np
import matplotlib.pyplot as plt
import argparse
from rich import progress

class atom:
    def __init__(self, coords, rad, cartesian=True):
        if cartesian == True:
            self.coords = np.array(coords)
        else:
            self.coords = self.polar_to_cartesian(coords)
        self.rad = rad
        self.coords_array = []

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

    def save_coords(self):
        self.coords_array.append(self.coords)

class config:
    def __init__(self, n, bounds, rad, T, filename='config.xyz', periodic=False, num_cycles=10):
        self.n = n
        self.bounds = bounds
        self.atoms = self.random_start(rad)
        self.E = self.config_E()
        self.T = T
        self.Energy_minima_array = []
        self.Energy_array = []
        self.failures = 0
        self.filename = filename
        self.periodic = periodic
        self.cycles = num_cycles
        self.cooling = True
        self.T_array = []
        self.dE_array = []

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
        initial_T = self.T
        for _ in progress.track(range(self.cycles), description='Annealing...'):
            while self.failures < self.n*200:
                oldE = self.E
                self.step()
                self.dE_array.append(self.E - oldE)
                self.Energy_array.append(self.E)
                self.T_array.append(self.T)
            self.center_on_origin()
            self.Energy_minima_array.append(self.E)
            for atom in self.atoms:
                atom.save_coords()
            self.cooling = False
            #Start warming:
            while self.T < initial_T:
                oldE = self.E
                self.step()
                self.dE_array.append(self.E - oldE)
                self.Energy_array.append(self.E)
                self.T_array.append(self.T)
            self.cooling = True
        print('Annealing complete. Finding minimum energy configuration...')
        #the index of the minimum energy configuration:
        min_index = np.argmin(self.Energy_minima_array)
        print(f'Minimum energy: {self.Energy_minima_array[min_index]} is at index {min_index}')
        print(f"All energies: {self.Energy_minima_array}")
        #the minimum energy configuration:
        min_config = self.atoms
        for atom in min_config:
            atom.coords = atom.coords_array[min_index]
        self.atoms = min_config

    def simulate_fluid(self):
        for _ in progress.track(range(10000), description='Simulating fluid...'):
            self.step()
            self.Energy_array.append(self.E)

    def step(self):
        for atom in self.atoms:
            old_f = self.failures
            #Calculate energy of current configuration:
            E = self.config_E()
            current_coords = atom.coords
            #Nudge atom:
            angles = self.random_angles()
            atom.nudge(nudge, angles)
            #Check if new configuration is within bounds:
            if self.periodic:
                for i in range(3):
                    if atom.coords[i] > self.bounds[1]:
                        atom.coords[i] -= 2*self.bounds[1]
                    elif atom.coords[i] < self.bounds[0]:
                        atom.coords[i] += 2*self.bounds[1]
            #Calculate energy of new configuration:
            Enew = self.config_E()
            #Calculate change in energy:
            dE = Enew - E
            #If dE < 0, accept new r:
            if dE < 0:
                self.failures = 0
            #If dE > 0, accept new r with probability exp(-dE/T):
            elif np.random.uniform(0,1) < np.exp(-dE/self.T):
                self.failures = 0
            else:
                #If new r is rejected, revert to old r:
                atom.coords = current_coords
                self.failures += 1
        #Decrease temperature if not periodic:
        if self.cooling:
            if not self.periodic or self.failures > old_f:
                self.T *= cooling_rate
        else:
            self.T /= cooling_rate
        #Recalculate energy:
        self.E = self.config_E()

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
        with open(self.filename, 'w') as f:
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
    parser.add_argument('-f', '--filename', type=str, metavar='', help='Name of xyz file.', default='config.xyz')
    parser.add_argument('-p', '--periodic', action='store_true', help='Periodic boundary conditions.')
    return parser.parse_args()

args = get_args()
#Constants:
nudge = args.nudge
cooling_rate = args.cooling_rate
#Lennard-Jones potential parameters:
eps = args.epsilon
sig = args.sigma

if len(args.initial_bounds) == 1:
    args.initial_bounds.append(args.initial_bounds[0])
elif len(args.initial_bounds) > 2:
    raise ValueError('Initial bounds must be a list of length 1 or 2.')

config = config(args.number_of_atoms, args.initial_bounds, args.radius, args.temperature, args.filename, args.periodic)
if args.periodic:
    config.simulate_fluid()
else:
    config.anneal()
#config.plot_PE_surface()
config.plot_3D()
config.create_xyz_file()

#plot energy vs. cycle number and temperature vs. cycle number and dE vs. cycle number
fig, ax = plt.subplots(3,1)
ax[0].plot(config.Energy_array)
ax[0].set_xlabel('Cycle number')
ax[0].set_ylabel('Energy')
ax[1].plot(config.T_array)
ax[1].set_xlabel('Cycle number')
ax[1].set_ylabel('Temperature')
ax[2].plot(config.dE_array)
ax[2].set_xlabel('Cycle number')
ax[2].set_ylabel('dE')
plt.show()

#To appreciate the power of the simulated annealing method, find the minimum energy geometry of clusers with 3, 4 and 13 argon atoms and report the values of the minimum energy. For the cluster with 13 atoms run the program with three different initial temperatures, 10 K, 20 K and 30 K. Compare the final results. Do the final energy and geometry depend on the initial temperature? Why, or why not?

#The final structures and energies are the same, since all of these temperatures are sufficiently high to explore the entire configuration space before the temperature approaches zero.

#How would you compute a thermodynamic average at a fixed temperature T using the program for simulating annealing?

#I would make repeated measurements after an initial equilibration period, then average the results.

#Compute the radial distribution function g(r) for the cluster and compare it to the g(r) of a fluid of argon atoms at constant T,N,V by using periodic boundary conditions.
#The rdf is a spike for the cluster, and close to uniform for the fluid.

