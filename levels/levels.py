#!/usr/bin/env python3
import numpy as np
import argparse
import math
import matplotlib.pyplot as plt
from rich.progress import track
from decimal import Decimal
"""
Finds the population of each level based on the temperature of the system.
"""
global kb
kb=Decimal("1.38064852e-23")

class MonteCarlo:
    """
    total: total number of particles in the system
    Energy_levels: list of the energy levels of the system (in Jules)
    Temperature: temperature of the system
    starting_level: level to start the system in (integer indicies)
    probs: list of the probabilities of finding a particle in each level
    """
    def __init__(self, total, Energy_levels, Temperature, starting_level=0):
        self.total=total
        self.Energy_levels=Energy_levels
        self.Temperature=Temperature
        self.probs=probabilities(self.Temperature, self.Energy_levels)
        self.starting_level=starting_level
        self.entropy_array, self.energy_array, self.starting_array=self.populate_levels()
        
    def populate_levels(self):
        """
        Populates the levels of the system based on the probabilities of each level.
        """
        starting_array=[self.starting_level]*self.total
        #sum used to get count in level 1. If extending to more levels, will need to use other method.
        num_up=sum(starting_array)
        entropy_array=[S(self.total,num_up)]
        energy_array=[num_up*self.Energy_levels[1]+(self.total-num_up)*self.Energy_levels[0]]
        for i in track(range(self.total), description="[blue] Determining populations of energy levels:"):
            if np.random.rand()<self.probs[0]:
                starting_array[i]=0
            else:
                starting_array[i]=1

            if starting_array[i] == self.starting_level:
                continue

            num_up=sum(starting_array)
            new_entropy=S(self.total,num_up)
            new_energy=num_up*self.Energy_levels[1]+(self.total-num_up)*self.Energy_levels[0]
            entropy_array.append(new_entropy)
            energy_array.append(new_energy)
        return entropy_array, energy_array, starting_array


def omega(n1,n2):
    """
    n1: population of the lower level
    n2: population of the upper level
    returns omega, which is the number of possible arrangements of a system with n1 particles in the lower level and n2 particles in the upper level.
    """
    omega = math.factorial(n1+n2)/(math.factorial(n1)*math.factorial(n2))
    return Decimal(str(omega))

def stirling(n):
    """
    n: number to take the factorial of, then ln
    Use the stirling approximation for the factorial to calculate ln(n!).
    """
    stirling = n*np.log(n)-n
    return Decimal(str(stirling))

def ln_stirling_omega(n1,n2):
    return stirling(n1+n2)-stirling(n1)-stirling(n2)
    
def stirling_omega(n1,n2):
    """
    n1: population of the lower level
    n2: population of the upper level
    Use the sterling approximation for the factorial to calculate omega.
    """
    omega = np.exp(ln_stirling_omega) #omega = math.factorial(n1+n2)/(math.factorial(n1)*math.factorial(n2))
    return omega
def S(total,up):
    """
    total: total number of particles in the system
    up: number of particles in the upper level
    returns the entropy of the system.
    """
    down=total-up
    try:
        S = kb*omega(down,up).ln()
    except OverflowError:
        S = kb*ln_stirling_omega(down,up)
    return S

def S_curve(total,Energy_levels):
    """
    total: total number of particles in the system
    Energy_levels: list of the energy levels of the system
    returns the entropy of the system as a function of the number of particles in the upper level.
    """
    S_array=[]
    E_array=[]
    for up in track(range(1,total), description="[blue] Calculating S(E) curve:"):
        S_array.append(S(total,up))
        E_array.append(up*Energy_levels[1]+(total-up)*Energy_levels[0])
    return E_array,S_array

def Z(T, Energy_levels):
    """
    T: temperature of the system
    returns the partition function of the system.
    Energy_levels: list of the energy levels of the system
    Z = sum(exp(-E_j/(kb*T)))
    """
    Z=0
    for E in Energy_levels:
        Z+=np.exp(-E/(kb*T))
    return Z

def probabilities(T, Energy_levels):
    """
    T: temperature of the system
    Energy_levels: list of the energy levels of the system
    returns the probability of finding a particle in each level.
    """
    probs=[]
    for E in Energy_levels:
        probs.append(np.exp(-E/(kb*T)) / Z(T, Energy_levels))
    return probs

def P_curve(T_max, Energy_levels, allow_negative=False):
    """
    T_max: maximum temperature of the system
    Energy_levels: list of the energy levels of the system
    allow_negative: if True, allows negative temperatures
    returns the probability of finding a particle in each level as a function of temperature.
    """
    limit=int(T_max)
    if allow_negative:
        T_array=range(-limit,limit)
    else:
        T_array=range(1,limit)
    p1_array=[]
    p2_array=[]
    for T in track(T_array, description="[blue] Calculating probabilities vs temperature:"):
        try:
            probs=probabilities(T, Energy_levels)
            p1_array.append(probs[0])
            p2_array.append(probs[1])
        except ZeroDivisionError:
            p1_array.append(0)
            p2_array.append(0)
    return p1_array, p2_array, T_array

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default = 5000, type=int, help = "number of particles in the system")
    parser.add_argument('-t', '--temp', default = 25, help = "temperature of the system")
    parser.add_argument('-e', '--energy', nargs = 2, default = [0,6], help = "energy levels of the system in multiples of kb")
    parser.add_argument('-l', '--starting_level', default = 0, help = "starting level of the system. Either 0 or 1.")
    parser.add_argument('-m', '--monte', default=False, action="store_true", help = "run the monte carlo simulation")
    parser.add_argument('-p', '--prob', default=False, action="store_true", help = "plot the probability curves")
    parser.add_argument('-q', '--quiet', default=False, action="store_true", help = "don't plot the S(E) curve for the monte carlo simulation.")
    parser.add_argument('-s', '--se_curve', default=False, action="store_true", help = "plot the theoretical S(E) curve.")
    args = parser.parse_args()
    return args


args = getArgs()
if not args.monte and not args.prob and not args.se_curve:
    print("Please specify a job to run. Use the -h flag for a list of options.")
    quit()
total = args.n
Energy_levels = [kb*e for e in args.energy]
Max_T = Decimal(str(args.temp))
starting_level = args.starting_level

#Energy_levels=[0*kb,6*kb]
#total=10000
#Max_T=100
S_array=[]
E_array=[]

if args.monte:
    system=MonteCarlo(total,Energy_levels,Max_T,starting_level=starting_level)
    print(f"The number of particles in the upper level is {sum(system.starting_array)} out of {total} particles.")
    print(f"This ratio is {sum(system.starting_array)/total:.4f}.")
    energy_diff=system.energy_array[-1]-system.energy_array[-2]
    entropy_diff=system.entropy_array[-1]-system.entropy_array[-2]
    final_temp = energy_diff/entropy_diff

    print(f"The numerically determined temperature is {final_temp:.2f} K, compared to the expected temperature of {Max_T} K.")

    if not args.quiet:
        #plotting the curves for entropy vs energy
        plt.plot(system.energy_array,system.entropy_array)
        plt.xlabel("Energy (J)")
        plt.ylabel("Entropy (J/K)")
        plt.title(f"System with {total} particles")
        plt.show()


if args.prob:
    #plotting the curves for probabilities vs temperature
    p1_array, p2_array, T_array = P_curve(Max_T, Energy_levels, allow_negative=False)

    plt.plot(T_array,p1_array,label="Lower level")
    plt.plot(T_array,p2_array,label="Upper level")
    plt.xlabel("Temperature")
    plt.ylabel("Probability")
    plt.title(f"Probability of finding a particle in each level")
    plt.legend()
    plt.show()

if args.se_curve:
    #plotting the curves for entropy vs energy
    E_array,S_array=S_curve(total,Energy_levels)
    slope_array=[]
    temp_array=[]
    for i in range(len(E_array)-1):
        slope=(S_array[i+1]-S_array[i])/(E_array[i+1]-E_array[i])
        slope_array.append(slope)
        temp_array.append(1/slope)

    #plot all three curves as subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle(f"System with {total} particles")
    ax2.plot(E_array[1:],slope_array)
    ax2.set(ylabel="dS/dE (1/K)")
    ax3.plot(E_array[1:],temp_array)
    ax3.set(ylabel="T (K)")
    ax1.plot(E_array,S_array)
    ax1.set(xlabel="E (J)", ylabel="S (J/K)")
    plt.show()

