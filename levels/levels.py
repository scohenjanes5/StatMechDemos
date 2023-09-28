#!/usr/bin/env python3
import numpy as np
import math
import matplotlib.pyplot as plt
from rich.progress import track
"""
Finds the population of each level based on the temperature of the system.
"""
global kb
kb=1.38064852e-23


def omega(n1,n2):
    """
    n1: population of the lower level
    n2: population of the upper level
    returns omega, which is the number of possible arrangements of a system with n1 particles in the lower level and n2 particles in the upper level.
    """
    omega = math.factorial(n1+n2)/(math.factorial(n1)*math.factorial(n2))
    return omega

def stirling(n):
    """
    n: number to take the factorial of
    Use the stirling approximation for the factorial to calculate ln(n!).
    """
    stirling = n*np.log(n)-n
    return stirling

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
        S = kb*np.log(omega(down,up))
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

Energy_levels=[0,1e-20]
total=1000
S_array=[]
E_array=[]
p1_array=[]
p2_array=[]

#plotting the curves for probabilities vs temperature
T_array=range(1,30000)
for T in track(T_array, description="[blue] Calculating probabilities vs temperature:"):
    probs=probabilities(T, Energy_levels)
    p1_array.append(probs[0])
    p2_array.append(probs[1])

plt.plot(T_array,p1_array,label="Lower level")
plt.plot(T_array,p2_array,label="Upper level")
plt.xlabel("Temperature")
plt.ylabel("Probability")
plt.title(f"Probability of finding a particle in each level")
plt.legend()
plt.show()



#E_array,S_array=S_curve(total,Energy_levels)
#plt.plot(E_array,S_array)
#plt.xlabel("E (Number of particles in the upper level)")
#plt.ylabel("Entropy")
#plt.title(f"Entropy of a system with {total} particles")
#plt.show()


