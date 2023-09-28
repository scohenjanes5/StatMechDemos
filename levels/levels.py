import numpy as np
import math
import matplotlib.pyplot as plt
from rich.progress import track
"""
Finds the population of each level based on the temperature of the system.
"""

#def levels(n,T):
#    """
#    n: number of levels
#    T: temperature of system
#    """


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
    kb=1.38064852e-23
    try:
        S = kb*np.log(omega(down,up))
    except OverflowError:
        S = kb*ln_stirling_omega(down,up)
    return S

total=1000
S_array=[]
E_array=[]
for up in track(range(1,total), description="[blue] Calculating S(E) curve:"):
    S_array.append(S(total,up))
    E_array.append(up)


plt.plot(E_array,S_array)
plt.xlabel("E (Number of particles in the upper level)")
plt.ylabel("Entropy")
plt.title(f"Entropy of a system with {total} particles")
plt.show()



#real=np.log(omega(down,up))
#approx=ln_stirling_omega(down,up)

#S_real=kb*real
#S_approx=kb*approx




