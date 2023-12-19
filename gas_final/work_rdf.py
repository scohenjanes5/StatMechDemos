import numpy as np
import matplotlib.pyplot as plt

# we can calculate g(r) by the reversable work method
# g(r) = exp(-beta W(r))

def LJ(r, sigma=1, epsilon=1):
    return 4*epsilon*((sigma/r)**12 - (sigma/r)**6)

def g_r(r, sigma=1, epsilon=1, beta=1):
    return np.exp(-beta*LJ(r, sigma, epsilon))


def main():
    #v = 500
    #T = v**2 
    #beta = 1 / T

    beta = 0.0029702970297029703

    #plot g(r) at T
    r = np.linspace(1.0, 4, 500)
    plt.plot(r, g_r(r, beta=beta, sigma=1, epsilon=1))
    plt.xlabel('r')
    plt.ylabel('g(r)')
    plt.show()

if __name__ == '__main__':
    main()
