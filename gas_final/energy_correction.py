import matplotlib.pyplot as plt
import numpy as np
from work_rdf import LJ

def read_data(filename):
    #read in data
    g_r = np.loadtxt(filename, delimiter=',', skiprows=1)
    r=np.linspace(1,len(g_r),len(g_r))*0.01
    u_r=LJ(r)

    integrand=g_r*u_r

    number_of_points_to_remove=90
    integrand=integrand[number_of_points_to_remove:]
    r=r[number_of_points_to_remove:]

    return r, integrand


def calculate_correction(r,g_r,rho):
    integrand=g_r*LJ(r)
    integral=0
    for i in range(1,len(integrand)):
        integral+=integrand[i]*(r[i]-r[i-1])
    return (rho/2)*integral

def plot_integrand(r,integrand):
    plt.plot(r,integrand)
    plt.xlabel('r')
    plt.ylabel('g(r)*u(r)')
    plt.show()

def main():
    r, integrand=read_data('many_points_rdf.csv')
    N=4000
    V=10**2
    particle_radius=0.05
    particle_area=np.pi*particle_radius**2
    rho=particle_area*N/V
    correction=calculate_correction(r,integrand,rho)
    print(correction)
    print(correction+505)
    plot_integrand(r,integrand)

if __name__ == '__main__':
    main()
