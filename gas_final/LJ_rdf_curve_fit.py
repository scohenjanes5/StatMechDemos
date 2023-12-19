from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import numpy as np
from work_rdf import LJ
import matplotlib.pyplot as plt

#read in data
g_r = np.loadtxt('LJ-rdf.csv', delimiter=',', skiprows=1)

r=np.linspace(1,len(g_r),len(g_r))*0.01

g_r = savgol_filter(g_r, 51, 3)


#remove the first 10 points
r=r[100:-100]
g_r=g_r[100:-100]

print(len(r))
print(len(g_r))

# define function to fit to data
def work(r, b):
    return np.exp(-b*LJ(r))

# fit curve to data
popt, pcov = curve_fit(work, r, g_r)

# print results
print('b =', popt[0], '+/-', pcov[0,0]**0.5)

# plot data and fit
plt.plot(r, g_r, 'b-', label='data')
plt.plot(r, work(r, *popt), 'r-', label='fit')
plt.xlabel('r')
plt.ylabel('g(r)')
plt.legend()
plt.show()

