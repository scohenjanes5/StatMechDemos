import rdfpy
import numpy as np
import matplotlib.pyplot as plt

#read xy coordinates from file
xy = np.loadtxt('coords.csv', delimiter=',', skiprows=1)

g_r, radii = rdfpy.rdf(xy, 0.01)

#plot the rdf
plt.plot(radii, g_r)
plt.show()
