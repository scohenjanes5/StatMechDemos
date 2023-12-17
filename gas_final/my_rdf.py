import numpy as np
from gas import compute_rdf
import torch, rdfpy
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the coords
coords = np.loadtxt('coords.csv', delimiter=',')
xy = coords


#boilerplate to format the coords coorectly for each function.
coords = torch.tensor(coords.T)
# print(coords.shape) ## (2, 4000)
#add another dimension so shape is (1,2,4000)
coords = coords.unsqueeze(0).to(device)
# print(coords.shape)

#calculate rdf with both methods
g_r, radii =  compute_rdf(coords, 10, dr=0.01, box_type = "periodic")
# print(g_r.shape)
# print(radii.shape)
# print(radii)
# print(g_r)

real_g_r, real_radii = rdfpy.rdf(xy, 0.01, parallel=False)

#plot on different subplots
fig, axs = plt.subplots(2, 1, constrained_layout=True)
axs[0].plot(radii, g_r)
axs[0].set_title('g(r) from gas.py')
axs[1].plot(real_radii, real_g_r)
axs[1].set_title('g(r) from rdfpy')
plt.show()

