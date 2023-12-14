import numpy as np
import matplotlib.pyplot as plt
import argparse
from rich import progress
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_deltad2_pairs(r, ids_pairs):
    dx = torch.diff(torch.stack([r[0][ids_pairs[:,0]], r[0][ids_pairs[:,1]]]).T).squeeze()
    dy = torch.diff(torch.stack([r[1][ids_pairs[:,0]], r[1][ids_pairs[:,1]]]).T).squeeze()
    return dx**2 + dy**2

def compute_new_v(v1, v2, r1, r2):
    v1new = v1 - torch.sum((v1-v2)*(r1-r2), axis=0)/torch.sum((r1-r2)**2, axis=0) * (r1-r2)
    v2new = v2 - torch.sum((v1-v2)*(r1-r2), axis=0)/torch.sum((r2-r1)**2, axis=0) * (r2-r1)
    return v1new, v2new

def motion(r, v, ids_pairs, ts, dt, d_cutoff):
    rs = torch.zeros((ts, r.shape[0], r.shape[1])).to(device) #Store positions at each time step
    vs = torch.zeros((ts, v.shape[0], v.shape[1])).to(device) #Store velocities at each time step
    # Initial State
    rs[0] = r
    vs[0] = v
    for i in progress.track(range(1,ts), description='Simulating Steps'):
        ic = ids_pairs[get_deltad2_pairs(r, ids_pairs) < d_cutoff**2] #indices of colliding particles
        v[:,ic[:,0]], v[:,ic[:,1]] = compute_new_v(v[:,ic[:,0]], v[:,ic[:,1]], r[:,ic[:,0]], r[:,ic[:,1]]) #update velocities
        
        v[0,r[0]>1] = -torch.abs(v[0,r[0]>1]) #particles that hit the right wall bounce back
        v[0,r[0]<0] = torch.abs(v[0,r[0]<0]) #particles that hit the left wall bounce back
        v[1,r[1]>1] = -torch.abs(v[1,r[1]>1]) #particles that hit the top wall bounce back
        v[1,r[1]<0] = torch.abs(v[1,r[1]<0]) #particles that hit the bottom wall bounce back
        
        r = r + v*dt #update positions according to velocities
        rs[i] = r #store positions
        vs[i] = r #store velocities
    return rs, vs

N = 3000
dt = 8e-6
t_steps = 2000
v0 = 500
L = 1 #box size
r = L * torch.rand((2,N), device=device) #X,Y coordinates in each row
ixr = r[0]>0.5 #particles that start on the right
ixl = r[0]<=0.5 #particles that start on the left
ids = torch.arange(N)
ids_pairs = torch.combinations(ids,2).to(device)
v = torch.zeros((2,N)).to(device) #X,Y velocities in each row
v[0][ixr] = -v0 #particles on the right move left
v[0][ixl] = v0 #particles on the left move right
radius = 0.005
rs, vs = motion(r, v, ids_pairs, ts=t_steps, dt=dt, d_cutoff=2*radius)

plt.scatter(*rs[473].cpu())
plt.xlim(0,L)
plt.ylim(0,L)
plt.show()

