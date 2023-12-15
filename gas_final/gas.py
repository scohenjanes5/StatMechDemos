import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
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

def set_initial_velocities(N, v0):
    v = torch.zeros((2,N)).to(device) #X,Y velocities in each row
    #set random directions for velocities with magnitude v0
    v[0] = v0 * torch.cos(2*np.pi*torch.rand(v.shape[1]))
    v[1] = v0 * torch.sin(2*np.pi*torch.rand(v.shape[1]))
    return v

def motion(r, v, ids_pairs, ts, dt, d_cutoff, box_size=1, box_type='periodic'):
    rs = torch.zeros((ts, r.shape[0], r.shape[1])).to(device) #Store positions at each time step
    vs = torch.zeros((ts, v.shape[0], v.shape[1])).to(device) #Store velocities at each time step
    # Initial State
    rs[0] = r
    vs[0] = v
    for i in progress.track(range(1,ts), description='Simulating Steps'):
        ic = ids_pairs[get_deltad2_pairs(r, ids_pairs) < d_cutoff**2] #indices of colliding particles
        v[:, ic[:,0]], v[:,ic[:,1]] = compute_new_v(v[:, ic[:,0]], v[:,ic[:,1]], r[:, ic[:,0]], r[:,ic[:,1]]) #update velocities

        if box_type == 'reflective':
            v[0,r[0]>box_size] = -torch.abs(v[0,r[0]>box_size]) #particles that hit the right wall bounce back
            v[0,r[0]<0] = torch.abs(v[0,r[0]<0]) #particles that hit the left wall bounce back
            v[1,r[1]>box_size] = -torch.abs(v[1,r[1]>box_size]) #particles that hit the top wall bounce back
            v[1,r[1]<0] = torch.abs(v[1,r[1]<0]) #particles that hit the bottom wall bounce back
        
        r = r + v*dt #update positions according to velocities
        
        if box_type == 'periodic':
            #impose periodic boundary conditions
            r[0,r[0]>box_size] = r[0,r[0]>box_size] - box_size #particles that hit the right wall reappear on the left
            r[0,r[0]<0] = r[0,r[0]<0] + box_size #particles that hit the left wall reappear on the right
            r[1,r[1]>box_size] = r[1,r[1]>box_size] - box_size #particles that hit the top wall reappear on the bottom
            r[1,r[1]<0] = r[1,r[1]<0] + box_size #particles that hit the bottom wall reappear on the top
        
        rs[i] = r #store positions
        vs[i] = v #store velocities
    return rs, vs

def compute_rdf(final_r, L, dr, ids_pairs):
    #compute radial distribution function
    rdf = torch.zeros(int(L/dr))
    print(final_r.shape)
    #quit()
    for i in progress.track(range(final_r.shape[1]), description='Computing RDF'):
        d = torch.sqrt(get_deltad2_pairs(final_r[:,i].unsqueeze(1), ids_pairs)).squeeze()
        rdf += torch.histc(d, bins=int(L/dr), min=0, max=L)

    plt.plot(rdf)
    return rdf
def getArgs():
    parser = argparse.ArgumentParser(description='Simulate a gas')
    parser.add_argument('-N', '--N', type=int, default=4000, help='Number of particles')
    parser.add_argument('--dt', type=float, default=8e-6, help='Time step')
    parser.add_argument('--t_steps', type=int, default=2000, help='Number of time steps')
    parser.add_argument('--v0', type=float, default=500, help='Initial velocity')
    parser.add_argument('-L', '--L', type=float, default=10, help='Box size')
    parser.add_argument('--radius', type=float, default=0.005, help='Collision radius')
    parser.add_argument('--box_type', type=str, default='periodic', help='Box type')
    parser.add_argument('--test', action='store_true', help='Use easier parameters for testing')
    return parser.parse_args()

def animate(rs):
    fig, ax = plt.subplots()
    artists = []
    for snapshot in progress.track(rs, description='Creating Animation'):
        # Append the updated plot as an artist for this frame
        artists.append([ax.scatter(*snapshot.cpu(), s=1)])

    # Create the animation
    animation = ArtistAnimation(fig, artists, interval=1, blit=True)

    # To display the animation
    plt.show()

args = getArgs()

if args.test:
    args.N = 101
    args.v0 = 100
    args.dt = 1e-6
    args.t_steps = 1000
    args.radius = 0.05

L = args.L
N = args.N

print("Setting up initial conditions...")
r = L * torch.rand((2,N), device=device) #X,Y coordinates in each row
#ixr = r[0]>0.5*L #particles that start on the right
#ixl = r[0]<=0.5*L #particles that start on the left
ids = torch.arange(N)
ids_pairs = torch.combinations(ids,2).to(device)
v = set_initial_velocities(N, args.v0)
print("Done")
rs, vs = motion(r, v, ids_pairs, ts=args.t_steps, dt=args.dt, d_cutoff=2*args.radius, box_size=L)

rdf = compute_rdf(rs[-1], L, dr=0.01, ids_pairs=ids_pairs)

#plt.plot(rdf)
#plt.show()

animate(rs)

#plt.scatter(*rs[-1].cpu())
#plt.xlim(0,L)
#plt.ylim(0,L)
#plt.show()

