import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import argparse
from rich import progress
import torch, rdfpy

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
    thetas = 2*np.pi*torch.rand(N).to(device)
    v[0] = v0 * torch.cos(thetas)
    v[1] = v0 * torch.sin(thetas)
    # print(f"The average initial velocity is {torch.mean(torch.sqrt(v[0]**2 + v[1]**2)):.2f}")
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

def get_rdf(rs_arg, dr, L, cutoff=0.9):
    r_max = (L/2) * cutoff
    g_r_avg = np.zeros(int(r_max/dr))
    lengths = []
    # print(f'g_r_avg shape: {g_r_avg.shape}')

    for snapshot in progress.track(rs_arg, description='Computing RDF'):
        points = snapshot.T.cpu().numpy()
        # print(points.shape)
        g_r, radii = rdfpy.rdf(points, dr=dr, rcutoff=cutoff, parallel=True)
        lengths.append(len(g_r))
        # print(f'g_r shape: {g_r.shape}')
        #add the current g_r to the average. Extend the array if necessary
        if len(g_r) > len(g_r_avg):
            g_r_avg = np.pad(g_r_avg, (0,len(g_r)-len(g_r_avg)), 'constant', constant_values=(0))
        elif len(g_r) < len(g_r_avg):
            g_r = np.pad(g_r, (0,len(g_r_avg)-len(g_r)), 'constant', constant_values=(0))
        g_r_avg += g_r
    g_r_avg /= len(rs_arg)

    min_length = min(lengths)
    g_r_avg = g_r_avg[:min_length]

    #create a radii array that is the same length as g_r_avg
    radii = np.arange(len(g_r_avg)) * dr
    return g_r_avg, radii

def compute_rdf(rs_arg, L, dr):
    # print(rs_arg.shape)
    #rs=rs.T

    N = rs_arg.shape[2]
    bulk_density = N / L**2
    # print(f"bulk_density: {bulk_density}")

    max_dist = np.sqrt(2) * L

    # Initialize RDF array
    rdf_sum = torch.zeros(int(max_dist/dr), device=device)

    radii = torch.arange(len(rdf_sum), device=device) * dr

    for r in progress.track(radii, description='Computing RDF'):

        for points in rs_arg:
            # print(points.shape)

            points = points.T

            # Create a mask to exclude particles near the edges of the box for the current radius
            mask_edge = torch.all((points >= r + dr) & (points <= L - (r + dr)), dim=1)

            # Apply the mask to the points
            valid_points = points[mask_edge]

            # print(valid_points.shape)
            #number of valid points
            n_valid = valid_points.shape[0]
            # print(f"n_valid: {n_valid}")
            # quit()

            # Compute all pair distances for the valid points
            # Need to find distances between valid points and ALL points
            dists = torch.cdist(points, points)
            # print(dists.shape)
            dists = dists[mask_edge, :]
            # print(dists.shape)
            # dists = torch.cdist(valid_points, valid_points)

            # Remove lower triangle
            dists = torch.triu(dists)
        
            # Compute bin indices for each distance
            bins = (dists / dr).long()

            # Create a mask for distances between r and r+dr
            mask = (dists < r+dr) & (dists > r)

            masked_bins = bins[mask].flatten()

            # Increment RDF array only for distances less than L
            rdf = torch.zeros_like(rdf_sum)
            rdf.scatter_add_(0, masked_bins, torch.ones_like(masked_bins, dtype=rdf.dtype))
            
            ring_area = np.pi * ((r + dr)**2 - r**2)
            if n_valid > 0:
                rdf /= n_valid*bulk_density*ring_area
            else:
                rdf *= 0

            rdf_sum += rdf

    rdf = rdf_sum / len(rs_arg)

    # print(f'length of rdf: {len(rdf)}')

    #Max radius plotted is 0.9 * L / 2
    rdf = rdf[1:int(0.9 * L / 2 / dr)]
    radii = radii[1:len(rdf)+1]

    return rdf.cpu().numpy(), radii.cpu().numpy()

def getArgs():
    parser = argparse.ArgumentParser(description='Simulate a gas')
    parser.add_argument('-N', '--N', type=int, default=4000, help='Number of particles')
    parser.add_argument('--dt', type=float, default=8e-6, help='Time step')
    parser.add_argument('--t_steps', type=int, default=2000, help='Number of time steps')
    parser.add_argument('--v0', type=float, default=500, help='Initial velocity')
    parser.add_argument('-L', '--L', type=float, default=10, help='Box size')
    parser.add_argument('--radius', type=float, default=0.005, help='Collision radius')
    parser.add_argument('--box_type', type=str, default='periodic', help='Box type: periodic (p) or reflective (r)')
    parser.add_argument('--test', action='store_true', help='Use easier parameters for testing')
    return parser.parse_args()

def animate(rs_arg):
    fig, ax = plt.subplots()
    artists = []
    for snapshot in progress.track(rs_arg, description='Creating Animation'):
        # Append the updated plot as an artist for this frame
        artists.append([ax.scatter(*snapshot.cpu(), s=1)])

    # Create the animation
    animation = ArtistAnimation(fig, artists, interval=1, blit=True)

    # To display the animation
    plt.show()

def plot_rdf(rdf, radii):
    plt.plot(radii, rdf)
    plt.xlabel("r")
    plt.ylabel("g(r)")
    plt.show()

def main():

    args = getArgs()

    if args.box_type == 'p':
        args.box_type = 'periodic'
    elif args.box_type == 'r':
        args.box_type = 'reflective'

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
    ids = torch.arange(N)
    ids_pairs = torch.combinations(ids,2).to(device)
    v = set_initial_velocities(N, args.v0)
    print("Done")
    rs, vs = motion(r, v, ids_pairs, ts=args.t_steps, dt=args.dt, d_cutoff=2*args.radius, box_size=L)

    if not args.test:
        num_kept_steps = args.t_steps - 10 #int(args.t_steps/4)
    else:
        num_kept_steps = args.t_steps - 1

    kept_rs = rs[num_kept_steps:]
    #print(kept_rs.shape)

    #rdf, radii = compute_rdf(rs[num_kept_steps:], L, dr=0.01)
    rdf, radii = get_rdf(kept_rs, dr=0.01, L=L, cutoff=0.9)

    #write to file
    np.savetxt("rdf.csv", rdf, delimiter=",")
    #write coordinates to file
    np.savetxt("coords.csv", rs[-1].cpu().numpy().T, delimiter=",")

    #plot
    plot_rdf(rdf, radii)

    #animate(rs)

    #plt.scatter(*rs[-1].cpu())
    #plt.xlim(0,L)
    #plt.ylim(0,L)
    #plt.show()

if __name__ == "__main__":
    main()

