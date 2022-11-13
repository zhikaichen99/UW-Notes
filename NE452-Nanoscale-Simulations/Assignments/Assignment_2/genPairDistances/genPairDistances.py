import numpy as np
import mdtraj as md
from sys import argv

input_data = md.load(argv[1])
out_fname = argv[2]

#print (dir(input_data))
N = input_data.n_atoms
print('There are ', N, ' particles in the trajectory')
pairs = []
for i in range(N):
    for j in range(i+1,N):
        pairs.append([i,j])

print('There are ', len(pairs), ' pairs of particles')

distances = md.compute_distances(input_data,pairs)

print('There are ', len(distances), ' steps in the trajectory')

np.save(out_fname, distances)

