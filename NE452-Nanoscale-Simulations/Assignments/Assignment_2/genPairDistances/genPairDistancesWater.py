import numpy as np
import mdtraj as md
from sys import argv

input_data = md.load(argv[1])
out_fname = argv[2]
out_fname2 = argv[3]

#print (dir(input_data))
N = int(input_data.n_atoms/3)
print('There are ', N, ' waters in the trajectory')
OO_pairs = []
OH_pairs = []
for i in range(N):
    for j in range(i+1,N):
        OO_pairs.append([i*3,j*3])
        
        OH_pairs.append([i*3,j*3+1])
        OH_pairs.append([i*3,j*3+2])
        OH_pairs.append([j*3,i*3+1])
        OH_pairs.append([j*3,i*3+2])

print('There are ', len(OO_pairs), ' O-O pairs')
print('There are ', len(OH_pairs), ' O-H pairs')

OO_distances = md.compute_distances(input_data,OO_pairs)
OH_distances = md.compute_distances(input_data,OH_pairs)

print('There are ', len(OO_distances), ' steps in the trajectory')
print('There are ', len(OH_distances), ' steps in the trajectory')

np.save(out_fname, OO_distances)
np.save(out_fname2, OH_distances)

