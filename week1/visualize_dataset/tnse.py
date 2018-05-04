import os

import numpy as np
from tsne import bh_sne

vector_files = os.listdir('vectors/')

with open('filenames.txt', 'w') as f:
    for v in vector_files:
        f.write(v.replace('.npy', '') + '\n')

vectors = []

for fn in vector_files:
    vector = np.load('vectors/' + fn)
    vectors.append(vector)

np_vectors = np.array(vectors, dtype=float)

vectors_2d = bh_sne(np_vectors)

np.save('vectors_2d.npy', vectors_2d)
