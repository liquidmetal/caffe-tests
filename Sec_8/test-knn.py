#!/usr/bin/env python

import numpy as np
from sklearn.neighbors import NearestNeighbors

def find_closest_match_index(inp, features):
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(features)
    distances, indices = nbrs.kneighbors(np.array([inp]))
    return indices

features = [ [3,4], [5,6], [1,2] ]
idx = find_closest_match_index([1,2], features)
print(idx)
