import numpy as np
import scipy
from typing import List
from scipy.linalg import orthogonal_procrustes
import itertools

def compute_cross_correlation(input_matrices):
    cc = {}
    for emb1,emb2 in itertools.permutations(range(len(input_matrices)),2):
        original_emb_1 = input_matrices[emb1]
        original_emb_2 = input_matrices[emb2]
        cc[(emb1, emb2)] = np.matmul(original_emb_2.T,original_emb_1)
    return cc
def ssea(input_matrices, input_dim):
    cc = compute_cross_correlation(input_matrices)
    T = {i: input_matrices[i] for i in range(len(input_matrices))}
    T[0] = np.eye(input_dim)
    for i in range(1, len(input_matrices)):
        M = 0
        for j in range(i):
            M += np.matmul(T[i],cc[i,j])
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        T[i]= U.dot(V_t)
    for ii in range(10):
        for i in range(len(input_matrices)):
            M = 0
            for j in range(len(input_matrices)):
                if i==j:continue
                M += np.matmul(T[i], cc[i, j])
            U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
            T[i] = U.dot(V_t)
    return T
def get_projection_to_intersection_of_nullspaces_SSEA(input_dim: int,input_matrices: List[np.ndarray]):
    T = ssea(input_matrices,input_dim)
    return T