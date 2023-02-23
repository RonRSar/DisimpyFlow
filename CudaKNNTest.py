# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 09:50:17 2023

@author: ronit
"""
import os
from numba import cuda
import math
import numpy as np
import meshio

from disimpy import gradients, simulations, substrates, utils 


@cuda.jit
def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

@cuda.jit
def knn_cuda(data, query, result,bs,gs):
    n_data = data.shape[0]
    n_query = query.shape[0]
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if tid < n_query:
        distances = np.zeros(n_data)
        for i in range(n_data):
            distances[i] = distance[bs,gs](query[tid], data[i])

        indices = np.argpartition(distances, 1)[:1]
        result[tid] = indices[np.argsort(distances[indices])]

# Example usage:
mesh_path  = os.path.join(
    os.path.dirname(simulations.__file__), "tests","vascular_mesh_22-10-04_21-52-57_r4.ply")
mesh = meshio.read(mesh_path)

vertices = mesh.points.astype(np.float32) * 10e-6
faces = mesh.cells[0].data

mesh_path  = os.path.join(
        os.path.dirname(simulations.__file__), "tests\meshes","for_ronit_vloc_vdir_mesh(1).mat")
vdir, vloc, vdir_long, vloc_long, seg_length = utils.veloc_quiver(mesh_path)

substrate = substrates.mesh(vertices, faces, periodic=True, init_pos="intra", n_sv=np.array([200, 200, 200]))
positions = simulations._fill_mesh(n_points=1000, substrate = substrate, intra=True,seed=123, cuda_bs=512)


# Classify the test point 
vloc = vloc * 10e-6

d_positions = cuda.to_device(positions)
d_vloc = cuda.to_device(vloc)

bs = 128
gs = int(math.ceil(float(1000) / bs))

result = cuda.device_array((positions.shape), dtype=np.int32)

knn_cuda[bs,gs](d_positions, d_vloc, result, bs,gs)

result = result.copy_to_host()

print(result)