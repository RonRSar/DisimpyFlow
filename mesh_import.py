# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 10:11:55 2022

@author: ronit
"""

#Initialisation
import os
import pickle
import scipy.io as scp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import KDTree

from disimpy import gradients, simulations, substrates, utils
import meshio


#load mesh
mesh_path  = os.path.join(
    os.path.dirname(simulations.__file__), "tests","vascular_mesh_22-10-04_21-52-57_r4.ply")
mesh = meshio.read(mesh_path)
vertices = mesh.points.astype(np.float32) * 10e-6
faces = mesh.cells[0].data
mesh_path  = os.path.join(
    os.path.dirname(simulations.__file__), "tests\meshes","for_ronit_vloc_vdir_mesh(1).mat")
vdir, vloc, vdir_long, vloc_long, seg_length = utils.veloc_quiver(mesh_path)

#twst
n_walkers = 100
vloc = vloc * 10e-6
#1. initialise walkers
#scale = np.abs(np.max(vloc)) + np.abs(np.min(vloc))
#positions = np.random.rand(n_walkers,3)*scale + np.min(vloc)
substrate = substrates.mesh(vertices, faces, periodic=True, init_pos="intra", n_sv=np.array([100, 100, 100]))
positions = simulations._fill_mesh(n_walkers, substrate, intra=True,seed=123)
dist = 0
step = 1e-3*80e-3/(100-1) #v*dt
max_iter = 100
orig_pos = positions

#2. nearest neighbour search of walker with vloc
tree = KDTree(vloc)
#d, index = tree.query(positions,k=1) 
#vector_to_spin = vloc[index] - positions


itera = 0
while itera < max_iter:
    itera += 1
    d, index = tree.query(positions,k=1)

    #3. This gives nearest vector to each spin
    #vector_to_spin = vloc[index] - positions
 
    #4. step in direction of NN search, step of distance
    positions = positions + step*vdir[index]
 
dist = positions - orig_pos

#checking?
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# for i in range(0,np.size(dist,0)):
#     ax.plot(orig_pos[i,:], positions[i,:])

# for i in range(0,n_walkers):
#     #3. This gives nearest vector to each spin
#     vector_to_spin[i] = vloc[index[i]] - positions[i]

# mesh_path  = os.path.join(
#     os.path.dirname(simulations.__file__), "tests","vascular_mesh_22-10-04_21-52-57_r4.ply")
# mesh = meshio.read(mesh_path)
# vertices = mesh.points.astype(np.float32) * 10e-6
# faces = mesh.cells[0].data

# vess = utils.segment_mesh(mesh)

# #importing MATLAB File
# mat_path = os.path.join(
#     os.path.dirname(simulations.__file__),"tests", "for_ronit_vloc_vdir_mesh.mat")
# mat = scp.loadmat(mat_path)

# vdir = mat['vdir']
# vdir = np.array(vdir) * 10e-6

# vloc = mat['vloc']
# vloc = np.array(vloc) * 10e-6

# seg_length = mat['seg_length']
# seg_length = np.array(seg_length)

# ax = plt.figure().add_subplot(projection='3d')
# for i in range(0, np.size(vdir,0)):
#     rand = np.random.rand(3)
#     scl = seg_length[i]
#     ax.quiver(vloc[i,0],vloc[i,1],vloc[i,2],scl*vdir[i,0],scl*vdir[i,1],scl*vdir[i,2], color=rand)
#plot mesh transparent on top'

# #testing
# dt = 80e-3/(vdir.shape[1] - 1)
# dist = simulations.brain_flow(vdir,2e-9, dt)

# # path travelled by walker
# for i in range(0, np.size(dist,0)):
#     ax.plot(dist[i,0],dist[i,1],dist[i,2])
    


# #Mesh plotted with Matplotlib trisurf



# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_trisurf(
#     vertices[:, 0],
#     vertices[:, 1],
#     vertices[:, 2],
#     triangles=faces,
# )
# # ax.axes.xaxis.set_ticklabels([])
# # ax.axes.yaxis.set_ticklabels([])
# # ax.axes.zaxis.set_ticklabels([])
# ax.set_title("Vascular Mesh")
# plt.show()

# #running sim
# traj_file = "vascular_mesh_traj.txt"

# #padded = -1 * np.average(vertices, axis=0) * np.ones(3)
# padded = 0 * np.ones(3)
# substrate = substrates.mesh(vertices, faces, periodic=True, init_pos="intra", padding=padded, n_sv=np.array([100, 100, 100]))
# substrate.init_pos
# #intra bc pos should start in mesh

# ##uncomment code below to show mesh, but takes very long
# # utils.show_mesh(substrate) 

# gradient = np.zeros((1, 100, 3))
# gradient[0, 1:30, 0] = 1
# gradient[0, 70:99, 0] = -1
# T = 80e-3  # Duration in seconds


# # Increase the number of time points

# n_t = int(1e3)  # Number of time points in the simulation
# dt = T / (gradient.shape[1] - 1)  # Time step duration in seconds
# gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)

# #concat grad arrays with dif b values
# bs = np.linspace(0, 3e9, 100)  # SI units (s/m^2)
# gradient = np.concatenate([gradient for _ in bs], axis=0)
# gradient = gradients.set_b(gradient, dt, bs)


# signals = simulations.simulation(
#     n_walkers = int(1e3),
#     diffusivity = 2e-9,
#     gradient = gradient,
#     dt = dt,
#     substrate = substrate,
#     ballistics = 0.0,
#     traj = traj_file,
#     cuda_bs = 512
#     )
# utils.show_traj(traj_file)

# #plotting simulated sig

# fig, ax = plt.subplots(1, figsize=(7, 4))
# ax.scatter(bs, signals / int(1e3), s=10)
# ax.set_xlabel("b (s/m$^2$)")
# ax.set_ylabel("S/S$_0$")
# ax.set_title("Signal Attenuation for Vascular Mesh")
# plt.show()