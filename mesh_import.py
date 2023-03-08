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
import math
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
n_walkers = 1000
vloc = vloc * 10e-6
vloc_long = vloc_long * 10e-6
vdir_long = vdir_long
#1. initialise walkers
#scale = np.abs(np.max(vloc)) + np.abs(np.min(vloc))
#positions = np.random.rand(n_walkers,3)*scale + np.min(vloc)
# substrate = substrates.mesh(vertices, faces, periodic=True, init_pos="intra", n_sv=np.array([200, 200, 200]))
# positions = simulations._fill_mesh(n_walkers, substrate, intra=True,seed=123, cuda_bs=512)
# # dist = 0
# # step = 1e-3*80e-3/(100-1) #v*dt
# # max_iter = 1000
# #getting position through time?
# # orig_pos = positions

# # #2. nearest neighbour search of walker with vloc
# tree = KDTree(vloc_long)
# d, index = tree.query(positions,k=1) 
# vector_to_spin = vloc_long[index] - positions


# itera = 0
# time_pos = np.zeros((n_walkers,max_iter,3))
# while itera < max_iter:
#     d, index = tree.query(positions,k=1)
    
#     # Track each spin pos in time
#     time_pos[:,itera,:] = positions
    
#     #4. step in direction of NN search, step of distance
#     positions = positions + step*vdir_long[index]
    
#     itera += 1
 
# dist = positions - orig_pos

# #checking
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.set_title("Positional Vectors in Mesh")
# for i in range(0,np.size(dist,0)):
#     rand = np.random.rand(3)
#     ax.quiver(orig_pos[i,0],orig_pos[i,1],orig_pos[i,2],dist[i,0],dist[i,1],dist[i,2],color=rand)

# #plot spins through time
# #in time_pos axis 0 is spins, axis 1 is time
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# # ax.set_title("Spins Through Time")
# # for i in range(0,np.size(time_pos,0)):
# #     ax.plot(time_pos[i,:,0],time_pos[i,:,1],time_pos[i,:,2],linewidth=0.5,markevery=5,color='r')

# shifted_vertices = vertices - np.min(vertices, axis=0) 
# ax.plot_trisurf(shifted_vertices[:, 0],
#                 shifted_vertices[:, 1],
#                 shifted_vertices[:, 2],
#                 triangles=faces
#                 )
# ax.set_title("Shifted Mesh")

# for i in range(0,n_walkers):
#     #3. This gives nearest vector to each spin
#     vector_to_spin[i] = vloc[index[i]] - positions[i]

# vess = utils.segment_mesh(mesh)

shift = -np.min(vloc_long, axis=0)
vloc_long = vloc_long + shift
vdir_long = vdir_long + shift

vloc_shift = vloc + shift
vdir_shift = vdir + shift

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.set_title("Quiver Plot of Velocity Vectors")
# for i in range(0, np.size(vdir,0)):
#      rand = np.random.rand(3)
#      scl = seg_length[i]*10e-6
#      ax.quiver(vloc_shift[i,0],vloc_shift[i,1],vloc_shift[i,2],
#                scl*vdir_shift[i,0],scl*vdir_shift[i,1],scl*vdir_shift[i,2],
#                color=rand)
# fig.tight_layout()
     
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for i in range(0, np.size(vdir,0)):
#     rand = np.random.rand(3)
#     scl = seg_length[i]
#     ax.quiver(vloc[i,0],vloc[i,1],vloc[i,2],scl*vdir[i,0],scl*vdir[i,1],scl*vdir[i,2], color=rand)
# plot mesh transparent on top'

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

#running sim
traj_file = "example_flow_traj.txt"


substrate = substrates.mesh(vertices, faces, periodic=True, init_pos="intra", n_sv=np.array([200, 200, 200]))
# # #intra bc pos should start in mesh

##uncomment code below to show mesh, but takes very long
# utils.show_mesh(substrate) 

gradient = np.zeros((1, 100, 3))
gradient[0, 1:30, 0] = 1
gradient[0, 70:99, 0] = -1
T = 80e-3  # Duration in seconds

v = 0.5e-3 # not more than 1 
v_m = 6 + np.round(math.log10(v))

# Increase the number of time points

n_t = int(10**v_m)  # Number of time points in the simulation #nt scaled based on v
dt = T / (gradient.shape[1] - 1)  # Time step duration in seconds
gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)

#concat grad arrays with dif b values
bs = np.linspace(0, 3e9, 100)  # SI units (s/m^2)
gradient = np.concatenate([gradient for _ in bs], axis=0)
gradient = gradients.set_b(gradient, dt, bs)


signals, phases = simulations.simulation_flow(
    n_walkers = int(1e3),
    diffusivity = 2e-9,
    gradient = gradient,
    dt = dt,
    substrate = substrate,
    vdir = vdir_long,
    vloc = vloc_long, 
    v = v,
    traj = traj_file,
    cuda_bs = 512
    )
utils.show_traj(traj_file)

# #plotting simulated sig

fig, ax = plt.subplots(1, figsize=(7, 4))
ax.scatter(bs, np.abs(signals) / int(1e3), s=10)
ax.set_xlabel("b (s/m$^2$)")
ax.set_ylabel("|S/S$_0$|")
ax.text(1.5e9,1, 'Number of timepoints: %s ' %n_t)
ax.text(1.5e9,0.9, 'Magnitude of v: %s' %np.round(np.log10(v), decimals=1))
ax.set_title("Signal Attenuation for Vascular Mesh")
plt.show()



# trajectories = np.loadtxt(traj_file)
# trajectories = trajectories.reshape(
#     (trajectories.shape[0], int(trajectories.shape[1] / 3), 3)
#     )

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_trisurf(
#     shifted_vertices[:, 0],
#     shifted_vertices[:, 1],
#     shifted_vertices[:, 2],
#     triangles=faces,
# )
# for i in range(trajectories.shape[1]):
#     ax.plot(
#             trajectories[:, i, 0],
#             trajectories[:, i, 1],
#             trajectories[:, i, 2],
#             alpha=0.5,
#         )
# #ax.set_title("For walker = %d" % (i))
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# ax.ticklabel_format(style="sci", scilimits=(0, 0))
# fig.tight_layout()
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.set_title("Spins Through Time")
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# ax.ticklabel_format(style="sci", scilimits=(0, 0))
# ax.plot(trajectories[1,:,0],trajectories[1,:,1],trajectories[1,:,2],
#         linewidth=0, marker='o', markersize=1.0, color='r')
# ax.plot_trisurf(
#     shifted_vertices[:, 0],
#     shifted_vertices[:, 1],
#     shifted_vertices[:, 2],
#     triangles=faces,
#     alpha = 0.4
# )
    
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.set_title("Phases for v=0.001")
# ax.imshow(phases,'hot')
