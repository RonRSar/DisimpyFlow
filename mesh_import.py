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
n_walkers = 1000
vloc_long = vloc_long * 10e-6
#1. initialise walkers
#scale = np.abs(np.max(vloc)) + np.abs(np.min(vloc))
#positions = np.random.rand(n_walkers,3)*scale + np.min(vloc)
substrate = substrates.mesh(vertices, faces, periodic=True, init_pos="intra", n_sv=np.array([200, 200, 200]))
positions = simulations._fill_mesh(n_walkers, substrate, intra=True,seed=123, cuda_bs=512)
dist = 0
step = 1e-3*80e-3/(100-1) #v*dt
max_iter = 1000
#getting position through time?
# orig_pos = positions

# #2. nearest neighbour search of walker with vloc
# tree = KDTree(vloc_long)
# #d, index = tree.query(positions,k=1) 
# #vector_to_spin = vloc[index] - positions


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
# ax = fig.add_subplot(111,projection='3d')
# ax.set_title("Spins Through Time")
# for i in range(0,np.size(time_pos,0)):
#     ax.plot(time_pos[i,:,0],time_pos[i,:,1],time_pos[i,:,2],linewidth=0.5,markevery=5,color='r')
    
# ax.plot_trisurf(vertices[:, 0],
#                 vertices[:, 1],
#                 vertices[:, 2],
#                 triangles=faces,
#                 alpha=0.4)

# def is_inside(triangles, X):
# 	# Compute euclidean norm along axis 1
# 	def anorm2(X):
# 		return np.sqrt(np.sum(X ** 2, axis = 1))



# 	# Compute 3x3 determinant along axis 1
# 	def adet(X, Y, Z):
# 		ret  = np.multiply(np.multiply(X[:,0], Y[:,1]), Z[:,2])
# 		ret += np.multiply(np.multiply(Y[:,0], Z[:,1]), X[:,2])
# 		ret += np.multiply(np.multiply(Z[:,0], X[:,1]), Y[:,2])
# 		ret -= np.multiply(np.multiply(Z[:,0], Y[:,1]), X[:,2])
# 		ret -= np.multiply(np.multiply(Y[:,0], X[:,1]), Z[:,2])
# 		ret -= np.multiply(np.multiply(X[:,0], Z[:,1]), Y[:,2])
# 		return ret



# 	# One generalized winding number per input vertex
# 	ret = np.zeros(X.shape[0], dtype = X.dtype)
# 	
# 	# Accumulate generalized winding number for each triangle
# 	for U, V, W in triangles:	
# 		A, B, C = U - X, V - X, W - X
# 		omega = adet(A, B, C)

# 		a, b, c = anorm2(A), anorm2(B), anorm2(C)
# 		k  = a * b * c 
# 		k += c * np.sum(np.multiply(A, B), axis = 1)
# 		k += a * np.sum(np.multiply(B, C), axis = 1)
# 		k += b * np.sum(np.multiply(C, A), axis = 1)

# 		ret += np.arctan2(omega, k)

# 	# Job done
# 	return ret >= 2 * np.pi 

# triangles = vertices
# min_corner = np.amin(np.amin(triangles, axis = 0), axis = 0)
# max_corner = np.amax(np.amax(triangles, axis = 0), axis = 0)
# P = (max_corner - min_corner) * np.random.random((8198, 3)) + min_corner

# # Filter out points which are not inside the mesh
# P = P[is_inside(triangles, P)]

# # Display
# fig = plt.figure()
# ax = fig.gca(projection = '3d')
# ax.scatter(P[:,0], P[:,1], P[:,2], lw = 0., c = 'k')
# plt.show()

# for i in range(0,n_walkers):
#     #3. This gives nearest vector to each spin
#     vector_to_spin[i] = vloc[index[i]] - positions[i]

mesh_path  = os.path.join(
    os.path.dirname(simulations.__file__), "tests","vascular_mesh_22-10-04_21-52-57_r4.ply")
mesh = meshio.read(mesh_path)
vertices = mesh.points.astype(np.float32) * 10e-6
faces = mesh.cells[0].data

# vess = utils.segment_mesh(mesh)

#importing MATLAB File
mat_path = os.path.join(
    os.path.dirname(simulations.__file__),"tests", "for_ronit_vloc_vdir_mesh.mat")
mat = scp.loadmat(mat_path)

vdir = mat['vdir']
vdir = np.array(vdir) * 10e-6

vloc = mat['vloc']
vloc = np.array(vloc) * 10e-6

seg_length = mat['seg_length']
seg_length = np.array(seg_length)

# ax = plt.figure().add_subplot(projection='3d')
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

# #running sim
traj_file = "vascular_mesh_traj.txt"

# #padded = -1 * np.average(vertices, axis=0) * np.ones(3)
# padded = 0 * np.ones(3)
substrate = substrates.mesh(vertices, faces, periodic=True, init_pos="intra", n_sv=np.array([100, 100, 100]))
# #intra bc pos should start in mesh

##uncomment code below to show mesh, but takes very long
# utils.show_mesh(substrate) 

gradient = np.zeros((1, 100, 3))
gradient[0, 1:30, 0] = 1
gradient[0, 70:99, 0] = -1
T = 80e-3  # Duration in seconds


# Increase the number of time points

n_t = int(1e3)  # Number of time points in the simulation
dt = T / (gradient.shape[1] - 1)  # Time step duration in seconds
gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)

#concat grad arrays with dif b values
bs = np.linspace(0, 3e9, 100)  # SI units (s/m^2)
gradient = np.concatenate([gradient for _ in bs], axis=0)
gradient = gradients.set_b(gradient, dt, bs)


signals = simulations.simulation_flow(
    n_walkers = int(1e3),
    diffusivity = 2e-9,
    gradient = gradient,
    dt = dt,
    substrate = substrate,
    vdir = vdir,
    vloc = vloc, 
    v = 1.0,
    ballistics = 0.0,
    traj = traj_file,
    cuda_bs = 512
    )
utils.show_traj(traj_file)

# #plotting simulated sig

fig, ax = plt.subplots(1, figsize=(7, 4))
ax.scatter(bs, signals / int(1e3), s=10)
ax.set_xlabel("b (s/m$^2$)")
ax.set_ylabel("S/S$_0$")
ax.set_title("Signal Attenuation for Vascular Mesh")
plt.show()