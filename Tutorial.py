# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:35:26 2022

@author: ronit
"""
###Test

#Initialisation
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import meshio

from disimpy import gradients, simulations, substrates, utils

#Simulation Parameters

n_walkers = int(1e3)
diffusivity = 2e-9  # SI units (m^2/s)

# Create a simple Stejskal-Tanner gradient waveform

gradient = np.zeros((1, 100, 3))
gradient[0, 1:30, 0] = 1
gradient[0, 70:99, 0] = -1
T = 80e-3  # Duration in seconds


# Increase the number of time points

n_t = int(1e3)  # Number of time points in the simulation
dt = T / (gradient.shape[1] - 1)  # Time step duration in seconds
gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)


# Concatenate 100 gradient arrays with different b-values together

bs = np.linspace(0, 3e9, 100)  # SI units (s/m^2)
gradient = np.concatenate([gradient for _ in bs], axis=0)
gradient = gradients.set_b(gradient, dt, bs)


# Show gradient magnitude over time for the last measurement

# fig, ax = plt.subplots(1, figsize=(7, 4))
# for i in range(3):
#     ax.plot(np.linspace(0, T, n_t), gradient[-1, :, i])
# ax.legend(["G$_x$", "G$_y$", "G$_z$"])
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Gradient magnitude (T/m)")
# ax.set_title("Gradient Magnitude over Time for a Stejskal-Tanner Waveform")
# plt.show()

# # Free Diffusion

# # Create a substrate object for free diffusion

# substrate = substrates.free()


# Run simulation and show the random walker trajectories

# traj_file = "free_traj.txt"
# signals = simulations.simulation(
#     n_walkers=n_walkers,
#     diffusivity=diffusivity,
#     gradient=gradient,
#     dt=dt,
#     substrate=substrate,
#     traj=traj_file,
#     ballistics = 1.0
#)
# utils.show_traj(traj_file)


# # Plot the simulated signal

# fig, ax = plt.subplots(1, figsize=(7, 4))
# ax.scatter(bs, signals / n_walkers, s=10)
# ax.set_xlabel("b (s/m$^2$)")
# ax.set_ylabel("S/S$_0$")
# ax.set_title("Free Diffusion Plot")
# plt.show()

## Test flow
# substrate = substrates.cylinder(radius=5e-6,   
#                                 orientation=np.array([1.0, 1.0, 1.0]))

# test_1 = simulations.brain_flow_cylinder(n_walkers=n_walkers,
#                                  diffusivity=diffusivity,
#                                  gradient=gradient,
#                                  dt=dt,
#                                  substrate=substrate,
#                                  traj=traj_file,
#                                  )

## Sphere Mesh
# # Create a substrate object for diffusion inside a sphere

# substrate = substrates.sphere(radius=5e-6)


# # Run simulation and show the random walker trajectories

# traj_file = "example_traj.txt"
# signals = simulations.simulation(
#     n_walkers=n_walkers,
#     diffusivity=diffusivity,
#     gradient=gradient,
#     dt=dt,
#     substrate=substrate,
#     traj=traj_file,
# )
# utils.show_traj(traj_file)


# # Plot the simulated signal

# fig, ax = plt.subplots(1, figsize=(7, 4))
# ax.scatter(bs, signals / n_walkers, s=10)
# ax.set_xlabel("b (s/m$^2$)")
# ax.set_ylabel("S/S$_0$")
# ax.set_title("Diffusion inside Sphere")
# plt.show()

## Cylinder Mesh

# # Create a substrate object for diffusion inside an infinite cylinder

# substrate = substrates.cylinder(
#     radius=5e-6,
#     orientation=np.array([1.0, 1.0, 1.0])
# )


# # Run simulation and show the random walker trajectories

# traj_file = "example_traj.txt"
# signals = simulations.simulation(
#     n_walkers=n_walkers,
#     diffusivity=diffusivity,
#     gradient=gradient,
#     dt=dt,
#     substrate=substrate,
#     traj=traj_file,
# )
# utils.show_traj(traj_file)


# # Plot the simulated signal

# fig, ax = plt.subplots(1, figsize=(7, 4))
# ax.scatter(bs, signals / n_walkers, s=10)
# ax.set_xlabel("b (s/m$^2$)")
# ax.set_ylabel("S/S$_0$")
# ax.set_title("Diffusion inside Cylinder")
# plt.show()

# ## Ellipsoids Mesh

# # Create a substrate object for diffusion inside an ellipsoid

# v = np.array([1.0, 0, 0])
# k = np.array([1.0, 1.0, 1.0])
# R = utils.vec2vec_rotmat(v, k)  # Rotation matrix for aligning v with k
# substrate = substrates.ellipsoid(
#     semiaxes=np.array([10e-6, 5e-6, 2.5e-6]),
#     R=R,
# )


# # Run simulation and show the random walker trajectories

# traj_file = "example_traj.txt"
# signals = simulations.simulation(
#     n_walkers=n_walkers,
#     diffusivity=diffusivity,
#     gradient=gradient,
#     dt=dt,
#     substrate=substrate,
#     traj=traj_file,
# )
# utils.show_traj(traj_file)


# # Plot the simulated signal

# fig, ax = plt.subplots(1, figsize=(7, 4))
# ax.scatter(bs, signals / n_walkers, s=10)
# ax.set_xlabel("b (s/m$^2$)")
# ax.set_ylabel("S/S$_0$")
# ax.set_title("Diffusion inside Ellipsoid")
# plt.show()

## Triangular Mesh

# Load an example triangular mesh

# mesh_path = os.path.join(
#     os.path.dirname(simulations.__file__), "tests", "example_mesh.pkl"
# )
# with open(mesh_path, "rb") as f:
#     example_mesh = pickle.load(f)
# faces = example_mesh["faces"]
# vertices = example_mesh["vertices"]

# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_trisurf(
#     vertices[:, 0],
#     vertices[:, 1],
#     vertices[:, 2],
#     triangles=faces,
# )
# plt.show()
# # Create a substrate object

# substrate = substrates.mesh(
#     vertices, faces, padding=np.zeros(3), periodic=True, init_pos="intra"
# )


# ## Show the mesh

# # utils.show_mesh(substrate)


# # Run simulation and show the random walker trajectories

# traj_file = "triangular_traj.txt"
# signals = simulations.simulation(
#     n_walkers=n_walkers,
#     diffusivity=diffusivity,
#     gradient=gradient,
#     dt=dt,
#     substrate=substrate,
#     traj=traj_file,
# )
# utils.show_traj(traj_file)


# # Plot the simulated signal

# fig, ax = plt.subplots(1, figsize=(7, 4))
# ax.scatter(bs, signals / n_walkers, s=10)
# ax.set_xlabel("b (s/m$^2$)")
# ax.set_ylabel("S/S$_0$")
# ax.set_title("Triangular Mesh")
# plt.show()

# Load an example triangular mesh

# #load mesh
# mesh_path  = os.path.join(
#     os.path.dirname(simulations.__file__), "tests","example_mesh.pkl")
# with open(mesh_path, "rb") as f:
#     example_mesh = pickle.load(f)
# points = [example_mesh['faces']]
# cells = ["triangle",(example_mesh['vertices'])]

# test_mesh = meshio.Mesh(points,cells)
# meshio.write_points_cells("foo.vtk", points, cells)

# mesh_path_vasc  = os.path.join(
#     os.path.dirname(simulations.__file__), "tests","vascular_mesh_22-10-04_21-52-57_r4.ply")
# mesh_vasc = meshio.read(mesh_path_vasc)

# faces = example_mesh["faces"]
# vertices = example_mesh["vertices"]

# vertices = mesh.points.astype(np.float32)
# faces = mesh.cells[0].data


# # Plot

# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection="3d")
# ax.plot_trisurf(
#     vertices[:, 0],
#     vertices[:, 1],
#     vertices[:, 2],
#     triangles=faces,
# )
# ax.axes.xaxis.set_ticklabels([])
# ax.axes.yaxis.set_ticklabels([])
# ax.axes.zaxis.set_ticklabels([])
# ax.set_title("Triangular Mesh")
# plt.show()


# # Create a substrate object

# substrate = substrates.mesh(
#     vertices, faces, padding=np.zeros(3), periodic=True, init_pos="intra"
# )


# # Show the mesh

# utils.show_mesh(substrate)