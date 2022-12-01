# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 01:22:37 2022

@author: ronit
"""

from numba import cuda
import numba
import math

from numba.cuda.random import (
    create_xoroshiro128p_states,
    xoroshiro128p_normal_float64,
    xoroshiro128p_uniform_float64,
)

from disimpy.gradients import GAMMA

@cuda.jit(device=True)
def _cuda_dot_product(a, b):
    """Calculate the dot product between two 1D arrays of length 3.

    Parameters
    ----------
    a : numba.cuda.cudadrv.devicearray.DeviceNDArray
    b : numba.cuda.cudadrv.devicearray.DeviceNDArray

    Returns
    -------
    float
    """
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

@cuda.jit(device=True)
def _cuda_normalize_vector(v):
    """Scale 1D array of length 3 so that it has unit length.

    Parameters
    ----------
    v : numba.cuda.cudadrv.devicearray.DeviceNDArray

    Returns
    -------
    None
    """
    length = math.sqrt(_cuda_dot_product(v, v))
    for i in range(3):
        v[i] = v[i] / length
    return

@cuda.jit(device=True)
def _cuda_random_step(step, rng_states, thread_id):
    """Generate a random step from a uniform distribution over a sphere.

    Parameters
    ----------
    step : numba.cuda.cudadrv.devicearray.DeviceNDArray
    rng_states : numba.cuda.cudadrv.devicearray.DeviceNDArray
    thread_id : int

    Returns
    -------
    None
    """
    for i in range(3):
        step[i] = xoroshiro128p_normal_float64(rng_states, thread_id)
    _cuda_normalize_vector(step)
    return

@cuda.jit()
def _cuda_step_free(positions, g_x, g_y, g_z, phases, rng_states, t, step_l, dt):
    """Kernel function for free diffusion."""
    thread_id = cuda.grid(1) #abs pos in current thread of grid
    if thread_id >= positions.shape[0]:#if thread is more than length of positions          
        return
    step = cuda.local.array(3, numba.float64)   #step array made             
    _cuda_random_step(step, rng_states, thread_id)        # random steps taken   
    for i in range(3):
        positions[thread_id, i] = positions[thread_id, i] + step[i] * step_l #xyz positions = position + stepsize*steplength
    for m in range(g_x.shape[0]): #for gradient, the phases are Gamma*dt(dot product gradient with positions)
        phases[m, thread_id] += (
            GAMMA
            * dt
            * (
                (g_x[m, t] * positions[thread_id, 0])
                + (g_y[m, t] * positions[thread_id, 1])
                + (g_z[m, t] * positions[thread_id, 2])
            )
        )
    return

@cuda.jit()
def _test_cuda():
    thread_id = cuda.grid(1)
    return print(thread_id)


@cuda.jit(device=True)
def _cuda_line_sphere_intersection(r0, step, radius):
    """Calculate the distance from r0 to a sphere centered at origin along
    step. r0 must be inside the sphere.

    Parameters
    ----------
    r0 : numba.cuda.cudadrv.devicearray.DeviceNDArray
    step : numba.cuda.cudadrv.devicearray.DeviceNDArray
    radius : float

    Returns
    -------
    float
    """
    dp = _cuda_dot_product(step, r0)
    d = -dp + math.sqrt(dp ** 2 - (_cuda_dot_product(r0, r0) - radius ** 2))
    return d

@numba.jit(nopython=True)
def _cuda_reflection(r0, step, d, normal, epsilon):
    """Calculate reflection and update r0 and step. Epsilon is the amount by
    which the new position differs from the reflection point.

    Parameters
    ----------
    r0 : numba.cuda.cudadrv.devicearray.DeviceNDArray
    step : numba.cuda.cudadrv.devicearray.DeviceNDArray
    d : float
    normal : numba.cuda.cudadrv.devicearray.DeviceNDArray
    semiaxes : numba.cuda.cudadrv.devicearray.DeviceNDArray
    epsilon : float

    Returns
    -------
    float
    """
    intersection = cuda.local.array(3, numba.float64)
    v = cuda.local.array(3, numba.float64)
    for i in range(3):
        intersection[i] = r0[i] + d * step[i]
        v[i] = intersection[i] - r0[i]
    dp = _cuda_dot_product(v, normal)
    if dp > 0:  # Make sure the normal vector points against the step
        for i in range(3):
            normal[i] *= -1
        dp = _cuda_dot_product(v, normal)
    for i in range(3):
        step[i] = (v[i] - 2 * dp * normal[i] + intersection[i]) - intersection[i]
    _cuda_normalize_vector(step)
    for i in range(3):  # Move walker slightly away from the surface
        r0[i] = intersection[i] + epsilon * normal[i]
    return

@cuda.jit()
def _cuda_step_sphere(
    positions,
    g_x,
    g_y,
    g_z,
    phases,
    rng_states,
    t,
    step_l,
    dt,
    radius,
    iter_exc,
    max_iter,
    epsilon,
):
    """Kernel function for diffusion inside a sphere."""
    thread_id = cuda.grid(1) #same deal with free
    if thread_id >= positions.shape[0]:
        return
    step = cuda.local.array(3, numba.float64) # same as free
    _cuda_random_step(step, rng_states, thread_id)
    r0 = positions[thread_id, :]
    iter_idx = 0
    check_intersection = True #check intersection
    while check_intersection and step_l > 0 and iter_idx < max_iter:
        iter_idx += 1
        d = _cuda_line_sphere_intersection(r0, step, radius) #is it intersecting
        if d > 0 and d < step_l:
            normal = cuda.local.array(3, numba.float64) #normal array
            for i in range(3):
                normal[i] = -(r0[i] + d * step[i])
            _cuda_normalize_vector(normal)
            _cuda_reflection(r0, step, d, normal, epsilon) #reflection
            step_l -= d + epsilon
        else:
            check_intersection = False
    if iter_idx >= max_iter:
        iter_exc[thread_id] = True
    for i in range(3):
        positions[thread_id, i] = r0[i] + step[i] * step_l
    for m in range(g_x.shape[0]):
        phases[m, thread_id] += (
            GAMMA
            * dt
            * (
                (g_x[m, t] * positions[thread_id, 0])
                + (g_y[m, t] * positions[thread_id, 1])
                + (g_z[m, t] * positions[thread_id, 2])
            )
        )
    return