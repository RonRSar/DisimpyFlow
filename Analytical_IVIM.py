# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 12:36:09 2023

@author: ronit
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from disimpy import gradients

gradient = np.zeros((1, 100, 3))
gradient[0, 1:30, 0] = 1
gradient[0, 70:99, 0] = -1
T = 80e-3  # Duration in seconds

v = 1e-3 # not more than 1 
v_m = 6 + np.round(math.log10(v))

# Increase the number of time points

n_t = int(10**v_m)  # Number of time points in the simulation #nt scaled based on v
dt = T / (gradient.shape[1] - 1)  # Time step duration in seconds
gradient, dt = gradients.interpolate_gradient(gradient, dt, n_t)

#concat grad arrays with dif b values
bs = np.linspace(0, 1e3, 10)  # SI units (s/m^2)
gradient = np.concatenate([gradient for _ in bs], axis=0)
gradient = gradients.set_b(gradient, dt, bs)


c = np.sqrt(bs*dt)
f = 1
D_b = 0
D_star = (v**2)*(c**2)/6*bs

S_sinc = f*np.exp(-bs*D_b)*np.sinc(c*v)
S_psuedo = f*np.exp(-bs*(D_star+D_b))

fig = plt.figure()
ax = fig.add_subplot()
ax.set_title("Sinc signal")
ax.set_xlabel("b (s/m$^2$)")
ax.set_ylabel("S/S$_0$")
ax.scatter(bs, S_sinc)

fig = plt.figure()
ax = fig.add_subplot()
ax.set_title("Psuedo signal")
ax.set_xlabel("b (s/m$^2$)")
ax.set_ylabel("S/S$_0$")
ax.scatter(bs, S_psuedo)
