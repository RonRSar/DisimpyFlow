# -*- coding: utf-8 -*-
"""

@author: ronit
"""
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def show_traj(traj_file):
    """Plot walker trajectories saved in a trajectories file.

    Parameters
    ----------
    traj_file : str
        Path of a trajectories file where every line represents a time point
        and every line contains the positions as follows: walker_1_x walker_1_y
        walker_1_z walker_2_x walker_2_y walker_2_z...

    Returns
    -------
    None
    """
    trajectories = np.loadtxt(traj_file)
    trajectories = trajectories.reshape(
        (trajectories.shape[0], int(trajectories.shape[1] / 3), 3)
    )
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(trajectories.shape[1]):
        ax.plot(
            trajectories[:, i, 0],
            trajectories[:, i, 1],
            trajectories[:, i, 2],
            alpha=0.5,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.ticklabel_format(style="sci", scilimits=(0, 0))
    fig.tight_layout()
    plt.show()
    return

vasc = "vascular_mesh_traj.txt"
free = "free_traj.txt"
example = "triangular_traj.txt"

vasc_traj = np.loadtxt(vasc)
free_traj = np.loadtxt(free)
example_traj = np.loadtxt(example)

vasc_traj = vasc_traj.reshape(vasc_traj.shape[0], int(vasc_traj.shape[1]/3), 3)
free_traj = free_traj.reshape(free_traj.shape[0], int(free_traj.shape[1]/3), 3)
example_traj = example_traj.reshape(example_traj.shape[0], int(example_traj.shape[1]/3), 3)

# show_traj(vasc)
# show_traj(free)

# test = np.diff(vasc_traj, axis=0)
test2 = vasc_traj[-1] - vasc_traj[0]