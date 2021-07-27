import pandas as pd
from sklearn.preprocessing import StandardScaler
from ase.io import Trajectory, read, write
import numpy as np
import os
import re
from ase.db import connect
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from ase.constraints import constrained_indices


def pca(traj_dict, fig_title=None):
    """Perform a PCA analysis for the trajectories.
    Parameters
    ----------
    traj_dict: dict
        Dictionary of calculation types and trajectories.
        e.g.: {'oal': oal_traj}, where oal_traj is an ASE Trajectory
        object or the path to the trajectory.
        or {'oal': [path_to_oal_traj, path_to_oal_db]}, where path_to_oal_traj
        is the path to the trajectory and path_to_oal_db is the path to the ase Database.

    fig_title: str
        Title of the PCA plot.
    """
    types = []
    trajs = []
    pos = []
    energy = []
    for key, value in traj_dict.items():
        types.append(key)
        if isinstance(value, str):
            value = Trajectory(value)
            trajs.append(value)
        elif isinstance(value, list):
            traj = Trajectory(value[0])
            db = connect(value[1])
            dft_call = [i.id - 1 for i in db.select(check=True)]
            dft_traj = [traj[i] for i in db_dft_call]
            trajs.append(dft_traj)
        else:
            trajs.append(value)
    # assuming constraints (FixAtom) are applied to the same atoms
    constrained_id = constrained_indices(trajs[0][0])
    for i in trajs:
        all_pos = [j.get_positions() for j in i]
        free_atom_pos = [np.delete(i, constrained_id, 0) for i in all_pos]
        pos.append([j.get_positions() for j in i])
        energy.append([j.get_potential_energy() for j in i])
    label = []
    for i in range(len(types)):
        label += [types[i]] * len(pos[i])
    attr = []
    for i in range(1, np.shape(pos[0])[1] + 1):
        attr.append("%dx" % i)
        attr.append("%dy" % i)
        attr.append("%dz" % i)
    df = []
    for i in range(len(pos)):
        reshape = np.array(pos[i]).reshape(np.shape(pos[i])[0], np.shape(pos[i])[1] * 3)
        df.append(pd.DataFrame(reshape, columns=attr))
    df = pd.concat([df[i] for i in range(len(df))], ignore_index=True)

    df.insert(len(df.columns), "label", label)

    x = df.loc[:, attr].values
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(
        data=principalComponents,
        columns=["principal component 1", "principal component 2"],
    )
    finalDf = pd.concat([principalDf, df[["label"]]], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    if fig_title is not None:
        ax.set_title(fig_title, fontsize=20)
    else:
        ax.set_title("Principal Component Analysis", fontsize=20)
    targets = types
    colors = energy
    mark = ["v", "^", "<", ">", "o", "s", "D"]
    for target, color, mark in zip(targets, colors, mark):
        indicesToKeep = finalDf["label"] == target
        #     if target == 'oal':
        ax.scatter(
            finalDf.loc[indicesToKeep, "principal component 1"],
            finalDf.loc[indicesToKeep, "principal component 2"],
            c=color,
            marker=mark,
            cmap="winter",
            s=50,
            label=target,
        )
        ax.plot(
            finalDf.loc[indicesToKeep, "principal component 1"],
            finalDf.loc[indicesToKeep, "principal component 2"],
        )
    sm = plt.cm.ScalarMappable(cmap="winter")
    fig.colorbar(sm)
    ax.legend()
    plt.savefig("pca.png")
