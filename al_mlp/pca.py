import pandas as pd
from sklearn.preprocessing import StandardScaler
from ase.io import Trajectory, read, write
import numpy as np
import os
import re
from ase.db import connect
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pca(traj_dict):
    """Perform a PCA analysis for the trajectories.
    Parameters
    ----------
    traj_dict: dict
        Dictionary of calculation types and trajectories.
        e.g.: {'oal': oal_traj}, where oal_traj is an ASE Trajectory object.
    """
    types = []
    trajs = []
    pos = []
    energy = []
    for key, value in traj_dict.items():
        types.append(key)
        trajs.append(value)
    for i in trajs:
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
    ax.set_title("2 component PCA", fontsize=20)
    targets = types
    colors = energy
    mark = ["x", "o", "^"]
    for target, color, mark in zip(targets, colors, mark):
        indicesToKeep = finalDf["label"] == target
        #     if target == 'oal':
        ax.scatter(
            finalDf.loc[indicesToKeep, "principal component 1"],
            finalDf.loc[indicesToKeep, "principal component 2"],
            c=color,
            marker=mark,
            cmap="winter",
            s=100,
        )
    #     else:
    #         ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
    #                    , finalDf.loc[indicesToKeep, 'principal component 2']
    #                    , c = color
    #                    , s = 50)
    sm = plt.cm.ScalarMappable(cmap="winter")
    fig.colorbar(sm)
    ax.legend(targets)
    plt.savefig("pca.png")
