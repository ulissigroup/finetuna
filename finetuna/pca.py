"""Principal component analysis module."""

from ase.atoms import Atoms
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ase.io import Trajectory
import numpy as np
from ase.db import connect
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from ase.constraints import constrained_indices
from flare_pp._C_flare import Structure, B2
from finetuna.ml_potentials.ocp_models.gemnet_t.pos_descriptor_gemnet_t import (
    PosDescriptorGemNetT,
)


class TrajPCA:
    """
    Perform PCA on a given trajectory object. Then save that analysis
    for use on later atoms objects parameters.
    """

    def __init__(
        self,
        traj,
        gemnet_descriptor_model_checkpoint_path=None,
    ):
        """
        Arguments
        ----------
        traj: Trajectory
            the parent Trajectory for this system to be compared to
        """
        if gemnet_descriptor_model_checkpoint_path is not None:
            self.descriptor_model = PosDescriptorGemNetT(
                gemnet_descriptor_model_checkpoint_path
            )

        self.species_map = init_species_map(traj[0])
        self.b2calc = B2(
            "chebyshev",
            "quadratic",
            [0, 5],
            [],
            [len(self.species_map), 12, 3],
        )

        energies = []
        des_list = []
        energies.append([j.get_potential_energy() for j in traj])
        for j in range(len(traj)):
            atoms = traj[j]
            structure_descriptor = Structure(
                atoms.get_cell(),
                [self.species_map[x] for x in atoms.get_atomic_numbers()],
                atoms.get_positions(),
                5,
                [self.b2calc],
            )
            des = structure_descriptor.descriptors[0].descriptors
            des_reshape = []
            for a in des:
                for b in a:
                    des_reshape.extend(np.ravel(np.array(b)))
            des_list.append(des_reshape)

        columns = []
        for i in range(np.shape(des_list[0])[-1]):
            columns.append(i)

        df = pd.DataFrame(des_list)
        self.keep_columns = ~df.eq(0).all()

        df = df.loc[:, self.keep_columns]
        columns = list(df.columns)
        sub_array = df.loc[:, columns].values
        self.standard_scaler = StandardScaler()
        transformed = self.standard_scaler.fit_transform(sub_array)

        self.pca = PCA(n_components=2)
        principal_components = self.pca.fit_transform(transformed)

    def analyze_image(self, image):
        """
        Arguments
        ----------
        image: Atoms
            the specific ase Atoms object to compare to the traj
        """
        image_structure_descriptor = Structure(
            image.get_cell(),
            [self.species_map[x] for x in image.get_atomic_numbers()],
            image.get_positions(),
            5,
            [self.b2calc],
        )
        des = image_structure_descriptor.descriptors[0].descriptors
        des_reshape = []
        for a in des:
            for b in a:
                des_reshape.extend(np.ravel(np.array(b)))
        df = pd.DataFrame([des_reshape]).loc[:, self.keep_columns]
        sub_array = df.values
        transformed_image = self.standard_scaler.transform(sub_array)

        pc_xy = self.pca.transform(transformed_image)

        x = pc_xy[0][0]
        y = pc_xy[0][1]
        return x, y

    def analyze_traj(self, traj):
        """
        Arguments
        ----------
        traj: Trajectory
            the specific ase Trajectory object to compare to the
            reference trajectory.
        """
        traj_pca_x = np.zeros(len(traj))
        traj_pca_y = np.zeros(len(traj))
        for idx, image in enumerate(traj):
            x, y = self.analyze_image(image)
            traj_pca_x[idx] = x
            traj_pca_y[idx] = y
        return [traj_pca_x, traj_pca_y]


def pca_xyz(traj_dict, fig_title=None):
    """Perform a PCA analysis for the trajectories.
    Parameters
    ----------
    traj_dict: dict
        Dictionary of calculation types and trajectories.
        e.g.: {'oal': oal_traj}, where oal_traj is an ASE Trajectory
        object or the path to the trajectory.
        or {'oal': [path_to_oal_traj, path_to_oal_db]}, where path_to_oal_traj
        is the path to the trajectory and path_to_oal_db is the path to the
        ase Database.
        or {'oal': [list_of_atoms]}, where the list of atoms serves as a
        trajectory to be read

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
        elif type(value) is list and type(value[0]) is Atoms:
            trajs.append(value)
        elif isinstance(value, list):
            traj = Trajectory(value[0])
            db = connect(value[1])
            dft_call = [i.id - 1 for i in db.select(check=True)]
            dft_traj = [traj[i] for i in dft_call]
            trajs.append(dft_traj)
        else:
            trajs.append(value)
    # assuming constraints (FixAtom) are applied to the same atoms
    constrained_id = constrained_indices(trajs[0][0])
    for i in trajs:
        all_pos = [j.get_positions() for j in i]
        # free_atom_pos = [np.delete(i, constrained_id, 0) for i in all_pos]
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
    mark = ["x", "s", "o", "^", "v", "<", ">", "D"]
    for target, color, mark in zip(targets, colors, mark):
        indicesToKeep = finalDf["label"] == target
        #     if target == 'oal':
        ax.scatter(
            finalDf.loc[indicesToKeep, "principal component 1"],
            finalDf.loc[indicesToKeep, "principal component 2"],
            c=color,
            marker=mark,
            cmap="viridis",
            s=50,
            label=target,
        )
        ax.plot(
            finalDf.loc[indicesToKeep, "principal component 1"],
            finalDf.loc[indicesToKeep, "principal component 2"],
        )
    sm = plt.cm.ScalarMappable(cmap="viridis")
    colorbar = fig.colorbar(sm)
    colorbar.set_label("-log(abs(energy))")
    ax.legend()
    plt.savefig("pca.png")


def init_species_map(image):
    species_map = {}
    a_numbers = np.unique(image.numbers)
    for i in range(len(a_numbers)):
        species_map[a_numbers[i]] = i
    return species_map


def des_pca(traj_dict, fig_title=None):
    """Perform a PCA analysis for the trajectories.
    Parameters
    ----------
    traj_dict: dict
        Dictionary of calculation types and trajectories.
        e.g.: {'oal': oal_traj}, where oal_traj is an ASE Trajectory
        object or the path to the trajectory.
        or {'oal': path_to_oal_db}, where path_to_oal_db is the path
        to the ASE Database.

    fig_title: str
        Title of the PCA plot.
    """
    types = []
    trajs = []
    energy = []
    des_list = []
    for key, value in traj_dict.items():
        types.append(key)
        des_list.append([])
        if isinstance(value, str):
            try:
                images = Trajectory(value)
            except:
                db = connect(value)
                images = [i.toatoms() for i in db.select(check=True)]
            finally:
                trajs.append(images)
        else:
            trajs.append(value)
    #     # assuming constraints (FixAtom) are applied to the same atoms
    #     constrained_id = constrained_indices(trajs[0][0])
    species_map = init_species_map(trajs[0][0])
    radial_hyps = [0, 5]
    settings = [len(species_map), 12, 3]
    B2calc = B2(
        "chebyshev",
        "quadratic",
        radial_hyps,
        [],
        settings,
    )

    for i in range(len(trajs)):
        energy.append([j.get_potential_energy() for j in trajs[i]])
        for j in range(len(trajs[i])):
            atoms = trajs[i][j]
            structure_descriptor = Structure(
                atoms.get_cell(),
                [species_map[x] for x in atoms.get_atomic_numbers()],
                atoms.get_positions(),
                5,
                [B2calc],
            )
            des = structure_descriptor.descriptors[0].descriptors
            des_reshape = []
            for a in des:
                for b in a:
                    des_reshape.extend(np.ravel(np.array(b)))
            des_list[i].append(des_reshape)

    label = []
    for i in range(len(types)):
        label += [types[i]] * len(des_list[i])

    attr = []
    for i in range(0, np.shape(des_list[0])[-1]):
        attr.append(i)

    df = []
    for i in range(len(des_list)):
        df.append(pd.DataFrame(des_list[i], columns=attr))
    df = pd.concat([df[i] for i in range(len(df))], ignore_index=True)

    df.insert(len(df.columns), "label", label)
    df = df.loc[:, ~df.eq(0).all()]
    attr = list(df.columns)[:-1]
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
    mark = ["x", "s", "o", "^", "v", "<", ">", "D"]
    for target, color, mark in zip(targets, colors, mark):
        indicesToKeep = finalDf["label"] == target
        #     if target == 'oal':
        ax.scatter(
            finalDf.loc[indicesToKeep, "principal component 1"],
            finalDf.loc[indicesToKeep, "principal component 2"],
            c=color,
            marker=mark,
            cmap="viridis",
            s=50,
            label=target,
        )
        ax.plot(
            finalDf.loc[indicesToKeep, "principal component 1"],
            finalDf.loc[indicesToKeep, "principal component 2"],
        )
    sm = plt.cm.ScalarMappable(cmap="viridis")
    ax.legend()
    colorbar = fig.colorbar(sm)
    colorbar.set_label("-log(abs(energy))")
    plt.savefig("pca.png")
