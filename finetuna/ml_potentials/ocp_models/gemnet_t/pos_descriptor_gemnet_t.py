from ocpmodels.models.gemnet.gemnet import GemNetT
from ocpmodels.common.registry import registry
import torch
import torch_geometric
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.gemnet.utils import (
    inner_product_normalized,
)
from torch_scatter import scatter
from ocpmodels.datasets.lmdb_dataset import data_list_collater
from tqdm import tqdm
from ocpmodels.common import distutils
import inspect
from ocpmodels.preprocessing import AtomsToGraphs


@registry.register_model("pos_descriptor_gemnet_t")
class PosDescriptorGemNetT(GemNetT):
    def __init__(
        self,
        checkpoint_path,
        cpu=True,
    ):

        if cpu:
            map_location = torch.device("cpu")
        else:
            map_location = torch.device(f"cuda:{0}")

        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        signature = inspect.signature(super().__init__)
        filter_keys = [param.name for param in signature.parameters.values()]
        filtered_dict = {
            filter_key: checkpoint["config"]["model_attributes"][filter_key]
            for filter_key in filter_keys
            if (
                (filter_key in checkpoint["config"]["model_attributes"])
                and (filter_key != "scale_file")
            )
        }

        super().__init__(num_atoms=0, bond_feat_dim=0, num_targets=1, **filtered_dict)

        self.a2g = AtomsToGraphs(
            max_neigh=self.max_neighbors,
            radius=self.cutoff,
            r_energy=True,
            r_forces=True,
            r_distances=True,
            r_edges=True,
        )

        first_key = next(iter(checkpoint["state_dict"]))
        if first_key.split(".")[0] == "module":
            new_dict = {k[7:]: v for k, v in checkpoint["state_dict"].items()}
            if next(iter(new_dict)).split(".")[0] == "module":
                new_dict = {k[7:]: v for k, v in new_dict.items()}
            self.load_state_dict(new_dict)
        else:
            self.load_state_dict(checkpoint["state_dict"])

    def get_positional_descriptor(self, atoms):
        data_object = self.a2g.convert(atoms)
        data_loader = data_list_collater([data_object])

        assert isinstance(
            data_loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()
        if isinstance(data_loader, torch_geometric.data.Batch):
            data_loader = [[data_loader]]

        descriptor = []

        for i, batch_list in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc="device {}".format(rank),
            disable=False,
        ):
            for batch in batch_list:
                out = self.forward(batch)
                descriptor.append(out)

        out_h = descriptor[0][0].detach().numpy()
        out_m = descriptor[0][1].detach().numpy()
        return out_h, out_m

    @conditional_grad(torch.enable_grad())
    def forward(self, data):
        pos = data.pos
        batch = data.batch
        atomic_numbers = data.atomic_numbers.long()

        if self.regress_forces and not self.direct_forces:
            pos.requires_grad_(True)

        (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        ) = self.generate_interaction_graph(data)
        idx_s, idx_t = edge_index

        # Calculate triplet angles
        cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        rad_cbf3, cbf3 = self.cbf_basis3(D_st, cosφ_cab, id3_ca)

        rbf = self.radial_basis(D_st)

        # Embedding block
        h = self.atom_emb(atomic_numbers)
        # (nAtoms, emb_size_atom)
        m = self.edge_emb(h, rbf, idx_s, idx_t)  # (nEdges, emb_size_edge)

        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(rad_cbf3, cbf3, id3_ca, id3_ragged_idx)

        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)

        E_t, F_st = self.out_blocks[0](h, m, rbf_out, idx_t)
        # (nAtoms, num_targets), (nEdges, num_targets)

        for i in range(self.num_blocks):
            # Interaction block
            h, m = self.int_blocks[i](
                h=h,
                m=m,
                rbf3=rbf3,
                cbf3=cbf3,
                id3_ragged_idx=id3_ragged_idx,
                id_swap=id_swap,
                id3_ba=id3_ba,
                id3_ca=id3_ca,
                rbf_h=rbf_h,
                idx_s=idx_s,
                idx_t=idx_t,
            )  # (nAtoms, emb_size_atom), (nEdges, emb_size_edge)

            E, F = self.out_blocks[i + 1](h, m, rbf_out, idx_t)
            # (nAtoms, num_targets), (nEdges, num_targets)
            F_st += F
            E_t += E

        return h, m
        # No need to do any of the force and energry calculations after this

        nMolecules = torch.max(batch) + 1
        if self.extensive:
            E_t = scatter(
                E_t, batch, dim=0, dim_size=nMolecules, reduce="add"
            )  # (nMolecules, num_targets)
        else:
            E_t = scatter(
                E_t, batch, dim=0, dim_size=nMolecules, reduce="mean"
            )  # (nMolecules, num_targets)

        if self.regress_forces:
            if self.direct_forces:
                # map forces in edge directions
                F_st_vec = F_st[:, :, None] * V_st[:, None, :]
                # (nEdges, num_targets, 3)
                F_t = scatter(
                    F_st_vec,
                    idx_t,
                    dim=0,
                    dim_size=data.atomic_numbers.size(0),
                    reduce="add",
                )  # (nAtoms, num_targets, 3)
                F_t = F_t.squeeze(1)  # (nAtoms, 3)
            else:
                if self.num_targets > 1:
                    forces = []
                    for i in range(self.num_targets):
                        # maybe this can be solved differently
                        forces += [
                            -torch.autograd.grad(
                                E_t[:, i].sum(), pos, create_graph=True
                            )[0]
                        ]
                    F_t = torch.stack(forces, dim=1)
                    # (nAtoms, num_targets, 3)
                else:
                    F_t = -torch.autograd.grad(E_t.sum(), pos, create_graph=True)[0]
                    # (nAtoms, 3)

            return E_t, F_t  # (nMolecules, num_targets), (nAtoms, 3)
        else:
            return E_t


if __name__ == "__main__":
    dgem = PosDescriptorGemNetT(
        "/home/jovyan/shared-scratch/joe/optim_cleaned_checkpoints/gemnet_t_direct_h512_all.pt"
    )
    from ase.io import Trajectory

    traj = Trajectory(
        "/home/jovyan/joe-job-vol/6_true_30_randoms/0_ft_unfrz_diff_blks/0_50_copy49_2g_dynunc25_kstep1000/online440438_0_50_copy49_2g_dynunc25_kstep1000_0/ft_en_C2H8In32N2Zr16_oal.traj"
    )
    pdesc = dgem.get_positional_descriptor(traj[0])
    print(pdesc)
