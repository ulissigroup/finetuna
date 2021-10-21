from ocpmodels.models.gemnet.utils import inner_product_normalized
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from ocpmodels.datasets.trajectory_lmdb import data_list_collater


class OCPDescriptor:
    """OCPDescriptor.
    This class serves as a helper class for calculators that want to generate descriptors from OCP models.

    Parameters
    ----------
    model_path: str
        path to gemnet model config, e.g. '/home/jovyan/working/ocp/configs/s2ef/all/gemnet/gemnet-dT.yml'

    checkpoint_path: str
        path to gemnet model checkpoint, e.g. '/home/jovyan/shared-datasets/OC20/checkpoints/s2ef/gemnet_t_direct_h512_all.pt'
    """

    def __init__(
        self,
        model_path: str,
        checkpoint_path: str,
    ) -> None:
        self.ocp_calc = OCPCalculator(
            config_yml=model_path,
            checkpoint=checkpoint_path,
        )

    def gemnet_forward(self, atoms):
        data_object = self.ocp_calc.a2g.convert(atoms)
        data = data_list_collater([data_object])
        model = self.ocp_calc.trainer.model.module

        pos = data.pos
        # batch = data.batch
        atomic_numbers = data.atomic_numbers.long()

        if model.regress_forces and not model.direct_forces:
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
        ) = model.generate_interaction_graph(data)
        idx_s, idx_t = edge_index

        # Calculate triplet angles
        cosφ_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        rad_cbf3, cbf3 = model.cbf_basis3(D_st, cosφ_cab, id3_ca)

        rbf = model.radial_basis(D_st)

        # Embedding block
        h = model.atom_emb(atomic_numbers)
        # (nAtoms, emb_size_atom)
        m = model.edge_emb(h, rbf, idx_s, idx_t)  # (nEdges, emb_size_edge)

        rbf3 = model.mlp_rbf3(rbf)
        cbf3 = model.mlp_cbf3(rad_cbf3, cbf3, id3_ca, id3_ragged_idx)

        rbf_h = model.mlp_rbf_h(rbf)
        rbf_out = model.mlp_rbf_out(rbf)

        E_t, F_st = model.out_blocks[0](h, m, rbf_out, idx_t)
        # (nAtoms, num_targets), (nEdges, num_targets)

        for i in range(model.num_blocks):
            # Interaction block
            h, m = model.int_blocks[i](
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

            E, F = model.out_blocks[i + 1](h, m, rbf_out, idx_t)
            # (nAtoms, num_targets), (nEdges, num_targets)
            F_st += F
            E_t += E

        return E_t, F_st
