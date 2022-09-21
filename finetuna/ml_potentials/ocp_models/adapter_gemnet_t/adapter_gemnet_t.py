from ocpmodels.models.gemnet.gemnet import GemNetT

from typing import Optional

import numpy as np
import torch
from torch_geometric.nn import radius_graph
from torch_scatter import scatter
from torch_sparse import SparseTensor

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    compute_neighbors,
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)

from ocpmodels.models.gemnet.layers.atom_update_block import OutputBlock
from ocpmodels.models.gemnet.layers.base_layers import Dense
from ocpmodels.models.gemnet.layers.efficient import EfficientInteractionDownProjection
from ocpmodels.models.gemnet.layers.embedding_block import AtomEmbedding, EdgeEmbedding
from ocpmodels.models.gemnet.layers.interaction_block import (
    InteractionBlockTripletsOnly,
)
from ocpmodels.models.gemnet.layers.radial_basis import RadialBasis

# from ocpmodels.models.gemnet.layers.scaling import AutomaticFit
from ocpmodels.models.gemnet.layers.spherical_basis import CircularBasisLayer
from ocpmodels.models.gemnet.utils import (
    inner_product_normalized,
    mask_neighbors,
    ragged_range,
    repeat_blocks,
)
from torch.nn.init import xavier_uniform_


@registry.register_model("adapter_gemnet_t")
class AdapterGemNetT(GemNetT):
    def __init__(
        self,
        num_atoms: Optional[int],
        bond_feat_dim: int,
        num_targets: int,
        num_spherical: int,
        num_radial: int,
        num_blocks: int,
        emb_size_atom: int,
        emb_size_edge: int,
        emb_size_trip: int,
        emb_size_rbf: int,
        emb_size_cbf: int,
        emb_size_bil_trip: int,
        num_before_skip: int,
        num_after_skip: int,
        num_concat: int,
        num_atom: int,
        regress_forces: bool = True,
        direct_forces: bool = False,
        cutoff: float = 6,
        max_neighbors: int = 50,
        rbf: dict = ...,
        envelope: dict = ...,
        cbf: dict = ...,
        extensive: bool = True,
        otf_graph: bool = False,
        use_pbc: bool = True,
        output_init: str = "HeOrthogonal",
        activation: str = "swish",
        scale_file: Optional[str] = None,
        adapter_dim: int = 32,
        adapter_activation: str = "swish",
        adapter_initializer_gain: float = 1e-3,
    ):
        super().__init__(
            num_atoms,
            bond_feat_dim,
            num_targets,
            num_spherical,
            num_radial,
            num_blocks,
            emb_size_atom,
            emb_size_edge,
            emb_size_trip,
            emb_size_rbf,
            emb_size_cbf,
            emb_size_bil_trip,
            num_before_skip,
            num_after_skip,
            num_concat,
            num_atom,
            regress_forces,
            direct_forces,
            cutoff,
            max_neighbors,
            rbf,
            envelope,
            cbf,
            extensive,
            otf_graph,
            use_pbc,
            output_init,
            activation,
            scale_file,
        )

        ada_blocks = []

        # Adapter Blocks
        for i in range(num_blocks):
            ada_blocks.append(
                AdapterBlock(
                    emb_size_atom=emb_size_atom,
                    emb_size_edge=emb_size_edge,
                    adapter_dim=adapter_dim,
                    adapter_activation=adapter_activation,
                    adapter_initializer_gain=adapter_initializer_gain,
                )
            )

        self.ada_blocks = torch.nn.ModuleList(ada_blocks)

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

            h_a, m_a = self.ada_blocks[i](h=h, m=m)

            E, F = self.out_blocks[i + 1](h_a, m_a, rbf_out, idx_t)
            # (nAtoms, num_targets), (nEdges, num_targets)
            F_st += F
            E_t += E

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


class AdapterBlock(torch.nn.Module):
    def __init__(
        self,
        emb_size_atom: int,
        emb_size_edge: int,
        adapter_dim: int,
        adapter_activation: str,
        adapter_initializer_gain: float,
    ):
        super().__init__()
        ## ------------------------------------------- Feedforward down-project ------------------------------------------ ##
        self.down_project_e = DenseNearZero(
            in_features=emb_size_atom,
            out_features=adapter_dim,
            bias=True,
            activation=adapter_activation,
            initializer_gain=adapter_initializer_gain,
        )

        self.down_project_f = DenseNearZero(
            in_features=emb_size_edge,
            out_features=adapter_dim,
            bias=True,
            activation=adapter_activation,
            initializer_gain=adapter_initializer_gain,
        )

        ## -------------------------------------------- Feedforward up-project ------------------------------------------- ##
        self.up_project_e = DenseNearZero(
            in_features=adapter_dim,
            out_features=emb_size_atom,
            bias=True,
            activation=None,
            initializer_gain=adapter_initializer_gain,
        )

        self.up_project_f = DenseNearZero(
            in_features=adapter_dim,
            out_features=emb_size_edge,
            bias=True,
            activation=None,
            initializer_gain=adapter_initializer_gain,
        )

    def forward(self, h, m):
        # -------------------------------------- Energy (h) -------------------------------------- #
        h_a = self.down_project_e(h)
        h_a = self.up_project_e(h_a)
        # skip connection
        h_a += h
        # -------------------------------------- Forces (m) -------------------------------------- #
        m_a = self.down_project_f(m)
        m_a = self.up_project_f(m_a)
        # skip connection
        m_a += m
        return h_a, m_a


class DenseNearZero(Dense):
    def __init__(
        self,
        in_features,
        out_features,
        bias=False,
        activation=None,
        initializer_gain=1e-3,
    ):
        self.gain = initializer_gain
        super().__init__(in_features, out_features, bias, activation)

    def reset_parameters(self, initializer=None):
        initializer = xavier_uniform_
        initializer(self.linear.weight, gain=self.gain)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)
