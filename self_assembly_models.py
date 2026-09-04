# model.py (替换整个文件 / 或按块替换)
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RBF(nn.Module):
    def __init__(self, vmin=0.0, vmax=10.0, n_centers=32):
        super().__init__()
        self.n_centers = n_centers
        centers = torch.linspace(vmin, vmax, n_centers)
        self.register_buffer("centers", centers)
        # keep gamma as float
        self.gamma = float((centers[1]-centers[0]).item()**-2)

    def forward(self, d):
        # d: [E]
        if d.numel() == 0:
            return torch.zeros((0, self.n_centers), device=d.device)
        d = d.unsqueeze(1)  # [E,1]
        c = self.centers.unsqueeze(0)  # [1,n_centers]
        diff = d - c
        return torch.exp(-self.gamma * diff * diff)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_emb_dim):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim*2 + edge_emb_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr):
        """
        x: [N, in_dim]
        edge_index: [2, E]
        edge_attr: [E, edge_emb_dim]
        returns updated x: [N, out_dim]
        """
        if edge_index is None or edge_index.numel() == 0:
            # no edges: return projected x (match out_dim)
            # simple linear to match dims
            return self.update_mlp(torch.cat([x, torch.zeros((x.size(0), self.msg_mlp[-1].out_features), device=x.device)], dim=1))

        row, col = edge_index[0], edge_index[1]  # both [E]
        h_col = x[col]  # [E, in_dim]
        h_row = x[row]  # [E, in_dim]
        msg_in = torch.cat([h_row, h_col, edge_attr], dim=1)  # [E, 2*in_dim + edge_dim]
        m = self.msg_mlp(msg_in)  # [E, out_dim]
        # aggregate messages per node (sum)
        N = x.size(0)
        aggr = torch.zeros((N, m.size(1)), device=x.device)
        aggr = aggr.index_add(0, row, m)
        # update: concat x and aggr
        new = self.update_mlp(torch.cat([x, aggr], dim=1))
        return new


class MolEncoder(nn.Module):
    def __init__(self,
                 node_feat_dim,
                 fp_dim,
                 hidden_dim=128,
                 n_layers=6,
                 rbf_dim=32,
                 dropout=0.1,
                 use_layernorm=True
                 ):
        super().__init__()
        self.node_proj = nn.Linear(node_feat_dim, hidden_dim)
        self.rbf = RBF(n_centers=rbf_dim)
        self.gcn_layers = nn.ModuleList([
            GCNLayer(hidden_dim if i == 0 else hidden_dim, hidden_dim, rbf_dim)
            for i in range(n_layers)
        ])

        self.use_layernorm = use_layernorm
        if self.use_layernorm:
            self.layernorm = nn.LayerNorm(hidden_dim)

        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # fuse: concat (hidden, fp_emb) -> project back to hidden_dim
        self.fuse_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, atom_feats, edge_index, edge_dist, ptr, fp):
        device = atom_feats.device
        B, Nmax, feat_dim = atom_feats.size()
        Ntot = int(ptr[-1].item())

        flat_feats = atom_feats.reshape(B * Nmax, feat_dim)[:Ntot].to(device)

        x = self.node_proj(flat_feats)  # [Ntot, hidden_dim]
        x = torch.relu(x)

        if edge_index is not None and edge_index.size(1) > 0:
            edge_attr = self.rbf(edge_dist.to(device))  # [E, rbf_dim]
        else:
            edge_attr = torch.zeros((0, self.rbf.n_centers), device=device)

        for gcn in self.gcn_layers:
            x = gcn(x, edge_index.to(device) if edge_index is not None else edge_index, edge_attr)

        # pool per graph
        hidden_list = []
        for i in range(len(ptr) - 1):
            s = int(ptr[i].item())
            e = int(ptr[i + 1].item())
            if e - s > 0:
                h = x[s:e].mean(dim=0)
            else:
                h = torch.zeros((x.size(1),), device=device)
            hidden_list.append(h)
        hidden = torch.stack(hidden_list, dim=0)  # [B, hidden_dim]

        if self.use_layernorm:
            hidden = self.layernorm(hidden)

        hidden = self.pool_proj(hidden)  # [B, hidden_dim]
        fp_emb = self.fp_encoder(fp.to(device))  # [B, hidden_dim]

        mol_emb = torch.cat([hidden, fp_emb], dim=1)  # [B, 2*hidden]
        mol_emb = self.fuse_proj(mol_emb)             # [B, hidden_dim]

        return mol_emb  # output dim == hidden_dim


class PairClassifier(nn.Module):
    def __init__(self, node_feat_dim, fp_dim, hidden_dim=128, n_layers=6, rbf_dim=32, dropout=0.1, use_layernorm=True):
        super().__init__()
        self.encoder = MolEncoder(node_feat_dim, fp_dim, hidden_dim, n_layers, rbf_dim, dropout, use_layernorm)
        M = hidden_dim   # <<< 修正：encoder 返回 hidden_dim
        # interaction: concat [a, b, |a-b|, a*b]
        in_dim = M * 4
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, M),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(M, max(M//2, 16)),
            nn.ReLU(),
            nn.Linear(max(M//2, 16), 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, batch):
        a_emb = self.encoder(batch['a_atom_feats'], batch.get('a_edge_index', None), batch.get('a_edge_dist', None), batch['a_ptr'], batch['a_fp'])
        b_emb = self.encoder(batch['b_atom_feats'], batch.get('b_edge_index', None), batch.get('b_edge_dist', None), batch['b_ptr'], batch['b_fp'])
        diff = torch.abs(a_emb - b_emb)
        prod = a_emb * b_emb
        cat = torch.cat([a_emb, b_emb, diff, prod], dim=1)  # [B, 4*M]
        logits = self.classifier(cat).squeeze(1)            # [B]  (raw logits)
        return logits  # 注意：返回 logits（未经过 sigmoid）
