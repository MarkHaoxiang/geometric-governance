import torch
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Module
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_scatter import scatter_log_softmax


class MessagePassingElectionLayer(MessagePassing):
    def __init__(self, edge_dim: int = 1, emb_dim: int = 32):
        super().__init__(aggr="add")
        self.mlp_msg = Sequential(
            Linear(2 * emb_dim + edge_dim, emb_dim),
            BatchNorm1d(emb_dim),
            ReLU(),
            Linear(emb_dim, emb_dim),
            BatchNorm1d(emb_dim),
            ReLU(),
        )

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_i, x_j, edge_attr):
        msg = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp_msg(msg)


class MessagePassingElectionModel(Module):
    def __init__(
        self,
        edge_dim: int = 1,
        emb_dim: int = 32,
        num_layers: int = 4,
    ):
        super().__init__()
        in_dim = 2
        # Initial embedding
        self.lin_in = Linear(in_dim, emb_dim)
        # Convolution layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(MessagePassingElectionLayer(edge_dim, emb_dim))
        # Readout
        self.lin_out = Linear(emb_dim, 1)

    def forward(self, data: Data):
        x = self.lin_in(data.x.to(torch.float32))
        for conv in self.convs:
            x = x + conv(x, data.edge_index, data.edge_attr)
        logits = self.lin_out(x[data.candidate_idxs]).squeeze(dim=-1)
        out = scatter_log_softmax(logits, data.batch[data.candidate_idxs])
        return out
