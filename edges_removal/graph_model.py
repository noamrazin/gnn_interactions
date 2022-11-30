import os
import torch
import torch_geometric.utils
from torch import nn
from torch.nn import functional as F
import torch_scatter
import edges_removal.ugs_utils as ugs_utils


class GraphModel(torch.nn.Module):
    def __init__(self, gnn_type, num_layers, dim0, h_dim, out_dim, num_edges, layer_norm,
            graph_output: bool = False, is_ugs_mask_train:bool=False,
            model_initialization_path=None):
        super(GraphModel, self).__init__()
        self.gnn_type = gnn_type
        self.use_layer_norm = layer_norm
        self.graph_output = graph_output

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        self.layers.append(gnn_type.get_layer(in_dim=dim0, out_dim=h_dim))
        self.layer_norms = nn.ModuleList()

        for i in range(1, num_layers):
            self.layers.append(gnn_type.get_layer(in_dim=h_dim, out_dim=h_dim))

        if self.use_layer_norm:
            for i in range(num_layers):
                self.layer_norms.append(nn.LayerNorm(h_dim))

        self.out_dim = out_dim
        self.out_layer = nn.Linear(in_features=h_dim, out_features=out_dim)

        # mask per undirected edge, assuming [all_edges, all_edges[::-1]]
        self.adj_mask_train = nn.Parameter(ugs_utils.mask_init(num_edges//2), requires_grad=True)
        self.is_adj_mask = is_ugs_mask_train

        if model_initialization_path:
            if os.path.exists(model_initialization_path):
                state_dict = torch.load(model_initialization_path)
                state_dict['adj_mask_train'] = self.adj_mask_train.data
                self.load_state_dict(state_dict)
            else:
                torch.save(self.state_dict(), model_initialization_path)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            layer = self.layers[i]

            if self.is_adj_mask:
                mask = torch.cat([self.adj_mask_train, self.adj_mask_train])
                x = layer(x, edge_index, mask)
            else:
                x = layer(x, edge_index)

            x = F.relu(x)

            if self.use_layer_norm:
                x = self.layer_norms[i](x)

        if not self.graph_output:
            return self.out_layer(x)

        x = torch_scatter.scatter_mean(x, batch, dim=0)
        return self.out_layer(x)

    def __get_fully_adjacent_layer_edges(self, batch):
        block_map = torch.eq(batch.unsqueeze(0), batch.unsqueeze(-1)).int()
        edges, _ = torch_geometric.utils.dense_to_sparse(block_map)
        return edges
