import torch
import torch_geometric.utils
from torch import nn
from torch.nn import functional as F
import torch_scatter


class GraphModel(torch.nn.Module):
    def __init__(self, gnn_type, num_layers, dim0, h_dim, out_dim, layer_norm,
                 use_activation, graph_output: bool = False):
        super(GraphModel, self).__init__()
        self.gnn_type = gnn_type
        self.use_layer_norm = layer_norm
        self.use_activation = use_activation
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

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i in range(self.num_layers):
            layer = self.layers[i]

            x = layer(x, edge_index)
            if self.use_activation:
                x = F.relu(x)

            if self.use_layer_norm:
                x = self.layer_norms[i](x)

        if not self.graph_output:
            return self.out_layer(x)

        x = torch_scatter.scatter_mean(x, batch, dim=0)
        return self.out_layer(x)
