import logging
import os

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import edges_removal.ugs_utils as ugs_utils
from edges_removal.deeper_gcn.torch_nn import norm_layer
from edges_removal.deeper_gcn.torch_vertex import GENConv


class DeeperGCN(torch.nn.Module):

    def __init__(self, in_channels, num_tasks, num_edges, num_layers=3, dropout=0.5, block='res+', hidden_channels=128, conv='gen', gcn_aggr='max',
                 t=1.0, learn_t=False, p=1.0, learn_p=False, y=0.0, learn_y=False, msg_norm=False, learn_msg_scale=False, norm='batch', mlp_layers=1,
                 is_ugs_mask_train: bool = False, model_initialization_path=None):
        super(DeeperGCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.block = block

        self.checkpoint_grad = False

        # in_channels = in_channels
        # hidden_channels = hidden_channels
        # num_tasks = args.num_tasks
        # conv = args.conv
        aggr = gcn_aggr

        # t = args.t
        self.learn_t = learn_t
        # p = args.p
        self.learn_p = learn_p
        # y = args.y
        self.learn_y = learn_y

        self.msg_norm = msg_norm
        learn_msg_scale = learn_msg_scale

        norm = norm
        mlp_layers = mlp_layers

        if aggr in ['softmax_sg', 'softmax', 'power'] and self.num_layers > 7:
            self.checkpoint_grad = True
            self.ckp_k = self.num_layers // 2

        print('The number of layers {}'.format(self.num_layers),
              'Aggregation method {}'.format(aggr),
              'block: {}'.format(self.block))

        if self.block == 'res+':
            print('LN/BN->ReLU->GraphConv->Res')
        elif self.block == 'res':
            print('GraphConv->LN/BN->ReLU->Res')
        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')
        elif self.block == "plain":
            print('GraphConv->LN/BN->ReLU')
        else:
            raise Exception('Unknown block Type')

        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        self.node_features_encoder = torch.nn.Linear(in_channels, hidden_channels)
        self.node_pred_linear = torch.nn.Linear(hidden_channels, num_tasks)

        for layer in range(self.num_layers):

            if conv == 'gen':
                gcn = GENConv(hidden_channels, hidden_channels,
                              aggr=aggr,
                              t=t, learn_t=self.learn_t,
                              p=p, learn_p=self.learn_p,
                              y=y, learn_y=self.learn_y,
                              msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                              norm=norm, mlp_layers=mlp_layers)
            else:
                raise Exception('Unknown Conv Type')

            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

        self.is_adj_mask = is_ugs_mask_train
        self.adj_mask_train = torch.nn.Parameter(ugs_utils.mask_init(num_edges // 2), requires_grad=True)

        if model_initialization_path:
            if os.path.exists(model_initialization_path):
                state_dict = torch.load(model_initialization_path)
                state_dict['adj_mask_train'] = self.adj_mask_train.data
                self.load_state_dict(state_dict)
            else:
                torch.save(self.state_dict(), model_initialization_path)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h = self.node_features_encoder(x)

        mask = torch.cat([self.adj_mask_train, self.adj_mask_train])
        mask = mask if self.is_adj_mask else None

        if self.block == 'res+':

            h = self.gcns[0](h, edge_index, mask)

            if self.checkpoint_grad:

                for layer in range(1, self.num_layers):
                    h1 = self.norms[layer - 1](h)
                    h2 = F.relu(h1)
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)

                    if layer % self.ckp_k != 0:
                        res = checkpoint(self.gcns[layer], h2, edge_index, mask)
                        h = res + h
                    else:
                        h = self.gcns[layer](h2, edge_index, mask) + h

            else:
                for layer in range(1, self.num_layers):
                    h1 = self.norms[layer - 1](h)
                    h2 = F.relu(h1)
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)
                    h = self.gcns[layer](h2, edge_index, mask) + h

            h = F.relu(self.norms[self.num_layers - 1](h))
            h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'res':

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, mask)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, mask)
                h2 = self.norms[layer](h1)
                h = F.relu(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')

        elif self.block == 'plain':

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, mask)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, mask)
                h2 = self.norms[layer](h1)
                h = F.relu(h2)
                h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            raise Exception('Unknown block Type')

        h = self.node_pred_linear(h)

        return h  # torch.log_softmax(h, dim=-1)

    def print_params(self, epoch=None, final=False):

        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                print('Final t {}'.format(ts))
            else:
                logging.info('Epoch {}, t {}'.format(epoch, ts))

        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                print('Final p {}'.format(ps))
            else:
                logging.info('Epoch {}, p {}'.format(epoch, ps))

        if self.learn_y:
            ys = []
            for gcn in self.gcns:
                ys.append(gcn.sigmoid_y.item())
            if final:
                print('Final sigmoid(y) {}'.format(ys))
            else:
                logging.info('Epoch {}, sigmoid(y) {}'.format(epoch, ys))

        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                print('Final s {}'.format(ss))
            else:
                logging.info('Epoch {}, s {}'.format(epoch, ss))
