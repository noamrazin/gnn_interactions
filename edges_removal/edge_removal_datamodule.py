import json

import numpy as np
import torch
import torch_geometric
import torch_geometric.loader as loader

from common.data.modules.datamodule import DataModule

DATA_DOWNLOAD_FOLDER = './data'


class EdgeRemovalDataModule(DataModule):

    def __init__(self, dataset_name: str, train_fraction: float, val_fraction: float, batch_size: int, dataloader_num_workers: int = 0,
                 load_dataset_to_device=None, train_frac_seed: float = 0, edges_ratio: float = 1, edges_remove_conf_file: str = ''):
        self.train_fraction = train_fraction
        self.batch_size = batch_size
        self.dataloader_num_workers = dataloader_num_workers
        self.load_dataset_to_device = load_dataset_to_device
        self.pin_memory = self.load_dataset_to_device is None
        self.dataset_name = dataset_name

        if dataset_name in ['cora', 'citeseer', 'pubmed']:
            self.train_dataset = torch_geometric.datasets.Planetoid(DATA_DOWNLOAD_FOLDER, dataset_name, split='public')
        elif dataset_name in ['dblp', 'cora_ml']:
            self.train_dataset = torch_geometric.datasets.CitationFull(DATA_DOWNLOAD_FOLDER, dataset_name)
        elif dataset_name in ['ogbn-arxiv']:
            import sklearn
            from ogb.nodeproppred import PygNodePropPredDataset
            self.train_dataset = PygNodePropPredDataset(dataset_name, DATA_DOWNLOAD_FOLDER)
        else:
            raise ValueError("Unknown dataset")
        self.num_vertices = self.train_dataset.data.x.shape[0]

        if train_fraction != -1:
            self.train_dataset.data['train_mask'], self.train_dataset.data['val_mask'], self.train_dataset.data['test_mask'] = self.generate_train_mask(train_frac_seed, train_fraction, val_fraction, self.num_vertices)
        elif dataset_name == 'ogbn-arxiv':
            self.train_dataset.data['train_mask'], self.train_dataset.data['val_mask'], self.train_dataset.data['test_mask'] = self.generate_train_mask_by_year(2017, 2018, self.train_dataset.data.node_year)


        self.rearrange_edges(edges_ratio, edges_remove_conf_file, device=load_dataset_to_device)

        data = self.train_dataset.data
        self.dim0 = data.x.shape[1]
        self.out_dim = max(data.y).item() + 1
        self.num_edges = self.train_dataset.data.edge_stores[0]['edge_index'].shape[-1]

        self.val_data = self.train_dataset.data.clone()
        self.test_data = self.train_dataset.data.clone()

        self.train_dataset.data.y[torch.logical_not(self.train_dataset.data.train_mask)] = -1
        self.val_data.y[torch.logical_not(self.val_data.val_mask)] = -1
        self.test_data.y[torch.logical_not(self.test_data.test_mask)] = -1

        self.train_dataset.data.y = self.train_dataset.data.y.flatten()
        self.val_data.y = self.val_data.y.flatten()
        self.test_data.y = self.test_data.y.flatten()

        if self.load_dataset_to_device is not None:
            self.train_dataset.data = self.train_dataset.data.to(self.load_dataset_to_device)
            self.val_data = self.val_data.to(self.load_dataset_to_device)
            self.test_data = self.test_data.to(self.load_dataset_to_device)

    def setup(self):
        pass

    def rearrange_edges(self, edges_ratio, edges_remove_conf_file, device=None):
        edges = self.train_dataset.data.edge_stores[0]['edge_index']
        edges = torch_geometric.utils.remove_self_loops(edges)[0]
        edges = torch_geometric.utils.to_undirected(edges)

        self.undirected_edges = torch.unique(torch.sort(edges, axis=0)[0], dim=1)
        # reorder all edges such that [all_edges_one_directions, all_edges_on_the_other_direction]
        edges = torch.concat([self.undirected_edges,
                              torch.stack([self.undirected_edges[1], self.undirected_edges[0]])], axis=1)

        self.train_dataset.data.edge_stores[0]['edge_index'] = edges
        self.base_num_edges = edges.shape[-1]

        self.removed_edges = [[], []]
        if edges_ratio == 0:
            self.train_dataset.data.edge_stores[0]['edge_index'] = torch.tensor([[], []], dtype=torch.long)
        elif edges_ratio < 1:
            num_edges = self.train_dataset.data.edge_stores[0]['edge_index'].shape[-1] // 2  # undirected
            new_edges_num = int(num_edges * edges_ratio)
            edges_to_remove = num_edges - new_edges_num
            with open(edges_remove_conf_file, 'r') as f:
                remove_conf = json.load(f)
            all_edges = self.train_dataset.data.edge_stores[0]['edge_index']
            self.removed_edges = [remove_conf['removed_edges'][0][:edges_to_remove], remove_conf['removed_edges'][1][:edges_to_remove]]

            removed_edges = torch.Tensor(remove_conf['removed_edges'])[:, :edges_to_remove]
            removed_edges = torch.concat([removed_edges, torch.stack([removed_edges[1], removed_edges[0]])],
                                         axis=1)
            removed_edges = removed_edges.cuda(device)
            all_edges_gpu = all_edges.cuda(device)
            mask_to_remove = torch.zeros(all_edges.shape[1], dtype=bool).cuda(device)
            for i in range(removed_edges.shape[1]):
                removed_edge_i_mask = torch.all(removed_edges[:,i:i+1] == all_edges_gpu, axis=0)
                mask_to_remove = torch.logical_or(mask_to_remove, removed_edge_i_mask)
            mask_to_remove = mask_to_remove.cpu()

            left_edges = all_edges[:, ~mask_to_remove]

            self.train_dataset.data.edge_stores[0]['edge_index'] = left_edges

        self.undirected_edges = torch.unique(torch.sort(self.train_dataset.data.edge_stores[0]['edge_index'], axis=0)[0], dim=1)
        # reorder all edges such that [all_edges_one_directions, all_edges_on_the_other_direction]
        if self.undirected_edges.shape[0] != 0:
            self.train_dataset.data.edge_stores[0]['edge_index'] = torch.concat([self.undirected_edges,
                                                                                 torch.stack([self.undirected_edges[1], self.undirected_edges[0]])],
                                                                                axis=1)

    def generate_train_mask_by_year(self, train_max_year, val_max_year, years):
        train_mask = years <= train_max_year
        val_mask = torch.logical_and(years <= val_max_year, years > train_max_year)
        test_mask = years > val_max_year
        return train_mask, val_mask, test_mask

    def generate_train_mask(self, train_frac_seed, train_fraction, val_fraction, num_vertices):
        train_num = int(train_fraction * num_vertices)
        val_num = int(val_fraction * num_vertices)
        vertices_indices = np.arange(num_vertices)

        random_state = np.random.get_state()
        np.random.seed(train_frac_seed)
        np.random.shuffle(vertices_indices)
        np.random.set_state(random_state)

        train_indices = vertices_indices[:train_num]
        val_indices = vertices_indices[train_num: train_num + val_num]
        test_indices = vertices_indices[train_num + val_num:]

        train_mask = torch.zeros(num_vertices, dtype=bool)
        train_mask[train_indices] = True
        val_mask = torch.zeros(num_vertices, dtype=bool)
        val_mask[val_indices] = True
        test_mask = torch.zeros(num_vertices, dtype=bool)
        test_mask[test_indices] = True

        return train_mask, val_mask, test_mask

    def train_dataloader(self):
        batch_size = self.batch_size if self.batch_size >= 0 else len(self.train_dataset.data.x)
        return loader.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, pin_memory=self.pin_memory,
                                 num_workers=self.dataloader_num_workers)

    def val_dataloader(self):
        batch_size = self.batch_size if self.batch_size >= 0 else len(self.val_data.x)
        return loader.DataLoader([self.val_data], batch_size=batch_size, shuffle=False, pin_memory=self.pin_memory,
                                 num_workers=self.dataloader_num_workers)

    def test_dataloader(self):
        batch_size = self.batch_size if self.batch_size >= 0 else len(self.test_data.x)
        return loader.DataLoader([self.test_data], batch_size=batch_size, shuffle=False, pin_memory=self.pin_memory,
                                 num_workers=self.dataloader_num_workers)
