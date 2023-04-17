import sys

import numpy as np
import torch_geometric

DATA_DOWNLOAD_FOLDER = './data'


def main(dataset_name, output_path):
    if dataset_name == 'cora':
        dataset = torch_geometric.datasets.Planetoid(DATA_DOWNLOAD_FOLDER, dataset_name, split='public')
    elif dataset_name == 'dblp':
        dataset = torch_geometric.datasets.CitationFull(DATA_DOWNLOAD_FOLDER, dataset_name)
    elif dataset_name in ['chameleon', 'squirrel']:
        dataset = torch_geometric.datasets.WikipediaNetwork(DATA_DOWNLOAD_FOLDER, dataset_name)
    elif dataset_name == 'computers':
        dataset = torch_geometric.datasets.Amazon(DATA_DOWNLOAD_FOLDER, dataset_name)
    elif dataset_name == 'ogbn-arxiv':
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(dataset_name, DATA_DOWNLOAD_FOLDER)
    else:
        raise ValueError("Unknown dataset")

    edges = dataset.data.edge_index.T.numpy() + 1

    np.savetxt(output_path, edges, delimiter=',', fmt='%d')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
