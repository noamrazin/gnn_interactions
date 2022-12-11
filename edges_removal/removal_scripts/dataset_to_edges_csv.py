import sys
import torch_geometric
import numpy as np

DATA_DOWNLOAD_FOLDER = './data'

def main(dataset_name, output_path):
    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        dataset = torch_geometric.datasets.Planetoid(DATA_DOWNLOAD_FOLDER, dataset_name, split='public')
    elif dataset_name in ['dblp', 'cora_ml']:
        dataset = torch_geometric.datasets.CitationFull(DATA_DOWNLOAD_FOLDER, dataset_name)
    elif dataset_name in ['ogbn-arxiv']:
        import sklearn
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(dataset_name, DATA_DOWNLOAD_FOLDER)
    else:
        raise ValueError("Unknown dataset")

    edges = dataset.data.edge_index.T.numpy() + 1
    
    np.savetxt(output_path, edges, delimiter=',', fmt='%d')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

