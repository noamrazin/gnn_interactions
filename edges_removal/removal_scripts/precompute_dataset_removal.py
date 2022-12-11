import os.path
import sys

sys.path.append(os.path.dirname(sys.argv[0]) + '/../..')

import argparse
import torch_geometric
import torch
import json
import common.utils.module as module_utils
import numpy as np
import sklearn
from edges_removal.walk_index_sparsification import WalkIndexSparsifier, EfficientOneWalkIndexSparsifier

OUTPUT_SUFFIX = '.json'
DATASETS_FOLDER = './data/'


def read_edges_and_num_vertices(dataset_name):
    if dataset_name == "cora":
        dataset = torch_geometric.datasets.Planetoid(DATASETS_FOLDER, "cora")
    elif dataset_name == "dblp":
        dataset = torch_geometric.datasets.CitationFull(DATASETS_FOLDER, "dblp")
    elif dataset_name == "ogbn-arxiv":
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(dataset_name, DATASETS_FOLDER)
    else:
        raise ValueError("Unknown dataset")

    dataset.data.edge_stores[0]['edge_index'] = torch_geometric.utils.remove_self_loops(dataset.data.edge_stores[0]['edge_index'])[0]
    dataset.data.edge_stores[0]['edge_index'] = torch_geometric.utils.to_undirected(dataset.data.edge_stores[0]['edge_index'])
    num_vertices = dataset.data.x.shape[0]
    edge_index = dataset.data.edge_stores[0]['edge_index']
    return edge_index, num_vertices


def find_undirected_edges_removal(num_vertices, edge_index, wis_chunk_size, gnn_depth, edge_prune_method, gpu_id):
    device = module_utils.get_device(cuda_id=gpu_id)

    if edge_prune_method == "wis":
        return find_edges_removal_by_wis(num_vertices, edge_index, wis_chunk_size, gnn_depth, device=device)
    elif edge_prune_method == "one_wis":
        return find_edges_removal_by_efficient_one_wis(num_vertices, edge_index)
    elif edge_prune_method == "random":
        return find_random_edges_removal(edge_index)
    elif edge_prune_method == "spectral":
        return find_spectral_edges_removal(num_vertices, edge_index)

    raise ValueError(f"Invalid prune method '{edge_prune_method}'")


def find_edges_removal_by_wis(num_vertices, edge_index, chunk_size, gnn_depth, device=torch.device("cpu")):
    """
    Remove edges according to (L-1)-WIS for L = gnn_depth.
    """
    wis = WalkIndexSparsifier(gnn_depth, chunk_size)
    num_undirected_edges = edge_index.shape[1] // 2
    _, removed_edges, _ = wis.sparsify(num_vertices, edge_index, num_edges_to_remove=num_undirected_edges,
                                       device=device, print_progress=True)
    return removed_edges.tolist()


def find_edges_removal_by_efficient_one_wis(num_vertices, edge_index):
    """
    Remove edges according to 1-WIS (efficient implementation).
    """
    one_wis = EfficientOneWalkIndexSparsifier()
    num_undirected_edges = edge_index.shape[1] // 2
    _, removed_edges, _ = one_wis.sparsify(num_vertices, edge_index, num_edges_to_remove=num_undirected_edges, print_progress=True)
    return removed_edges.tolist()


def find_random_edges_removal(edge_index):
    """
    Random edge pruning.
    """
    undirected_edges = set()
    for e in edge_index.T.tolist():
        if (e[1], e[0]) not in undirected_edges:
            undirected_edges.add((e[0], e[1]))

    undirected_edges = list(undirected_edges)
    np.random.shuffle(undirected_edges)
    undirected_edges = torch.tensor(undirected_edges, dtype=edge_index.dtype).T.tolist()
    return undirected_edges


def find_spectral_edges_removal(num_vertices: int, edge_index: torch.Tensor):
    """
    Removes edges according to the Graph Sparsification by Effective Resistances method (Spielman & Srivastava, 2011).
    """
    # Lazy load pygsp import to prevent the need for this dependency when not used
    import pygsp
    from edges_removal.spectral_sparsification import get_spectral_graph_sparsify_edge_removal_order

    sparse_adj_mat = torch_geometric.utils.to_scipy_sparse_matrix(edge_index, num_nodes=num_vertices)
    graph = pygsp.graphs.Graph(sparse_adj_mat)
    removed_edges = get_spectral_graph_sparsify_edge_removal_order(graph)
    return removed_edges.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True, help="Output path.")
    parser.add_argument("--gpu_id", type=int, default=-1, help="GPU id to use for wis. If -1 (default), run on CPU.")
    parser.add_argument("--dataset", default="cora", help="Dataset to run on. Supports 'cora' and 'dblp'")
    parser.add_argument("--edge_prune_method", type=str, default="random",
                        help="Edge sparsification method to use. Supports: 'wis', 'one_wis', 'random', and 'spectral'.")
    parser.add_argument("--gnn_depth", type=int, default=3, help="Depth of GNN according for which pruning is done.")
    parser.add_argument("--wis_chunk_size", type=int, default=100, help="Number of edges to remove per computation of removal criterion for wis.")
    args = parser.parse_args()

    output_path = args.output_path
    if not output_path.endswith(OUTPUT_SUFFIX):
        output_path += OUTPUT_SUFFIX

    print("============================================================")
    print(f"Starting to compute edge removal order for arguments:{args.__dict__}")

    edge_index, num_vertices = read_edges_and_num_vertices(args.dataset)
    removed_edges = find_undirected_edges_removal(num_vertices, edge_index, args.wis_chunk_size, args.gnn_depth, args.edge_prune_method,
                                                    args.gpu_id)

    output_data = {
        'dataset': args.dataset,
        'chunk_size': args.wis_chunk_size if args.edge_prune_method == "wis" else edge_index.shape[-1],
        'removed_edges': removed_edges
    }

    with open(args.output_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Created edge removal order file at: {args.output_path}")


if __name__ == '__main__':
    main()
