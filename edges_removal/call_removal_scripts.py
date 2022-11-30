import argparse
import os
import subprocess as sp


def run_random_pruning_algs(dataset, algs, repetitions, output_folder):
    for method in ['random', 'spectral']:
        if method not in algs:
            continue
        for i in range(repetitions):
            sp.run(["python", "./edges_removal/removal_scripts/precompute_dataset_removal.py",
                    "--dataset", dataset,
                    "--edge_prune_method", method,
                    "--output_path", f"{output_folder}/{dataset}_remove_by_{method}_{i}.json"
                    ])


def run_deterministic_pruning_algs(dataset, algs, gnn_depth, wis_chunk_size, gpu_id, output_folder):
    for method in ['wis', 'one_wis']:
        if method not in algs:
            continue
        sp.run(["python", "./edges_removal/removal_scripts/precompute_dataset_removal.py",
                "--dataset", dataset, "--gnn_depth", str(gnn_depth),
                "--edge_prune_method", method, "--wis_chunk_size", str(wis_chunk_size),
                "--gpu_id", str(gpu_id),
                "--output_path", f"{output_folder}/{dataset}_remove_by_{method}.json"
                ])


def main():
    parser = argparse.ArgumentParser(description="run the edge sparsification algorithms and save the results")
    parser.add_argument('--dataset', help="Dataset to compute removals for. Currently supports 'dblp' and 'cora'.")
    parser.add_argument('--output_folder', help="Output folder to save the edge removal orders in.")
    parser.add_argument('--repetitions', type=int, default=10, required=False, help="Number of repetitions for non-deterministic algorithms.")
    parser.add_argument("--gnn_depth", type=int, default=3, required=False, help="Depth L of a GNN for which (L - 1)-WIS is ran.")
    parser.add_argument("--wis_chunk_size", type=int, default=100, required=False,
                        help="Number of edges to remove per computation of removal criterion in WIS.")
    parser.add_argument("--gpu_id", type=int, default=-1, required=False, help="GPU id to use for wis")
    parser.add_argument('algs', nargs='+', metavar='algorithm',
                        help="List of pruning algorithms to run. Supports: 'random', 'spectral', 'wis' and 'one_wis'.")
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    run_random_pruning_algs(args.dataset, args.algs, args.repetitions, args.output_folder)
    run_deterministic_pruning_algs(args.dataset, args.algs, args.gnn_depth,
                                   args.wis_chunk_size, args.gpu_id, args.output_folder)


if __name__ == '__main__':
    main()
