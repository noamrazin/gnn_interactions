import argparse
import os
import subprocess as sp


def run_random_pruning_algs(dataset, algs, repetitions, output_folder):
    if 'random' not in algs:
        return
    for i in range(repetitions):
        sp.run(["python", "./edges_removal/removal_scripts/precompute_dataset_removal.py",
                "--dataset", dataset,
                "--edge_prune_method", "random",
                "--output_path", f"{output_folder}/{dataset}_remove_by_random_{i}.json"
                ])


def run_spectral_pruning_algs(dataset, algs, repetitions, output_folder, julia_impl):
    if 'spectral' not in algs:
        return

    if not julia_impl:
        for i in range(repetitions):
            sp.run(["python", "./edges_removal/removal_scripts/precompute_dataset_removal.py",
                    "--dataset", dataset,
                    "--edge_prune_method", "spectral",
                    "--output_path", f"{output_folder}/{dataset}_remove_by_spectral_{i}.json"
                    ])
    else:
        csv_path = f"{output_folder}/{dataset}_edges.csv"
        sp.run(["python", "./edges_removal/removal_scripts/dataset_to_edges_csv.py",
                dataset, csv_path
                ])
        for i in range(repetitions):
            sp.run(["julia", "./edges_removal/removal_scripts/generate_laplacian_sparsification.jl",
                    csv_path, f"{output_folder}/{dataset}_remove_by_spectral_{i}.json"
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
    parser.add_argument("--julia_spectral", action="store_true", help="Use Julia implementation for spectral pruning. "
                                                                      "Otherwise uses Python implementation.")
    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    run_random_pruning_algs(args.dataset, args.algs, args.repetitions, args.output_folder)
    run_spectral_pruning_algs(args.dataset, args.algs, args.repetitions, args.output_folder, args.julia_spectral)
    run_deterministic_pruning_algs(args.dataset, args.algs, args.gnn_depth,
                                   args.wis_chunk_size, args.gpu_id, args.output_folder)


if __name__ == '__main__':
    main()
