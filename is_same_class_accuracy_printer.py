import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import numpy as np


def __load_summary_json(exp_dir_path: Path):
    summary_json_path = exp_dir_path.joinpath("summary.json")

    if not summary_json_path.exists():
        return None

    with open(summary_json_path) as f:
        return json.load(f)


def __load_config_json(exp_dir_path: Path):
    config_json_path = exp_dir_path.joinpath("config.json")

    if not config_json_path.exists():
        return None

    with open(config_json_path) as f:
        return json.load(f)


def print_mean_and_std(experiments_dir: str, metric_name: str, group_by: Tuple[str, ...]):
    experiments_dir_path = Path(experiments_dir)
    experiments_paths = [path for path in experiments_dir_path.iterdir() if path.is_dir()]

    results = defaultdict(list)
    for experiment_dir_path in experiments_paths:
        exp_summary = __load_summary_json(experiment_dir_path)
        if not exp_summary:
            continue

        exp_config = __load_config_json(experiment_dir_path)
        key = tuple([exp_config[config_name] for config_name in group_by])
        result = exp_summary["last_tracked_values"][metric_name]["value"]
        results[key].append(result)

    for key, values in results.items():
        np_values = np.array(values)
        print(f"{key}: mean {np_values.mean():.4f} , std {np_values.std():.4f}, runs {len(values)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiments_dir", type=str, help="Path to directory of experiments")
    args = p.parse_args()

    print("Training Accuracies")
    print("=========================================")
    print_mean_and_std(args.experiments_dir, "train accuracy", ("model", "partition_type", "lr"))
    print("\nTest Accuracies")
    print("=========================================")
    print_mean_and_std(args.experiments_dir, "test accuracy", ("model", "partition_type", "lr"))


if __name__ == "__main__":
    main()
