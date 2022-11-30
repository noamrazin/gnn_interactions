import argparse
import json
import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.utils.data

import common.utils.logging as logging_utils
from common.data.modules.torchvision_datamodule import TorchvisionDataModule
from is_same_class.datasets.is_same_class_data import IsSameClassData


def __set_initial_random_seed(random_seed: int):
    if random_seed != -1:
        np.random.seed(random_seed)
        torch.random.manual_seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)


def __create_dataset_filename(dataset_name, num_patches, now_utc_str, num_train_samples, num_test_samples, ):
    filename = f"{dataset_name}_is_same_class_patches_{num_patches}_train_{num_train_samples}_" \
               f"test_{num_test_samples}_{now_utc_str}"
    filename = filename.replace(".", "-") + ".pt"
    return filename


def __sample_positive_image_pair_indices(per_relevant_label_image_indices: Dict[int, torch.Tensor]):
    label = np.random.randint(0, len(per_relevant_label_image_indices))
    current_label_image_indices = per_relevant_label_image_indices[label]

    first_image_sample = np.random.randint(0, len(current_label_image_indices))
    second_image_sample = np.random.randint(0, len(current_label_image_indices))
    return torch.stack([current_label_image_indices[first_image_sample],
                        current_label_image_indices[second_image_sample]], dim=0)


def __sample_negative_image_pair_indices(per_relevant_label_image_indices: Dict[int, torch.Tensor]):
    label = np.random.randint(0, len(per_relevant_label_image_indices))

    current_label_image_indices = per_relevant_label_image_indices[label]
    other_labels_image_indices = torch.cat([label_indices for l, label_indices in per_relevant_label_image_indices.items() if l != label])

    first_image_sample = np.random.randint(0, len(current_label_image_indices))
    first_image_index = current_label_image_indices[first_image_sample]

    second_image_sample = np.random.randint(0, len(other_labels_image_indices))
    second_image_index = other_labels_image_indices[second_image_sample]

    return torch.stack([first_image_index, second_image_index], dim=0)


def __create_image_pairs_indices_and_labels(original_dataset, num_samples):
    original_labels = original_dataset.targets.clone().detach()
    relevant_labels = original_labels.unique().tolist()
    per_relevant_label_image_indices = {label: torch.nonzero(original_labels == label).squeeze() for label in relevant_labels}

    pos_image_pairs_indices = []
    neg_image_pairs_indices = []
    for _ in range(num_samples):
        if np.random.rand() >= 0.5:
            pos_image_pair_indices = __sample_positive_image_pair_indices(per_relevant_label_image_indices)
            pos_image_pairs_indices.append(pos_image_pair_indices)
        else:
            neg_image_pair_indices = __sample_negative_image_pair_indices(per_relevant_label_image_indices)
            neg_image_pairs_indices.append(neg_image_pair_indices)

    pos_image_pairs_indices = torch.stack(pos_image_pairs_indices)
    neg_image_pairs_indices = torch.stack(neg_image_pairs_indices)
    image_pairs_indices = torch.cat([pos_image_pairs_indices, neg_image_pairs_indices])
    is_same_labels = torch.cat([torch.ones(pos_image_pairs_indices.shape[0], dtype=torch.float),
                                torch.zeros(neg_image_pairs_indices.shape[0], dtype=torch.float)])

    shuffle_perm = torch.randperm(len(image_pairs_indices))
    return image_pairs_indices[shuffle_perm], is_same_labels[shuffle_perm]


def __create_first_and_second_image_patches_list(dataset, image_pairs_indices, num_patches) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    first_image_patches_list = []
    second_image_patches_list = []

    for i in range(image_pairs_indices.shape[0]):
        first_image, _ = dataset[image_pairs_indices[i][0]]
        first_image = first_image.reshape(1, -1)
        first_image_patches_list.append(__get_patches(first_image, num_patches))

        second_image, _ = dataset[image_pairs_indices[i][1]]
        second_image = second_image.reshape(1, -1)
        second_image_patches_list.append(__get_patches(second_image, num_patches))

    return first_image_patches_list, second_image_patches_list


def __get_patches(flattened_image, num_patches):
    patches = []
    patch_size = flattened_image.shape[1] // num_patches
    for i in range(num_patches):
        patches.append(flattened_image[:, i * patch_size: (i + 1) * patch_size])

    return torch.cat(patches)


def create_and_save_dataset(args):
    now_utc_str = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")

    torchvision_datamodule = TorchvisionDataModule(dataset_name=args.dataset_name)
    torchvision_datamodule.setup()
    train_dataset = torchvision_datamodule.train_dataset

    train_image_pairs_indices, train_labels = __create_image_pairs_indices_and_labels(train_dataset, args.num_train_samples)
    train_first_image_patches_list, train_second_image_patches_list = __create_first_and_second_image_patches_list(train_dataset,
                                                                                                                   train_image_pairs_indices,
                                                                                                                   args.num_patches)

    test_dataset = torchvision_datamodule.test_dataset
    test_image_pairs_indices, test_labels = __create_image_pairs_indices_and_labels(test_dataset, args.num_test_samples)
    test_first_image_patches_list, test_second_image_patches_list = __create_first_and_second_image_patches_list(test_dataset,
                                                                                                                 test_image_pairs_indices,
                                                                                                                 args.num_patches)

    dataset = IsSameClassData(train_first_image_features_list=train_first_image_patches_list,
                              train_second_image_features_list=train_second_image_patches_list,
                              train_labels=train_labels,
                              test_first_image_features_list=test_first_image_patches_list,
                              test_second_image_features_list=test_second_image_patches_list,
                              test_labels=test_labels,
                              additional_metadata=args.__dict__)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    filename = __create_dataset_filename(args.dataset_name, args.num_patches, now_utc_str, args.num_train_samples, args.num_test_samples)
    output_path = os.path.join(args.output_dir, filename)
    dataset.save(output_path)

    logging_utils.info(f"Created dataset at: '{output_path}'\n"
                       f"Args: {json.dumps(args.__dict__, indent=2)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--random_seed", type=int, default=-1, help="Initial random seed")
    p.add_argument("--output_dir", type=str, default="data/gisc", help="Path to the directory to save the target matrix and dataset at")

    p.add_argument("--dataset_name", type=str, default="fmnist", help="Dataset to create a tensor completion dataset for. "
                                                                      "Currently supports: 'mnist', 'fmnist', 'cifar10', 'cifar100', 'svhn'.")
    p.add_argument("--num_patches", type=int, default=16, help="Number of non-overlapping patches to partition the images into. "
                                                               "Each patch has its own vertex, and patches of an image are a clique.")

    p.add_argument("--num_train_samples", type=int, default=10000, help="Number of training samples to generate")
    p.add_argument("--num_test_samples", type=int, default=2000, help="Number of test samples to generate")

    args = p.parse_args()

    logging_utils.init_console_logging()
    __set_initial_random_seed(args.random_seed)

    create_and_save_dataset(args)


if __name__ == "__main__":
    main()
