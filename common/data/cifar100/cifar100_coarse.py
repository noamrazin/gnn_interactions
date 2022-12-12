from typing import Optional, Callable

import torch
from torchvision.datasets import CIFAR100


class CIFAR100Coarse(CIFAR100):
    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
                 download: bool = False, use_coarse_labels: bool = True):
        super(CIFAR100Coarse, self).__init__(root, train, transform, target_transform, download)

        self.use_coarse_labels = use_coarse_labels
        self.coarse_labels = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                              ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                              ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                              ['bottle', 'bowl', 'can', 'cup', 'plate'],
                              ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                              ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                              ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                              ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                              ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                              ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                              ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                              ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                              ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                              ['crab', 'lobster', 'snail', 'spider', 'worm'],
                              ['baby', 'boy', 'girl', 'man', 'woman'],
                              ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                              ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                              ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                              ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                              ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]

        self.fine_label_to_coarse_target = {}
        for i, label_list in enumerate(self.coarse_labels):
            for label in label_list:
                self.fine_label_to_coarse_target[label] = i

        self.fine_target_to_coarse_target = torch.zeros(len(self.class_to_idx), dtype=torch.int)
        for label, target in self.class_to_idx.items():
            self.fine_target_to_coarse_target[target] = self.fine_label_to_coarse_target[label]

        if self.use_coarse_labels:
            # update targets and classes fields
            self.targets = self.fine_target_to_coarse_target[self.targets].tolist()
            self.classes = self.coarse_labels
