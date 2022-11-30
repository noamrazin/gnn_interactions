import os
import random
from collections import Counter

import numpy as np
import torch
from PIL import Image

from . import logging as logging_utils
from ..data.datasets.image.field_mapper import FieldMapper
from ..data.datasets.image.image_dataset import SubsetImageDataset


class ImageFolderDatasetCreator:
    """
    Used to create a dataset of ImageFolder format. Meaning for labeled images each image will be placed under a directory with the name of its
    label.
    """

    def __init__(self, output_dir, image_loader, label_name_extractor, image_extension="jpg"):
        self.output_dir = output_dir
        self.image_loader = image_loader
        self.label_name_extractor = label_name_extractor
        self.image_extension = image_extension
        self.label_image_counters = Counter()
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def add(self, datapoint_metadatas):
        for metadata in datapoint_metadatas:
            image = self.image_loader(metadata)
            label_name = self.label_name_extractor(metadata)
            self.label_image_counters[label_name] += 1

            label_dir = os.path.join(self.output_dir, label_name)
            if not os.path.exists(label_dir):
                os.mkdir(label_dir)

            image_path = os.path.join(label_dir,
                                      f"{label_name}{self.label_image_counters[label_name]}.{self.image_extension}")
            image.save(image_path)


class CropBoundingBoxTransform:
    """
    Crops an image according to a bounding box in its metadata.
    """

    def __init__(self, bbox_field_name="bbox"):
        self.bbox_field_name = bbox_field_name

    def __call__(self, image, metadata):
        bbox = metadata[self.bbox_field_name]
        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]
        return image.crop((x1, y1, x2, y2))


class TensorRandomHorizontalFlipTransform:

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_tensor):
        """
        Flips img in tensor format of CxHxW horizontally with probability p
        :param img_tensor: tensor image in CxHxW format.
        :return: with probability p a flipped horizontally tensor image and with probability 1-p the original input.
        """
        if random.random() < self.p:
            return torch.flip(img_tensor, dims=(2,))
        return img_tensor


def load_images_from_dir(dir_path, transform=lambda x: x):
    for path, subdirs, files in sorted(os.walk(dir_path)):
        for file_name in sorted(files):
            try:
                image = Image.open(os.path.join(path, file_name))
                yield transform(image)
            except IOError:
                logging_utils.error(f"Failed to load file '{file_name}' as an image")


def create_query_gallery_dataset_split(image_dataset, num_in_query_pred_id=1, split_field="id"):
    id_options_mapper = FieldMapper(image_dataset.images_metadata, split_field)
    query_indices = []
    gallery_indices = []

    for id_indices in id_options_mapper.by_field_indices():
        replace = num_in_query_pred_id > len(id_indices)
        for_query_indices = np.random.choice(id_indices, num_in_query_pred_id, replace=replace)
        query_indices.extend(set(for_query_indices.tolist()))
        gallery_indices.extend([i for i in id_indices if i not in for_query_indices])

    return SubsetImageDataset(image_dataset, query_indices), data.SubsetImageDataset(image_dataset, gallery_indices)
