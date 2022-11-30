import torch_geometric.loader as loader

from common.data.modules.datamodule import DataModule
from is_same_class.datasets.is_same_class_data import IsSameClassData


class IsSameClassDataModule(DataModule):

    def __init__(self, dataset_path: str, batch_size: int, partition_type: str, num_train_samples: int = -1,
                 dataloader_num_workers: int = 0, pin_memory: bool = False, load_dataset_to_device=None):
        """
        @param dataset_path: path to IsSameDataset file
        @param batch_size: batch size, if < 0 will use the size of the whole dataset
        @param partition_type: Determines how to arrange image patch vertices in the clusters. Supports "low_walk" and "high_walk".
        In "low_walk", the vertices of the images will be in separate cliques, and in "high_walk" each clique contains half of the vertices from
        each image.
        @param num_train_samples: number of training samples to use, if < 0 will use the whole training set
        @param dataloader_num_workers: num workers to use for data loading. When using multiprocessing, must use 0
        @param pin_memory: pin_memory argument of Dataloader
        @param load_dataset_to_device: device to load dataset to (default is CPU)
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.partition_type = partition_type
        self.num_train_samples = num_train_samples
        self.dataloader_num_workers = dataloader_num_workers
        self.pin_memory = pin_memory
        self.load_dataset_to_device = load_dataset_to_device

        self.dataset = IsSameClassData.load(dataset_path, partition_type=self.partition_type)

        if self.num_train_samples > 0:
            self.dataset.train_data_list = self.dataset.train_data_list[: min(self.num_train_samples, len(self.dataset.train_data_list))]

        if self.load_dataset_to_device is not None:
            self.dataset.train_data_list = [data.to(self.load_dataset_to_device) for data in self.dataset.train_data_list]
            self.dataset.test_data_list = [data.to(self.load_dataset_to_device) for data in self.dataset.test_data_list]

    def setup(self):
        pass

    def train_dataloader(self):
        batch_size = self.batch_size if self.batch_size > 0 else len(self.dataset.train_data_list)
        return loader.DataLoader(self.dataset.train_data_list, batch_size=batch_size, shuffle=True, pin_memory=self.pin_memory,
                                 num_workers=self.dataloader_num_workers)

    def val_dataloader(self):
        return self.test_dataloader()

    def test_dataloader(self):
        batch_size = self.batch_size if self.batch_size > 0 else len(self.dataset.test_data_list)
        return loader.DataLoader(self.dataset.test_data_list, batch_size=batch_size, shuffle=False, pin_memory=self.pin_memory,
                                 num_workers=self.dataloader_num_workers)
