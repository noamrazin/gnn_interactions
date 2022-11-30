from typing import List
import torch_geometric
import torch
import math
import itertools


class IsSameClassData:

    def __init__(self, train_first_image_features_list: List[torch.Tensor], train_second_image_features_list: List[torch.Tensor],
                 train_labels: torch.Tensor, test_first_image_features_list: List[torch.Tensor], test_second_image_features_list: List[torch.Tensor],
                 test_labels: torch.Tensor, partition_type: str = "low_walk", additional_metadata: dict = None):
        """
        @param train_first_image_features_list: List of tensors, each tensor is the features of vertices in the first cluster for a single graph
        @param train_second_image_features_list: List of tensors, each tensor is the features of vertices in the second cluster for a single graph
        @param train_labels: Training labels
        @param test_first_image_features_list: List of tensors, each tensor is the features of vertices in the first cluster for a single graph
        @param test_second_image_features_list: List of tensors, each tensor is the features of vertices in the second cluster for a single graph
        @param test_labels: Test labels
        @param partition_type: Determines how to distribute image patches among vertices. Supports "low_walk" and "high_walk".
        @param additional_metadata: additional metadata.
        """
        self.train_first_image_features_list = train_first_image_features_list
        self.train_second_image_features_list = train_second_image_features_list
        self.train_labels = train_labels
        self.test_first_image_features_list = test_first_image_features_list
        self.test_second_image_features_list = test_second_image_features_list
        self.test_labels = test_labels

        self.partition_type = partition_type

        cluster_size = self.train_first_image_features_list[0].shape[0]
        self.first_cluster_indices = list(range(cluster_size))
        self.second_cluster_indices = list(range(cluster_size, 2 * cluster_size))

        self.train_data_list = self.__create_data_list(train_first_image_features_list, train_second_image_features_list, train_labels)
        self.test_data_list = self.__create_data_list(test_first_image_features_list, test_second_image_features_list, test_labels)

        self.additional_metadata = additional_metadata
        self.in_dim = self.train_first_image_features_list[0].shape[-1]

    def __create_data_list(self, first_image_features_list, second_image_features_list, labels):
        data_list = []
        for i in range(len(first_image_features_list)):
            first_features = first_image_features_list[i]
            second_features = second_image_features_list[i]
            label = labels[i]

            x = self.__get_vertex_features_first_cluster_then_second_cluster(first_features, second_features)

            edges = [[], []]
            self.__populate_cluster_edges(edges, self.first_cluster_indices)
            self.__populate_cluster_edges(edges, self.second_cluster_indices)
            self.__add_between_clusters_edges(edges)

            edge_index = torch.tensor(edges, dtype=torch.long)
            edge_index = torch_geometric.utils.to_undirected(edge_index)
            edge_index, _ = torch_geometric.utils.add_remaining_self_loops(edge_index)

            data_list.append(torch_geometric.data.Data(x=x, edge_index=edge_index, y=label))

        return data_list

    def __get_vertex_features_first_cluster_then_second_cluster(self, first_image_features, second_image_features):
        if self.partition_type == "low_walk":
            return torch.cat([first_image_features, second_image_features])
        elif self.partition_type == "high_walk":
            x = torch.zeros(first_image_features.shape[0] + second_image_features.shape[0], first_image_features.shape[1],
                            dtype=first_image_features.dtype, device=first_image_features.device)
            num_vertices_per_image = first_image_features.shape[0]
            half_num_vertices = num_vertices_per_image // 2
            x[:num_vertices_per_image:2, :] = first_image_features[:half_num_vertices, :]
            x[1:num_vertices_per_image:2, :] = second_image_features[:half_num_vertices, :]
            x[num_vertices_per_image::2, :] = second_image_features[half_num_vertices:, :]
            x[num_vertices_per_image + 1::2, :] = first_image_features[half_num_vertices:, :]
            return x
        else:
            raise ValueError(f"Unsupported partition type: {self.partition_type}")

    def __populate_cluster_edges(self, edges: List[List[int]], cluster_indices: List[int]):
        for i in cluster_indices[:-1]:
            for j in cluster_indices[i + 1:]:
                edges[0].append(i)
                edges[1].append(j)

    def __add_between_clusters_edges(self, edges: List[List[int]]):
        edges[0].append(self.first_cluster_indices[0])
        edges[1].append(self.second_cluster_indices[0])

    def save(self, path: str):
        state_dict = {
            "train_first_image_features_list": self.train_first_image_features_list,
            "train_second_image_features_list": self.train_second_image_features_list,
            "train_labels": self.train_labels,
            "test_first_image_features_list": self.test_first_image_features_list,
            "test_second_image_features_list": self.test_second_image_features_list,
            "test_labels": self.test_labels,
            "additional_metadata": self.additional_metadata
        }
        torch.save(state_dict, path)

    @staticmethod
    def load(path: str, partition_type: str = "low_walk", device=torch.device("cpu")):
        state_dict = torch.load(path, map_location=device)
        return IsSameClassData(train_first_image_features_list=state_dict["train_first_image_features_list"],
                               train_second_image_features_list=state_dict["train_second_image_features_list"],
                               train_labels=state_dict["train_labels"],
                               test_first_image_features_list=state_dict["test_first_image_features_list"],
                               test_second_image_features_list=state_dict["test_second_image_features_list"],
                               test_labels=state_dict["test_labels"],
                               partition_type=partition_type,
                               additional_metadata=state_dict["additional_metadata"])
