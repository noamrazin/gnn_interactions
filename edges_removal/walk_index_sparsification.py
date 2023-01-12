from typing import Tuple

import numpy as np
import torch
import torch_geometric
from tqdm import tqdm


class WalkIndexSparsifier:
    """
    Implementation of the (L - 1)-Walk Index Sparsification (WIS) algorithm (see Algorithm 1 in the paper).
    """

    def __init__(self, L: int, chunk_size: int = 100):
        """
        Creates a WalkIndexSparsifier which implements the (L - 1)-WIS algorithm.
        @param L: Depth of the GNN for which the graph is sparsified.
        @param chunk_size: Computes edge removal order in chunks of this size (default is one).
        """
        self.L = L
        self.chunk_size = chunk_size

    def __edge_index_to_adj_mat(self, num_vertices, edge_index) -> torch.Tensor:
        if edge_index.shape[1] > 0:
            return torch_geometric.utils.to_dense_adj(edge_index, max_num_nodes=num_vertices)[0]

        return torch.zeros(num_vertices, num_vertices, dtype=torch.float, device=edge_index.device)

    def __get_per_edge_walk_index_tuples(self, num_vertices: int, edge_index: torch.Tensor, undirected: bool = True) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes for each edge, except for self-loops, what the walk indices with respect to every singleton partition will be after its removal.
        Assumes self-loops exist.
        @param num_vertices: Number of vertices in the graph.
        @param edge_index: Tensor of shape (2, num_edges) describing the graph edges according to PyTorch Geometric format (first row is source and
        second row is target).
        @param undirected: Whether to treat edges as undirected or not.
        @return: (edges, per_edge_walk_index_tuple), where edges is of the same format of edge_index, but it excludes self-loops and contains only
        one direction representative per edge if undirected is True.
        """
        adj_matrix = self.__edge_index_to_adj_mat(num_vertices, edge_index)
        adj_matrix.fill_diagonal_(1)  # ensures self-loops exist
        not_self_loop_edges = (edge_index[0] != edge_index[1]).nonzero().squeeze(dim=1)

        edges = []
        per_edge_walk_indices = []
        visited_edges = set()
        for i in not_self_loop_edges:
            edge = edge_index[:, i]
            edge_tuple = (edge[0].item(), edge[1].item())
            if undirected and (edge_tuple[1], edge_tuple[0]) in visited_edges:
                continue

            adj_matrix[edge[0], edge[1]] = 0
            if undirected:
                adj_matrix[edge[1], edge[0]] = 0

            walk_matrix = torch.matrix_power(adj_matrix, self.L - 1)
            adj_matrix_no_self_loops = adj_matrix.clone()
            adj_matrix_no_self_loops.fill_diagonal_(0)

            per_vertex_out_neighbors_walks = (walk_matrix * adj_matrix_no_self_loops.t()).sum(dim=0)
            per_vertex_walks_to_self_if_has_incoming_edge = torch.diagonal(walk_matrix) * (adj_matrix_no_self_loops.sum(dim=0) > 0)
            per_edge_walk_indices.append(per_vertex_out_neighbors_walks + per_vertex_walks_to_self_if_has_incoming_edge)

            edges.append(edge)
            visited_edges.add(edge_tuple)

            adj_matrix[edge[0], edge[1]] = 1
            if undirected:
                adj_matrix[edge[1], edge[0]] = 1

        edges = torch.stack(edges, dim=1)
        per_edge_walk_indices = torch.stack(per_edge_walk_indices)
        return edges, per_edge_walk_indices

    def __sort_edges_by_walk_index_tuples(self, edge_index: torch.Tensor, per_edge_walk_indices: torch.Tensor) -> torch.Tensor:
        """
        Sorts edges by walk index tuples.
        @param edge_index: Tensor of shape (2, num_edges) describing the graph edges according to PyTorch Geometric format (first row is source and
        second row is target).
        @param per_edge_walk_indices: Tensor of shape (num_edges, num_vertices) describing the walk indices with respect to every singleton partition
        for each edge.
        @return: A tensor of shape (2, num_edges) containing the edges sorted by walk index tuples in descending order.
        """
        # sorts according to descending order since numpy lexsort considers last row as primary key
        per_edge_walk_indices = torch.sort(per_edge_walk_indices, descending=True, dim=1)[0]
        np_per_edge_walk_indices = per_edge_walk_indices.cpu().detach().numpy()
        sorting_indices = np.lexsort(np_per_edge_walk_indices.transpose())
        return edge_index[:, sorting_indices].flip(-1)

    def sparsify(self, num_vertices: int, edge_index: torch.Tensor, num_edges_to_remove: int, undirected: bool = True, device=torch.device('cpu'),
                 print_progress: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes edge_index of sparsified graph according to the (L - 1)-WIS algorithm.
        @param num_vertices: Number of vertices in the graph.
        @param edge_index: Tensor of shape (2, num_edges) describing the graph edges according to PyTorch Geometric format (first row is source and
        second row is target).
        @param num_edges_to_remove: Number of edges to remove.
        @param undirected: Whether to treat edges as undirected or not.
        @param device: PyTorch device to use.
        @param print_progress: Whether to print iteration of edge removal progress.
        @return: A tuple consisting of: (i) A tensor of shape (2, num_edges_remaining) containing the remaining edges;
        (ii) a tensor of shape (2, num_edges_to_remove) containing the removed edges in order of removal (first to last), where for undirected graphs only one side per edge appears;
        and (iii) a tensor holding the indices of all removed edges in the original edge_index in order of removal (first to last), where for undirected graphs the indices of both
        directions of a removed edge appear.
        """
        remaining_edges = edge_index.to(device)
        curr_indices = torch.arange(0, edge_index.shape[1], device=device)

        edges_to_remove_by_order = [[], []]
        indices_of_removed_edges = []
        for curr in tqdm(range(0, num_edges_to_remove, self.chunk_size), disable=not print_progress):
            if remaining_edges.shape[1] == 0:
                break

            removal_candidates_edge_index, per_edge_walk_indices = self.__get_per_edge_walk_index_tuples(num_vertices, remaining_edges, undirected)
            sorted_removal_candidates_edge_index = self.__sort_edges_by_walk_index_tuples(removal_candidates_edge_index, per_edge_walk_indices)

            chunk_size = min(self.chunk_size, num_edges_to_remove - curr)
            edges_to_remove = sorted_removal_candidates_edge_index[:, :chunk_size]
            edges_to_remove_by_order[0] += edges_to_remove[0]
            edges_to_remove_by_order[1] += edges_to_remove[1]

            directed_edges_to_remove = edges_to_remove
            if undirected:
                other_dir_edges_to_remove = torch.stack([edges_to_remove[1], edges_to_remove[0]])
                directed_edges_to_remove = torch.concat([edges_to_remove, other_dir_edges_to_remove], dim=1)

            mask_to_remove = torch.any(torch.all(remaining_edges[:, None, :] == directed_edges_to_remove[:, :, None], dim=0), dim=0)

            indices_of_removed_edges.append(curr_indices[mask_to_remove])
            curr_indices = curr_indices[~mask_to_remove]
            remaining_edges = remaining_edges[:, ~mask_to_remove]

        edges_to_remove_by_order = torch.tensor(edges_to_remove_by_order, device=device)
        indices_of_removed_edges = torch.cat(indices_of_removed_edges)
        return remaining_edges, edges_to_remove_by_order, indices_of_removed_edges


class EfficientOneWalkIndexSparsifier:
    """
    Efficient implementation for the 1-Walk Index Sparsification (WIS) algorithm based only on vertex degrees (see Algorithm 2 in the paper).
    """

    def __remove_self_loops(self, remaining_edges, curr_indices):
        mask = remaining_edges[0] != remaining_edges[1]
        remaining_edges = remaining_edges[:, mask]
        curr_indices = curr_indices[mask]
        return remaining_edges, curr_indices

    def __compute_index_of_edge_to_remove(self, remaining_edges, vertex_degrees):
        per_edge_degrees = vertex_degrees[remaining_edges]
        larger_degree_side_per_edge = torch.max(per_edge_degrees, dim=0)[0]
        smaller_degree_side_per_edge = torch.min(per_edge_degrees, dim=0)[0]

        max_min_degree = torch.max(smaller_degree_side_per_edge)
        maximal_min_degree_edges = smaller_degree_side_per_edge == max_min_degree
        maximal_max_degree_edge = torch.argmax(larger_degree_side_per_edge[maximal_min_degree_edges])

        index_of_edge_to_remove = torch.where(maximal_min_degree_edges)[0][maximal_max_degree_edge]
        return index_of_edge_to_remove

    def sparsify(self, num_vertices: int, edge_index: torch.Tensor, num_edges_to_remove: int, undirected: bool = True, device=torch.device('cpu'),
                 print_progress: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes edge_index of sparsified graph for the 1-WIS algorithm (efficient implementation).
        @param num_vertices: Number of vertices in the graph.
        @param edge_index: Tensor of shape (2, num_edges) describing the graph edges according to PyTorch Geometric format (first row is source and
        second row is target).
        @param num_edges_to_remove: Number of edges to remove.
        @param undirected: Whether to treat edges as undirected or not.
        @param device: PyTorch device to use.
        @param print_progress: Whether to print iteration of edge removal progress.
        @return: A tuple consisting of: (i) A tensor of shape (2, num_edges_remaining) containing the remaining edges;
        (ii) a tensor of shape (2, num_edges_to_remove) containing the removed edges in order of removal (first to last), where for undirected graphs only one side per edge appears;
        and (iii) a tensor holding the indices of all removed edges in the original edge_index in order of removal (first to last), where for undirected graphs the indices of both
        directions of a removed edge appear.
        """
        remaining_edges = edge_index.to(device)
        curr_indices = torch.arange(0, remaining_edges.shape[1], device=device)

        # Remove self-loops, as they are not removed by the algorithm
        remaining_edges, curr_indices = self.__remove_self_loops(remaining_edges, curr_indices)

        edges_to_remove_by_order = [[], []]
        indices_of_removed_edges = []
        vertex_degrees = torch_geometric.utils.degree(remaining_edges[1], num_nodes=num_vertices)

        for _ in tqdm(range(num_edges_to_remove), disable=not print_progress):
            if remaining_edges.shape[1] == 0:
                break

            index_of_edge_to_remove = self.__compute_index_of_edge_to_remove(remaining_edges, vertex_degrees)
            edges_to_remove_by_order[0].append(remaining_edges[0][index_of_edge_to_remove].item())
            edges_to_remove_by_order[1].append(remaining_edges[1][index_of_edge_to_remove].item())

            vertex_degrees[remaining_edges[1][index_of_edge_to_remove]] -= 1

            edges_to_remove = remaining_edges[:, index_of_edge_to_remove].unsqueeze(dim=1)
            if undirected:
                vertex_degrees[remaining_edges[0][index_of_edge_to_remove]] -= 1
                other_dir_edge_to_remove = torch.stack([edges_to_remove[1], edges_to_remove[0]])
                edges_to_remove = torch.concat([edges_to_remove, other_dir_edge_to_remove], dim=1)

            mask_to_remove = torch.any(torch.all(remaining_edges[:, None, :] == edges_to_remove[:, :, None], dim=0), dim=0)

            indices_of_removed_edges.append(curr_indices[mask_to_remove])
            curr_indices = curr_indices[~mask_to_remove]
            remaining_edges = remaining_edges[:, ~mask_to_remove]

        edges_to_remove_by_order = torch.tensor(edges_to_remove_by_order, device=device)
        indices_of_removed_edges = torch.cat(indices_of_removed_edges)
        return remaining_edges, edges_to_remove_by_order, indices_of_removed_edges
