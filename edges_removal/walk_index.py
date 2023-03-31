from typing import List

import torch
import torch_geometric.utils


def compute_walk_index(num_vertices: int, edge_index: torch.Tensor, L: int, vertices_subset: List[int], target_vertex: int = -1) -> int:
    """
    Computes the (L - 1)-walk index for a set of vertices.
    @param num_vertices: number of vertices in the graph.
    @param edge_index: edge index of the graph (according to PyTorch Geometric format).
    @param L: depth of the GNN for which the walk index is computed. That is, will compute the (L-1)-walk index.
    @param vertices_subset: a set of vertex indices for which to compute the walk index.
    @param target_vertex: if >= 0, the walk index is computed with respect to the given target vertex.
    """
    edge_index = torch_geometric.utils.add_remaining_self_loops(edge_index)[0]
    adj_matrix = __edge_index_to_adj_mat(num_vertices, edge_index)
    num_walks_mat = torch.matrix_power(adj_matrix, L - 1)

    vertices_on_boundary = __get_vertices_on_boundary_in_subset(num_vertices, adj_matrix, vertices_subset)
    remaining_vertices = [i for i in range(num_vertices) if i not in vertices_subset]
    vertices_on_boundary.extend(__get_vertices_on_boundary_in_subset(num_vertices, adj_matrix, remaining_vertices))

    if target_vertex < 0:
        return num_walks_mat[vertices_on_boundary].sum()
    else:
        return num_walks_mat[vertices_on_boundary, target_vertex].sum()


def __get_vertices_on_boundary_in_subset(num_vertices: int, adj_matrix: torch.Tensor, vertices_subset: List[int]) -> List[int]:
    remaining_vertices = [i for i in range(num_vertices) if i not in vertices_subset]
    remaining_to_vertices_subset = adj_matrix[remaining_vertices][:, vertices_subset]

    vertices_in_subset_on_boundary = torch.nonzero(remaining_to_vertices_subset.sum(dim=0) > 0).squeeze(dim=-1).tolist()
    vertices_in_subset_on_boundary_orig_indices = [vertices_subset[i] for i in vertices_in_subset_on_boundary]
    return vertices_in_subset_on_boundary_orig_indices


def __edge_index_to_adj_mat(num_vertices: int, edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.shape[1] > 0:
        return torch_geometric.utils.to_dense_adj(edge_index, max_num_nodes=num_vertices)[0]

    return torch.zeros(num_vertices, num_vertices, dtype=torch.float, device=edge_index.device)
