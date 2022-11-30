import numpy as np
import torch
from pygsp import graphs, utils
from scipy import sparse


def spectral_graph_sparsify(G, num_edges_to_keep: int):
    r"""Sparsify a graph (with Spielman-Srivastava).
    Adapted from the PyGSP implementation (https://pygsp.readthedocs.io/en/v0.5.1/reference/reduction.html).

    Parameters
    ----------
    G : PyGSP graph or sparse matrix
        Graph structure or a Laplacian matrix
    num_edges_to_keep : int
        Number of edges to keep in graph.

    Returns
    -------
    Mnew : Graph or sparse matrix
        New graph structure or sparse matrix

    References
    ----------
    See :cite:`spielman2011graph`, :cite:`rudelson1999random` and :cite:`rudelson2007sampling` for more information.
    """
    # Test the input parameters
    if isinstance(G, graphs.Graph):
        if not G.lap_type == 'combinatorial':
            raise NotImplementedError
        L = G.L
    else:
        L = G

    N = np.shape(L)[0]

    # Not sparse
    resistance_distances = utils.resistance_distance(L).toarray()
    # Get the Weight matrix
    if isinstance(G, graphs.Graph):
        W = G.W
    else:
        W = np.diag(L.diagonal()) - L.toarray()
        W[W < 1e-10] = 0

    W = sparse.coo_matrix(W)
    W.data[W.data < 1e-10] = 0
    W = W.tocsc()
    W.eliminate_zeros()

    start_nodes, end_nodes, weights = sparse.find(sparse.tril(W))

    weights = np.maximum(0, weights)
    Re = np.maximum(0, resistance_distances[start_nodes, end_nodes])
    Pe = weights * Re
    Pe = Pe / np.sum(Pe)

    results = np.random.choice(np.arange(np.shape(Pe)[0]), size=num_edges_to_keep, p=Pe, replace=False)
    new_weights = np.zeros(np.shape(weights)[0])
    new_weights[results] = 1

    sparserW = sparse.csc_matrix((new_weights, (start_nodes, end_nodes)),
                                 shape=(N, N))
    sparserW = sparserW + sparserW.T
    sparserL = sparse.diags(sparserW.diagonal(), 0) - sparserW

    if isinstance(G, graphs.Graph):
        sparserW = sparse.diags(sparserL.diagonal(), 0) - sparserL
        if not G.is_directed():
            sparserW = (sparserW + sparserW.T) / 2.

        Mnew = graphs.Graph(W=sparserW)
    else:
        Mnew = sparse.lil_matrix(sparserL)

    return Mnew


def get_spectral_graph_sparsify_edge_removal_order(G):
    r"""Gets an edge removal order for sparsifying a graph while maintaining spectral properties (with Spielman-Srivastava).
    Adapted from the PyGSP implementation (https://pygsp.readthedocs.io/en/v0.5.1/reference/reduction.html).

    Parameters
    ----------
    G : PyGSP graph or sparse matrix
        Graph structure or a Laplacian matrix

    Returns
    -------
    A tensor of shape (2, num_edges) containing the edges in removal order. Contains only one direction per edge.

    References
    ----------
    See :cite:`spielman2011graph`, :cite:`rudelson1999random` and :cite:`rudelson2007sampling` for more information.
    """
    # Test the input parameters
    if isinstance(G, graphs.Graph):
        if not G.lap_type == 'combinatorial':
            raise NotImplementedError
        L = G.L
    else:
        L = G

    # Not sparse
    resistance_distances = utils.resistance_distance(L).toarray()
    # Get the Weight matrix
    if isinstance(G, graphs.Graph):
        W = G.W
    else:
        W = np.diag(L.diagonal()) - L.toarray()
        W[W < 1e-10] = 0

    W = sparse.coo_matrix(W)
    W.data[W.data < 1e-10] = 0
    W = W.tocsc()
    W.eliminate_zeros()

    start_nodes, end_nodes, weights = sparse.find(sparse.tril(W))

    weights = np.maximum(0, weights)
    Re = np.maximum(0, resistance_distances[start_nodes, end_nodes])
    Pe = weights * Re
    Pe = Pe / np.sum(Pe)

    results = np.random.choice(np.arange(np.shape(Pe)[0]), size=np.shape(Pe)[0], p=Pe, replace=False)
    ordered_start_nodes = start_nodes[results][::-1]
    ordered_end_nodes = end_nodes[results][::-1]
    return torch.tensor(np.stack([ordered_start_nodes, ordered_end_nodes]))
