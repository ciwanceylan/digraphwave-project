import pytest
import numpy as np
import scipy.sparse as sp
import scipy

import digraphwave.test_graphs as test_graphs


@pytest.mark.parametrize("d,beta,ell", [(1, 5, 3), (10, 1, 5), (5, 2, 1)])
def test_source_star_edges(d, beta, ell):
    """ Test generation of source star edges  """
    sources, targets = test_graphs._source_star_edges(d, beta, ell)
    num_nodes = test_graphs._ss_num_nodes(d, beta, ell)
    max_node_index = max(sources.max(), targets.max())
    assert num_nodes == max_node_index + 1
    assert len(sources) == num_nodes - 1
    assert len(targets) == num_nodes - 1


def test_edges2adj():
    """ Test conversion of edges to adjacency matrix for directed and undirected cases. """
    sources = np.asarray([0, 0, 1, 2])
    targets = np.asarray([1, 2, 2, 0])
    adj = test_graphs.edges2adj(sources, targets, directed=True, use_csc=True)
    assert isinstance(adj, sp.csc_matrix)
    assert adj.shape[0] == 3
    assert adj.shape[1] == 3

    assert adj[1, 0] == 1.
    assert adj[0, 1] == 0.
    assert adj[2, 0] == 1.
    assert adj[0, 2] == 1.

    adj = test_graphs.edges2adj(sources, targets, directed=False, use_csc=False)
    assert isinstance(adj, sp.csr_matrix)
    assert adj[1, 0] == 1.
    assert adj[0, 1] == 1.
    assert adj[2, 0] == 1.
    assert adj[0, 2] == 1.
