import pytest
import networkx as nx
import numpy as np
import tempfile

import rolxrefex.core as core
import rolxrefex.refex as refex
import rolxrefex.refex_sparse as refex_sparse
import rolxrefex.refex_dense as refex_dense


@pytest.fixture()
def five_node_graph_adj():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(0, 2)
    graph.add_edge(0, 3)
    graph.add_edge(1, 2)
    graph.add_edge(2, 1)
    graph.add_edge(2, 3)
    graph.add_edge(3, 4)
    graph.add_edge(2, 4)
    graph.add_edge(1, 4)
    graph.add_edge(4, 3)
    graph.add_edge(4, 2)
    graph.add_edge(4, 1)
    return nx.adjacency_matrix(graph).T


@pytest.fixture()
def triplet_adj():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_node(3)
    graph.add_node(4)
    graph.add_node(5)
    return nx.adjacency_matrix(graph).T


@pytest.fixture()
def triplet_features():
    features = np.asarray([
        [1, 0],
        [-1, -1],
        [2, 2],
        [0, 0],
        [0, 0],
        [0, 0]
    ])
    return features


@pytest.fixture()
def large_adj():
    graph = nx.fast_gnp_random_graph(1000, 0.01)
    return nx.adjacency_matrix(graph).T


@pytest.mark.parametrize('use_weights', [True, False])
def test_egonet_features(five_node_graph_adj, use_weights):
    ego_features = refex_sparse.extract_egonet_features(five_node_graph_adj, use_weights=use_weights)
    answer = np.asarray([[6., 3., 3., 6., 3., 3.],
                         [8., 3., 1., 8., 3., 1.],
                         [12., 0., 0., 12., 0., 0.],
                         [7., 3., 2., 7., 3., 2.],
                         [9., 0., 3., 9., 0., 3.]])
    if use_weights:
        assert np.allclose(ego_features, answer)
    else:
        assert np.allclose(ego_features, answer[:, :3])


@pytest.mark.parametrize('use_weights', [True, False])
def test_node_features(five_node_graph_adj, use_weights):
    node_features = refex_sparse.extract_node_features(five_node_graph_adj, use_weights=use_weights)

    answer = np.asarray([[3., 0., 3., 0.],
                         [2., 3., 2., 3.],
                         [3., 3., 3., 3.],
                         [1., 3., 1., 3.],
                         [3., 3., 3., 3.]])
    if use_weights:
        assert np.allclose(node_features, answer)
    else:
        assert np.allclose(node_features, answer[:, :2])


@pytest.mark.parametrize('adj', ['triplet_adj', 'five_node_graph_adj'])
def test_conversion_to_adj_dict(adj, request):
    adj = request.getfixturevalue(adj)
    dense_adj = adj.toarray()
    adj_dict_sp, weights_sp = core.sparse2symmetric_adj_dict(adj, return_weights=True, as_numba_dict=True)
    adj_dict_dense, weights_dense = core.dense2adj_dict(dense_adj, return_values=True)
    assert len(adj_dict_sp) == len(adj_dict_dense) == len(weights_sp) == len(weights_dense)
    for i in range(len(adj_dict_sp)):
        assert np.allclose(adj_dict_sp[i], adj_dict_dense[i])
        assert np.allclose(weights_sp[i], weights_dense[i])


def test_conversion_readonly():
    """ Ensure that weights are not references to original array """
    adj = np.random.rand(10, 12)
    adj_copy = adj.copy()

    adj_dict_dense, weights_dense = core.dense2row_adj_dict(adj, return_values=True)

    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            weights_dense[i][j] = 10.
            assert adj[i, j] != 10

    assert np.allclose(adj, adj_copy)


@pytest.mark.parametrize('tol', [-1., 0.])
def test_recursion(triplet_adj, triplet_features, tol):
    adj_dict, _ = core.sparse2symmetric_adj_dict(triplet_adj,
                                                 return_weights=False,
                                                 as_numba_dict=True)
    features, _ = core.do_feature_recursion(triplet_features, adj_dict, max_steps=1, tol=tol)
    answer = np.asarray([
        [1, 0, -1, -1, -1, -1],
        [-1, -1, 1.5, 1, 3, 2],
        [2, 2, -1, -1, -1, -1],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])
    answer[:, 2:] = answer[:, 2:] / np.linalg.norm(answer[:, 2:], axis=0)
    if tol < 0:
        assert np.allclose(features, answer)
    else:
        assert np.allclose(features, answer[:, :3])


@pytest.mark.parametrize('tol', [-1, 1e-10, 1e-6, 0.001, 1.])
def test_refex(large_adj, tol):
    param = core.RefexParam(max_emb_size=50, use_weights=False, prune_tol=tol)
    features, reprod_param = refex.refex(large_adj, param)

    features_dense, reprod_param = refex.refex(large_adj.toarray(), param)

    if tol < 0:
        assert features.shape[0] == 1000
        assert features.shape[1] == 45
    else:
        assert features.shape[0] == 1000
        assert features.shape[1] <= 45
        assert features.shape[1] >= 5

    assert np.allclose(features, features_dense)


@pytest.mark.parametrize('tol', [-1, 1e-10, 1e-6, 0.001, 1.])
def test_refex_reprod(large_adj, tol):
    param = core.RefexParam(max_emb_size=128, use_weights=False, prune_tol=tol)
    features1, reprod_param = refex.refex(large_adj, param)
    with tempfile.NamedTemporaryFile() as fp:
        reprod_param.save(fp.name)
        fp.seek(0)
        param2 = core.RefexParam.load(fp.name)

    features2, reprod_param = refex.refex(large_adj, param2)

    assert np.allclose(features1, features2)

    # if tol < 0:
    #     assert features.shape[0] == 1000
    #     assert features.shape[1] == 45
    # else:
    #     assert features.shape[0] == 1000
    #     assert features.shape[1] <= 45
    #     assert features.shape[1] >= 5
