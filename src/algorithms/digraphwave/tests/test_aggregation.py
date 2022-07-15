import pytest
import networkx as nx
import numpy as np

import digraphwave.aggregation as diagg


@pytest.fixture()
def triplet_adj():
    graph = nx.DiGraph()
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)
    graph.add_edge(3, 2)
    graph.add_edge(3, 1)
    graph.add_node(4)
    return nx.adjacency_matrix(graph).T


@pytest.fixture()
def triplet_features():
    features = np.asarray([
        [1, 0],
        [-1, -1],
        [2, 2],
        [0, 0],
        [0, 0]
    ])
    return features


def test_aggregation(triplet_adj, triplet_features):

    features = diagg.FeatureAggregator(triplet_adj).create_enhanced_features(triplet_features)
    answer_new_features = np.asarray([
        [-1, -1],
        [1, 2. / 3.],
        [-0.5, -0.5],
        [0.5, 0.5],
        [0, 0]
    ])
    answer = np.concatenate((triplet_features, answer_new_features), axis=1)

    assert np.allclose(features, answer)
