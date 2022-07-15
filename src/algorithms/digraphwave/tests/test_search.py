import networkx as nx
import pytest
import numpy as np

import digraphwave.test_graphs as tgraphs
import digraphwave.search as search


def nx_dists(adj, directed: bool):
    if directed:
        graph = nx.from_scipy_sparse_matrix(adj.T, create_using=nx.DiGraph, parallel_edges=True)
    else:
        graph = nx.from_scipy_sparse_matrix(adj.T, create_using=nx.Graph, parallel_edges=True)
    distances = nx.shortest_path_length(graph)
    return distances


@pytest.fixture(scope='function')
def cycle(request):
    adj = tgraphs.cycle_adj(10, directed=request.param)
    nx_distances = nx_dists(adj, request.param)
    return adj, nx_distances


@pytest.fixture(scope='function')
def cycle_with_chords(request):
    adj = tgraphs.cycle_with_chords_adj(10, 2, directed=request.param)
    nx_distances = nx_dists(adj, request.param)
    return adj, nx_distances


class TestShortestPath:

    @pytest.mark.parametrize('cycle', [True, False], indirect=True)
    def test_cycle(self, cycle):
        """ Test that joblib_parallel_sp_to_targets finds shortest path distances on a cycle graph """
        adj = cycle[0]
        distances_nx = cycle[1]
        n = adj.shape[0]
        sources = list(range(n))
        targets = np.asarray(n * [sources], dtype=np.int64)
        distances_search = search.joblib_parallel_sp_to_targets(adj, sources, targets)
        for i, (dists_s, (nx_i, dists_nx)) in enumerate(zip(distances_search, distances_nx)):
            assert i == nx_i
            for d_s, target in zip(dists_s, targets[i]):
                assert d_s == dists_nx[target]

    @pytest.mark.parametrize('cycle_with_chords', [True, False], indirect=True)
    def test_cycle_chord(self, cycle_with_chords):
        """ Test that joblib_parallel_sp_to_targets finds shortest path distances on a cycle graph with chords"""
        adj = cycle_with_chords[0]
        distances_nx = cycle_with_chords[1]
        n = adj.shape[0]
        sources = list(range(n))
        targets = np.asarray(n * [sources], dtype=np.int64)
        distances_search = search.joblib_parallel_sp_to_targets(adj, sources, targets)
        for i, (dists_s, (nx_i, dists_nx)) in enumerate(zip(distances_search, distances_nx)):
            assert i == nx_i
            for d_s, target in zip(dists_s, targets[i]):
                assert d_s == dists_nx[target]

    @pytest.mark.parametrize('cycle', [True, False], indirect=True)
    def test_cycle_arbitrary_source_order(self, cycle):
        """ Test that joblib_parallel_sp_to_targets finds shortest path distances on a cycle graph """
        adj = cycle[0]
        distances_nx = cycle[1]
        n = adj.shape[0]
        sources = np.asarray([3, 1, 6], dtype=np.int64)
        targets = {
            1: np.asarray([3, 9, 2], dtype=np.int64),
            6: np.asarray([1, 5], dtype=np.int64),
            3: np.asarray([0, 3, 9, 8], dtype=np.int64)
        }
        distances_search = search.joblib_parallel_sp_to_targets(adj, sources, targets)

        distances_nx = list(distances_nx)
        for i, s in enumerate(sources):
            dists_s = distances_search[i]
            nx_i, dists_nx = distances_nx[s]
            assert nx_i == s
            for j, t in enumerate(targets[s]):
                assert dists_s[j] == dists_nx[t]

    # @pytest.mark.parametrize('cycle,cycle_with_chords,degree',
    #                          [[True, True, 1], [False, False, 2]],
    #                          indirect=["cycle", "cycle_with_chords"])
    # def test_limit_max_degree(self, cycle, cycle_with_chords, degree):
    #     """ Test ability to """
    #     adj = cycle_with_chords[0]
    #     distances_nx = cycle[1]
    #     n = adj.shape[0]
    #     sources = list(range(n))
    #     targets = np.asarray(n * [sources])
    #     distances_search = search.joblib_parallel_sp_to_targets(adj, sources, targets, max_degree=degree)
    #     for i, (dists_s, (nx_i, dists_nx)) in enumerate(zip(distances_search, distances_nx)):
    #         assert i == nx_i
    #         for d_s, target in zip(dists_s, targets[i]):
    #             assert d_s == dists_nx[target]
