import pytest
import numpy as np
import scipy.sparse as sp
import networkx as nx
import math
import torch

import digraphwave.utils as utils
import digraphwave.test_graphs as tgraphs


@pytest.fixture(scope='class')
def square_tail_adj():
    """ A graph
        5   4◄───3
            ▲    │
            │    │
        0───1───►2
    """
    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=2)
    graph.add_edge(1, 0, weight=1)
    graph.add_edge(1, 1, weight=6)
    graph.add_edge(1, 2, weight=3)
    graph.add_edge(2, 3, weight=1)
    graph.add_edge(3, 2, weight=1)
    graph.add_edge(3, 4, weight=5)
    graph.add_edge(1, 4, weight=0.5)

    graph.add_node(5)
    return nx.adjacency_matrix(graph, dtype=np.float64).T


@pytest.fixture(scope='class')
def square_tail_adj_no_self_loops(square_tail_adj):
    """ A graph
        5   4◄───3
            ▲    │
            │    │
        0───1───►2
    """
    return utils.remove_self_loops(square_tail_adj)


@pytest.fixture(scope='class')
def square_tail_dict():
    """ A graph
        5   4◄───3
            ▲    │
            │    │
        0───1───►2
    """
    adj_dict = {
        0: np.asarray([1], dtype=np.int64),
        1: np.asarray([0, 1, 2, 4], dtype=np.int64),
        2: np.asarray([3], dtype=np.int64),
        3: np.asarray([2, 4], dtype=np.int64),
        4: np.asarray([], dtype=np.int64),
        5: np.asarray([], dtype=np.int64)
    }

    weights = {
        0: np.asarray([2], dtype=np.float64),
        1: np.asarray([1, 6, 3, 0.5], dtype=np.float64),
        2: np.asarray([1], dtype=np.float64),
        3: np.asarray([1, 5], dtype=np.float64),
        4: np.asarray([], dtype=np.int64),
        5: np.asarray([], dtype=np.int64)
    }

    return adj_dict, weights


@pytest.fixture(scope='class')
def square_tail_dict_row():
    """ A graph
        5   4◄───3
            ▲    │
            │    │
        0───1───►2
    """
    adj_dict = {
        0: np.asarray([1], dtype=np.int64),
        1: np.asarray([0, 1], dtype=np.int64),
        2: np.asarray([1, 3], dtype=np.int64),
        3: np.asarray([2], dtype=np.int64),
        4: np.asarray([1, 3], dtype=np.int64),
        5: np.asarray([], dtype=np.int64)
    }

    weights = {
        0: np.asarray([1], dtype=np.float64),
        1: np.asarray([2, 6], dtype=np.float64),
        2: np.asarray([3, 1], dtype=np.float64),
        3: np.asarray([1], dtype=np.float64),
        4: np.asarray([0.5, 5], dtype=np.float64),
        5: np.asarray([], dtype=np.int64)
    }

    return adj_dict, weights


def test_remove_self_loops(dir_adj_mat: sp.spmatrix):
    assert dir_adj_mat[301, 301] > 0
    adj = utils.remove_self_loops(dir_adj_mat)
    assert adj[301, 301] == 0


class TestDegrees:

    def test_outdegrees_weighted(self, dir_graph: nx.DiGraph, dir_adj_mat: sp.spmatrix):
        out_degs = utils.calc_out_degrees(dir_adj_mat, weighted=True)
        for node, deg in dir_graph.out_degree(weight='weight'):
            assert out_degs[node] == pytest.approx(deg)

        out_degs_dense = utils.calc_out_degrees(torch.from_numpy(dir_adj_mat.toarray()), weighted=True)
        assert np.allclose(out_degs, out_degs_dense.numpy())

    def test_indegrees_weighted(self, dir_graph: nx.DiGraph, dir_adj_mat: sp.spmatrix):
        in_degs = utils.calc_in_degrees(dir_adj_mat, weighted=True)
        for node, deg in dir_graph.in_degree(weight='weight'):
            assert in_degs[node] == pytest.approx(deg)

        in_degs_dense = utils.calc_in_degrees(torch.from_numpy(dir_adj_mat.toarray()), weighted=True)
        assert np.allclose(in_degs, in_degs_dense.numpy())

    def test_outdegrees(self, dir_graph: nx.DiGraph, dir_adj_mat: sp.spmatrix):
        out_degs = utils.calc_out_degrees(dir_adj_mat, weighted=False)
        for node, deg in dir_graph.out_degree():
            assert out_degs[node] == pytest.approx(deg)

        out_degs_dense = utils.calc_out_degrees(torch.from_numpy(dir_adj_mat.toarray()), weighted=False)
        assert np.allclose(out_degs, out_degs_dense.numpy())

    def test_indegrees(self, dir_graph: nx.DiGraph, dir_adj_mat: sp.spmatrix):
        in_degs = utils.calc_in_degrees(dir_adj_mat, weighted=False)
        for node, deg in dir_graph.in_degree():
            assert in_degs[node] == pytest.approx(deg)

        in_degs_dense = utils.calc_in_degrees(torch.from_numpy(dir_adj_mat.toarray()), weighted=False)
        assert np.allclose(in_degs, in_degs_dense.numpy())


class TestBetas:

    @pytest.mark.parametrize("d,beta,ell", [(1, 5, 1), (10, 1, 5), (2, 2, 2), (3, 2, 5)])
    def test_ss_beta(self, d, beta, ell):
        adj = tgraphs.source_star_adj(d, beta, ell, directed=True)
        ss_beta = utils.calc_source_star_beta(adj)
        if ell == 1:
            assert ss_beta == d
        else:
            num_nodes_with_degree = 1 + d * np.sum(np.power(beta, np.arange(ell - 1)))
            assert num_nodes_with_degree == np.sum(utils.calc_out_degrees(adj, weighted=False) > 0)
            assert ss_beta == ((num_nodes_with_degree - 1) * beta + d) / num_nodes_with_degree

    @pytest.mark.parametrize("d,beta,ell", [(1, 5, 1), (10, 1, 5), (2, 2, 2), (3, 2, 5)])
    def test_ss_beta_i(self, d, beta, ell):
        adj = tgraphs.source_star_adj(d, beta, ell, directed=True)
        ss_beta_i = utils.calc_ss_beta_per_node(adj)
        if ell == 1:
            assert ss_beta_i[0] == 0
        else:
            assert ss_beta_i[0] == beta

    def test_ss_beta_i_square_tail(self, square_tail_adj_no_self_loops):
        """ Test the beta_i on square tail graph """
        ss_beta_i = utils.calc_ss_beta_per_node(square_tail_adj_no_self_loops)
        answer = np.asarray([2., 4. / 3., 2., 5. / 3., 7. / 4, 7. / 4])
        assert np.allclose(ss_beta_i, answer)


@pytest.mark.parametrize("R", [0, 1, 2, 10, 20])
def test_thresholds(square_tail_adj_no_self_loops, R):
    min_value = 1e-6
    theta = utils.calculate_thetas(square_tail_adj_no_self_loops, radius=R, min_value=min_value)
    deg = utils.calc_out_degrees(adj=square_tail_adj_no_self_loops, weighted=False)
    beta = np.asarray([2., 4. / 3., 2., 5. / 3., 7. / 4, 7. / 4])
    if R == 0:
        answer = np.asarray([min_value, min_value, min_value, min_value, min_value, min_value])
        assert np.allclose(theta, answer)
    elif R == 1:
        answer = np.asarray([np.exp(-1), np.exp(-1) / 3, np.exp(-1) / 1, np.exp(-1) / 2, np.exp(-1), np.exp(-1)])
        assert np.allclose(theta, answer)
    elif R > 1:
        answer = np.exp(-1) / (np.power(beta[:4], R - 1) * deg[:4] * math.factorial(R))
        answer_isolated = np.exp(np.asarray([-R, -R]))
        assert np.allclose(theta[:4], np.maximum(answer, min_value))
        assert np.allclose(theta[4:], np.maximum(answer_isolated, min_value))


class TestInvWithZero:

    def test_inv(self, dir_adj_mat: sp.spmatrix):
        out_degs = utils.calc_out_degrees(dir_adj_mat, weighted=True)
        inverse = utils.inversewithzero(out_degs)
        assert np.all(np.isfinite(inverse))

        in_degs = utils.calc_in_degrees(dir_adj_mat, weighted=True)
        inverse = utils.inversewithzero(in_degs)
        assert np.all(np.isfinite(inverse))

    def test_sqrt_inv(self, dir_adj_mat: sp.spmatrix):
        out_degs = utils.calc_out_degrees(dir_adj_mat, weighted=True)
        inverse = utils.sqrtinversewithzero(out_degs)
        assert np.all(np.isfinite(inverse))

        in_degs = utils.calc_in_degrees(dir_adj_mat, weighted=True)
        inverse = utils.sqrtinversewithzero(in_degs)
        assert np.all(np.isfinite(inverse))


class TestNormalizedLaplacians:

    def test_format(self, dir_adj_mat):
        lap = utils.normalized_laplacian(dir_adj_mat.tocsc())
        assert lap.getformat() == 'csc'

        lap = utils.normalized_laplacian(dir_adj_mat.tocsr())
        assert lap.getformat() == 'csr'

    def test_dtype(self, dir_adj_mat):
        lap = utils.normalized_laplacian(dir_adj_mat, index_dtype=np.int64, data_dtype=np.float64)
        assert lap.indices.dtype == np.int64
        assert lap.indptr.dtype == np.int64
        assert lap.data.dtype == np.float64

    def test_normalized(self, dir_adj_mat):
        out_degrees = utils.calc_out_degrees(dir_adj_mat, weighted=True)
        in_degrees = utils.calc_in_degrees(dir_adj_mat, weighted=True)
        lap = utils.normalized_laplacian(dir_adj_mat).tocoo()
        for i, j, w in zip(lap.row, lap.col, lap.data):
            denominator = (np.sqrt(in_degrees[i] * out_degrees[j]))
            value = 0 if not denominator > 0 else dir_adj_mat[i, j] / denominator
            if i == j:
                assert w == pytest.approx(1. - value)
            else:
                assert w == pytest.approx(- value)

    def test_normalized_undirected(self, undir_adj_mat):
        out_degrees = utils.calc_out_degrees(undir_adj_mat, weighted=True)
        in_degrees = utils.calc_in_degrees(undir_adj_mat, weighted=True)
        lap = utils.normalized_laplacian(undir_adj_mat).tocoo()
        for i, j, w in zip(lap.row, lap.col, lap.data):
            denominator = (np.sqrt(in_degrees[i] * out_degrees[j]))
            value = 0 if not denominator > 0 else undir_adj_mat[i, j] / denominator
            if i == j:
                assert w == pytest.approx(1. - value)
            else:
                assert w == pytest.approx(- value)


class TestRWNormalizedLaplacians:

    def test_format(self, dir_adj_mat):
        lap = utils.rw_laplacian(dir_adj_mat.tocsc())
        assert lap.getformat() == 'csc'

        lap = utils.rw_laplacian(dir_adj_mat.tocsr())
        assert lap.getformat() == 'csr'

    def test_dtype(self, dir_adj_mat):
        lap = utils.rw_laplacian(dir_adj_mat, index_dtype=np.int64, data_dtype=np.float64)
        assert lap.indices.dtype == np.int64
        assert lap.indptr.dtype == np.int64
        assert lap.data.dtype == np.float64

    def test_normalized_out(self, dir_adj_mat):
        out_degrees = utils.calc_out_degrees(dir_adj_mat, weighted=True)
        lap = utils.rw_laplacian(dir_adj_mat, use_out_degrees=True)
        assert lap[300, 300] == 0
        assert lap[301, 301] == 0
        lap = lap.tocoo()
        for i, j, w in zip(lap.row, lap.col, lap.data):
            denominator = out_degrees[j]
            value = 0 if not denominator > 0 else dir_adj_mat[i, j] / denominator
            if i == j:
                assert w == pytest.approx(1. - value)
            else:
                assert w == pytest.approx(- value)

    def test_normalized_in(self, dir_adj_mat):
        in_degrees = utils.calc_in_degrees(dir_adj_mat, weighted=True)
        lap = utils.rw_laplacian(dir_adj_mat, use_out_degrees=False)
        assert lap[300, 300] == 0
        assert lap[301, 301] == 0
        lap = lap.tocoo()
        for i, j, w in zip(lap.row, lap.col, lap.data):
            denominator = in_degrees[i]
            value = 0 if not denominator > 0 else dir_adj_mat[i, j] / denominator
            if i == j:
                assert w == pytest.approx(1. - value)
            else:
                assert w == pytest.approx(- value)

    def test_normalised_dense_out(self, dir_adj_mat):
        lap = utils.rw_laplacian_dense(torch.from_numpy(dir_adj_mat.toarray())).numpy()
        lap_answer = utils.rw_laplacian(dir_adj_mat).toarray()
        assert np.allclose(lap, lap_answer)


class TestSparseDictConversions:

    def test_sparse2dict(self, square_tail_adj, square_tail_dict):
        targets, weights = utils.sparse2dict(square_tail_adj, return_weights=True)
        ansert_targets, answer_weights = square_tail_dict
        for i in range(6):
            assert np.all(targets[i] == ansert_targets[i])
            assert np.allclose(weights[i], answer_weights[i])

    def test_sparse2dict_row(self, square_tail_adj, square_tail_dict_row):
        targets, weights = utils.sparse2dict(square_tail_adj, return_weights=True, use_rows=True)
        ansert_targets, answer_weights = square_tail_dict_row
        for i in range(6):
            assert np.all(targets[i] == ansert_targets[i])
            assert np.allclose(weights[i], answer_weights[i])

    @pytest.mark.parametrize("use_rows", [False, True])
    def test_dict2sparse(self, square_tail_dict, square_tail_adj, use_rows):
        targets, weights = square_tail_dict
        spmat = utils.mapping2sparse(targets, weights, num_targets=6, sources_are_rows=use_rows)
        if use_rows:
            assert isinstance(spmat, sp.csr_matrix)
            assert (spmat != square_tail_adj.T).nnz == 0
        else:
            assert isinstance(spmat, sp.csc_matrix)
            assert (spmat != square_tail_adj).nnz == 0

    @pytest.mark.parametrize("use_rows", [False, True])
    def sparse2dict_sources_order(self, square_tail_dict, square_tail_adj, square_tail_dict_row, use_rows):
        sources = [4, 3, 5, 2, 1, 0]
        targets, weights = utils.sparse2dict(square_tail_adj, return_weights=True, use_rows=use_rows)

        if use_rows:
            for i, s in enumerate(sources):
                assert np.all(targets[s] == square_tail_dict[0][i])
                assert np.allclose(weights[s], square_tail_dict[1][i])
        else:
            for i, s in enumerate(sources):
                assert np.all(targets[s] == square_tail_dict_row[0][i])
                assert np.allclose(weights[s], square_tail_dict_row[1][i])

        spmat = utils.mapping2sparse(targets, weights, num_targets=6, sources=sources, sources_are_rows=use_rows)
        assert (spmat != square_tail_adj).nnz == 0


@pytest.fixture(scope="class")
def four_node_chord_adj():
    edges = np.asarray([
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 2]
    ])
    adj = sp.coo_matrix((np.ones(edges.shape[0], dtype=np.float64), (edges[:, 1], edges[:, 0])), shape=[4, 4])
    return adj.tocsc()


@pytest.fixture(scope="class")
def four_node_chord_adj_sym():
    adj_s = np.asarray([
        [0., 0.5, 0.5, 0.5],
        [0.5, 0, 0.5, 0],
        [0.5, 0.5, 0, 0.5],
        [0.5, 0, 0.5, 0]
    ])
    return adj_s


def four_node_chord_phase_mat(q):
    adj_diff = np.asarray([
        [0., -1., -1., 1.],
        [1., 0, -1., 0],
        [1., 1., 0, -1.],
        [-1., 0, 1., 0]
    ])
    phase_mat = np.exp(2 * np.pi * 1j * q * adj_diff)
    return phase_mat


class TestHermitianLaplacian:

    @pytest.mark.parametrize("adj_name,q", [
        ("square_tail_adj_no_self_loops", 0),
        ("square_tail_adj_no_self_loops", 0.25),
        ("cycle", 0.25)])
    def test_help(self, adj_name, q, request):
        if adj_name == "cycle":
            adj = tgraphs.cycle_adj(10, directed=True)
        else:
            adj = request.getfixturevalue(adj_name)

        adj_s, phase_mat = utils._hermitian_lap_help(adj, q=q)
        answer1 = adj_s.copy()
        answer1.data = np.ones_like(answer1.data)
        print(phase_mat.toarray())
        if q < 1e-8:
            assert np.allclose(phase_mat.toarray(), answer1.toarray())

        hem = adj_s.multiply(phase_mat).toarray()
        if q < 1e-8:
            assert np.allclose(hem, adj_s.toarray().astype(np.complex128))
        elif np.isclose(q, 0.25) and adj_name == 'cycle':
            assert np.allclose(hem, -hem.T)

    @pytest.mark.parametrize("q", [0., 0.25, 0.5, 0.9])
    def test_help2(self, four_node_chord_adj, four_node_chord_adj_sym, q):
        adj_s, phase_mat = utils._hermitian_lap_help(four_node_chord_adj, q=q)
        phase_mat_ans = four_node_chord_phase_mat(q)
        assert np.allclose(adj_s.toarray(), four_node_chord_adj_sym)
        assert np.allclose(four_node_chord_adj_sym * phase_mat.toarray(), four_node_chord_adj_sym * phase_mat_ans)

    # @pytest.mark.parametrize("adj,q", [
    #     ("square_tail_adj", 0),
    #     ("square_tail_adj", 0.25),
    #     ("dir_adj_mat", 0.),
    #     ("dir_adj_mat", 0.25),
    #     ("dir_adj_mat", 0.99),
    #     ("cycle", 0.25)
    # ])
    # def test_hem_lap(self, adj, q, request):
    #     if adj == "cycle":
    #         adj = tgraphs.cycle_adj(10, directed=True)
    #     else:
    #         adj = request.getfixturevalue(adj)
    #     lap = utils.normalised_hermitian_laplacian(adj, q)
    #     assert utils.one_matrix_norm(lap - sp.eye(lap.shape[0])) <= 1.
