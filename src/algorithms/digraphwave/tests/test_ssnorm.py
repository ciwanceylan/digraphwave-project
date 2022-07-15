import pytest
import numpy as np
import scipy.sparse as sp
import scipy
import torch
import torch_sparse as tsp

import digraphwave.test_graphs as test_graphs
import digraphwave.utils as utils
import digraphwave.ss_norm as ss_norm


# @pytest.fixture(scope="class")
# def ss_dir_adj() -> sp.spmatrix:
#     """The rw Laplacian of a directed source star graph"""
#     adj = test_graphs.source_star_adj(source_degree=5, beta=2, ell=3)
#     return utils.rw_laplacian(adj)
#
#
# @pytest.fixture(scope="class")
# def cycle_with_chords_dir_lap() -> sp.spmatrix:
#     """The rw Laplacian of a directed source star graph"""
#     adj = test_graphs.cycle_with_chords_adj(n=50, num_chords=2)
#     return utils.rw_laplacian(adj)


@pytest.mark.parametrize("d,beta,ell,tau", [(1, 5, 1, 1.), (10, 1, 5, 2.), (2, 2, 2, 4.), (3, 2, 5, 5.)])
def test_norm_values(d, beta, ell, tau):
    adj = test_graphs.source_star_adj(d, beta, ell, directed=True)
    n = adj.shape[0]
    lap = utils.rw_laplacian(adj)
    scipy_psi = scipy.linalg.expm(-tau * lap)
    sources = [0]
    targets_per_source, _ = utils.sparse2dict(scipy_psi[:, 0])

    norm_values = ss_norm._calc_log_normalization_values(adj, tau=tau,
                                                         sources=sources,
                                                         targets_per_source=targets_per_source)
    norm_values = norm_values.toarray()
    scipy_res = np.log(scipy_psi[:, 0].toarray())
    assert norm_values.shape == scipy_res.shape
    assert np.allclose(norm_values, scipy_res)


@pytest.mark.parametrize("d,beta,ell,tau", [(1, 5, 1, 1.), (10, 1, 5, 2.), (2, 2, 2, 4.), (3, 2, 5, 5.), (3, 3, 5, 5.)])
def test_normalise_dense(d, beta, ell, tau):
    adj = test_graphs.source_star_adj(d, beta, ell, directed=True)
    n = adj.shape[0]
    lap = utils.rw_laplacian(adj)
    psi = tsp.SparseTensor.from_scipy(scipy.linalg.expm(-tau * lap)).to_dense()

    # sources = np.random.permutation(n)[:10]
    sources = [0, 1, n // 2, n - 2, n - 1]
    sources = np.unique(sources)
    norm_psi = ss_norm.normalize(psi[:, sources], adj, tau, sources)
    assert isinstance(norm_psi, torch.Tensor)
    assert norm_psi.shape[0] == n
    assert norm_psi.shape[1] == len(sources)
    normalised_values = norm_psi.numpy()
    # answer = tsp.SparseTensor.from_dense(psi[:, sources])
    # answer = answer.fill_value(1.).to_dense().numpy()
    assert np.allclose(normalised_values[:, 0], np.ones(n))
    other_nodes_values = normalised_values[:, 1:].ravel()
    other_nodes_values = other_nodes_values[other_nodes_values > 0]
    if beta == d:
        assert np.allclose(other_nodes_values, 1.)
    else:
        assert np.all(other_nodes_values >= 1. - 1e-8)


@pytest.mark.parametrize("d,beta,ell,tau", [(1, 5, 1, 1.), (10, 1, 5, 2.), (2, 2, 2, 4.), (3, 2, 5, 5.), (3, 3, 5, 5.)])
def test_normalise_sparse(d, beta, ell, tau):
    """ Test expm normalisation using the source-star graph """
    adj = test_graphs.source_star_adj(d, beta, ell, directed=True)
    n = adj.shape[0]
    lap = utils.rw_laplacian(adj)
    psi = tsp.SparseTensor.from_scipy(scipy.linalg.expm(-tau * lap))

    # sources = np.random.permutation(n)[:10]
    sources = [0, 1, n // 2, n - 2, n - 1]
    sources = list(set(sources))
    norm_psi = ss_norm.normalize(psi[:, sources], adj, tau, sources)
    assert isinstance(norm_psi, tsp.SparseTensor)
    assert norm_psi.size(0) == n
    assert norm_psi.size(1) == len(sources)
    normalised_values1 = norm_psi[:, 0].storage.value().numpy()
    assert np.allclose(normalised_values1, np.ones(n))  # For centre node, all normalised values should be 1
    normalised_values2 = norm_psi[:, 1:].storage.value().numpy()
    if beta == d:
        assert np.allclose(normalised_values2, 1.)
    else:
        assert np.all(normalised_values2 >= 1. - 1e-8)  # For all other nodes should be larger or equal to 1.
