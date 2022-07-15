import pytest
import numpy as np
import scipy.sparse as sp
import scipy
import torch

import digraphwave.test_graphs as test_graphs
import digraphwave.utils as utils
import digraphwave.ss_norm as ssexpm
import digraphwave.search as search
import digraphwave.expm as expm

from tests.utils import compare_cheb_coeff, compare_taylor_coeff


@pytest.fixture(scope="class")
def ss_dir_adj() -> sp.spmatrix:
    """The rw Laplacian of a directed source star graph"""
    adj = test_graphs.source_star_adj(source_degree=5, beta=2, ell=3)
    return utils.rw_laplacian(adj)


@pytest.fixture(scope="class")
def cycle_with_chords_dir_lap() -> sp.spmatrix:
    """The rw Laplacian of a directed source star graph"""
    adj = test_graphs.cycle_with_chords_adj(n=50, num_chords=2)
    return utils.rw_laplacian(adj)


@pytest.mark.parametrize("d,beta,ell,tau", [(1, 5, 1, 1.), (10, 1, 5, 2.), (2, 2, 2, 4.)])
def test_source_star_analytical_solution(d, beta, ell, tau):
    """Test that the analytical solution of the source star graph agrees with matrix exponential calculation """
    adj = test_graphs.source_star_adj(d, beta, ell, directed=True)
    n = adj.shape[0]
    lap = utils.rw_laplacian(adj)
    scipy_res = np.log(scipy.linalg.expm(-tau * lap.toarray())[:, 0])
    degrees = utils.calc_out_degrees(adj, weighted=False)
    distances = search.joblib_parallel_sp_to_targets(adj,
                                                     [0],
                                                     targets_per_source=[np.arange(0, n, dtype=np.int64)])
    distances = distances[0]
    assert len(degrees) == len(distances)
    ss_expm_res = ssexpm.log_source_star_expm(distances, degrees, tau, beta, d)
    assert np.abs(scipy_res - ss_expm_res).max() < 1e-14


class TestTaylorApproximation:
    """ Test Taylor approximation of matrix exponential. Error compared against bound used in paper.
    Test for both directed and undirected graphs."""

    @pytest.mark.parametrize("scale,order", [(1, 8), (1, 10), (4, 20), (8, 60)])
    def test_coeffs(self, scale, order):
        """Test coefficients calculation using FFT against alternative implementation """
        compare_coeffs = compare_taylor_coeff(scale, order)

        coeffs = expm._taylor_expm_coeffs(tau=scale, order=order).numpy()
        assert len(compare_coeffs) == len(coeffs)
        assert np.abs(compare_coeffs - coeffs).max() < utils.one_taylor_bound(1, order - 1, scale)

    @pytest.mark.parametrize("batch_size", [1, 11, None])
    @pytest.mark.parametrize("directed", [True, False])
    @pytest.mark.parametrize("as_dense", [False, True])
    def test_expm_ss(self, batch_size, directed, as_dense):
        """ Test using a source star graph."""
        d = 5
        beta = 2
        ell = 3
        order = 40
        taus = (1, 2, 4, 8)
        adj = test_graphs.source_star_adj(d, beta, ell, directed=directed)
        n = adj.shape[0]
        lap = utils.rw_laplacian(adj)

        if as_dense:
            adj = torch.from_numpy(adj.toarray())

        expm_res = expm.lapexpm_taylor(adj=adj, taus=taus, order=order, batch_size=batch_size)

        for k_tau, tau in enumerate(taus):
            scipy_res = scipy.linalg.expm(-tau * lap.toarray())
            error = utils.one_matrix_norm(scipy_res - expm_res[k_tau].numpy())
            assert error < utils.one_taylor_bound(n, order - 1, tau)

    @pytest.mark.parametrize("batch_size", [1, 11, None])
    @pytest.mark.parametrize("directed", [True, False])
    @pytest.mark.parametrize("as_dense", [False, True])
    def test_expm_cycle_with_chords(self, batch_size, directed, as_dense):
        """ Test using a cycle graph with chords."""

        order = 40
        taus = (1, 2, 4, 8)
        adj = test_graphs.cycle_with_chords_adj(n=50, num_chords=2, directed=directed)
        n = adj.shape[0]
        lap = utils.rw_laplacian(adj)
        if as_dense:
            adj = torch.from_numpy(adj.toarray())

        expm_res = expm.lapexpm_taylor(adj=adj, taus=taus, order=order, batch_size=batch_size)

        for k_tau, tau in enumerate(taus):
            scipy_res = scipy.linalg.expm(-tau * lap.toarray())
            error = utils.one_matrix_norm(scipy_res - expm_res[k_tau].numpy())
            assert error < utils.one_taylor_bound(n, order - 1, tau)

    # @pytest.mark.parametrize("order,directed", [(12, True), (20, True), (60, True), (12, False), (60, False)])
    @pytest.mark.parametrize("order", [12, 20, 60])
    @pytest.mark.parametrize("directed", [True, False])
    @pytest.mark.parametrize("as_dense", [False, True])
    def test_expm_dimond(self, order, directed, as_dense):
        """ Test different orders using a diamond graph."""
        batch_size = None
        taus = (1, 2, 4, 8)
        adj = test_graphs.diamond_adj(n=100, directed=directed)
        n = adj.shape[0]
        lap = utils.rw_laplacian(adj)
        if as_dense:
            adj = torch.from_numpy(adj.toarray())

        expm_res = expm.lapexpm_taylor(adj=adj, taus=taus, order=order, batch_size=batch_size)

        for k_tau, tau in enumerate(taus):
            scipy_res = scipy.linalg.expm(-tau * lap.toarray())
            error = utils.one_matrix_norm(scipy_res - expm_res[k_tau].numpy())
            assert error < utils.one_taylor_bound(n, order - 1, tau)


class TestChebyshevApproximation:
    """ Test Chebyshev approximation of matrix exponential. Error compared against bound used in paper.
    Test only for undirected graphs."""

    @pytest.mark.parametrize("scale,order", [(1, 8), (1, 10), (4, 20), (6, 60)])
    def test_coeffs(self, scale, order):
        """Test coefficients calculation using FFT against alternative implementation """
        compare_coeffs = compare_cheb_coeff(scale, order)

        coeffs = expm._chebyshev_expm_coeffs(tau=scale, order=order).numpy()
        assert len(compare_coeffs) == len(coeffs)
        assert np.abs(compare_coeffs - coeffs).max() < utils.one_taylor_bound(1, order - 1, scale)

    @pytest.mark.parametrize("batch_size", [1, 11, None])
    @pytest.mark.parametrize("as_dense", [False, True])
    def test_expm_ss(self, batch_size, as_dense):
        """ Test using a source star graph."""
        d = 5
        beta = 2
        ell = 3
        order = 40
        taus = (1, 2, 4, 8)
        adj = test_graphs.source_star_adj(d, beta, ell, directed=False)
        n = adj.shape[0]
        lap = utils.rw_laplacian(adj)

        if as_dense:
            adj = torch.from_numpy(adj.toarray())

        expm_res = expm.lapexpm_chebyshev(adj=adj, taus=taus, order=order, batch_size=batch_size)

        for k_tau, tau in enumerate(taus):
            scipy_res = scipy.linalg.expm(-tau * lap.toarray())
            error = utils.one_matrix_norm(scipy_res - expm_res[k_tau].numpy())
            assert error < utils.one_taylor_bound(n, order - 1, tau)

    @pytest.mark.parametrize("batch_size", [1, 11, None])
    @pytest.mark.parametrize("as_dense", [False, True])
    def test_expm_cycle_with_chords(self, batch_size, as_dense):
        """ Test using a cycle graph with chords."""
        order = 40
        taus = (1, 2, 4, 8)
        adj = test_graphs.cycle_with_chords_adj(n=50, num_chords=2, directed=False)
        n = adj.shape[0]
        lap = utils.rw_laplacian(adj)

        if as_dense:
            adj = torch.from_numpy(adj.toarray())

        expm_res = expm.lapexpm_chebyshev(adj=adj, taus=taus, order=order, batch_size=batch_size)

        for k_tau, tau in enumerate(taus):
            scipy_res = scipy.linalg.expm(-tau * lap.toarray())
            error = utils.one_matrix_norm(scipy_res - expm_res[k_tau].numpy())
            assert error < utils.one_taylor_bound(n, order - 1, tau)

    @pytest.mark.parametrize("order", [12, 20, 60])
    @pytest.mark.parametrize("as_dense", [False, True])
    def test_expm_dimond(self, order, as_dense):
        """ Test different orders using a diamond graph."""
        batch_size = None
        taus = (1, 2, 4, 8)
        adj = test_graphs.diamond_adj(n=100, directed=False)
        n = adj.shape[0]
        lap = utils.rw_laplacian(adj)

        if as_dense:
            adj = torch.from_numpy(adj.toarray())

        expm_res = expm.lapexpm_chebyshev(adj=adj, taus=taus, order=order, batch_size=batch_size)

        for k_tau, tau in enumerate(taus):
            scipy_res = scipy.linalg.expm(-tau * lap.toarray())
            error = utils.one_matrix_norm(scipy_res - expm_res[k_tau].numpy())
            assert error < utils.one_taylor_bound(n, order - 1, tau)
