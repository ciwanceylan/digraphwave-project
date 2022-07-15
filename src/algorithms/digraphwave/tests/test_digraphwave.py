import pytest
import numpy as np
import torch

import digraphwave.test_graphs as tgraphs
import digraphwave.digraphwave as digw


@pytest.mark.parametrize("k_emb", [32, 64, 128, 256, 512])
def test_create_parameters(k_emb):
    d = 3
    beta = 3
    ell = 3
    R = 5
    adj = tgraphs.source_star_adj(d, beta, ell, directed=True)
    param = digw.DigraphwaveHyperparameters.create(num_nodes=adj.shape[0], num_edges=adj.nnz, R=R, k_emb=k_emb,
                                                   arctan_log_transform=True, n_flag=True,
                                                   t_flag=True, a_flag=True)
    if k_emb == 32:
        assert param.k_tau == 1
        assert param.k_phi == 4
        assert np.all(param.get_taus() == np.asarray([1], dtype=np.float64))
    elif k_emb == 64:
        assert param.k_tau == 2
        assert param.k_phi == 4
        assert np.all(param.get_taus() == np.asarray([1, R], dtype=np.float64))
    elif k_emb == 128:
        assert param.k_tau == 2
        assert param.k_phi == 8
        assert np.all(param.get_taus() == np.asarray([1, R], dtype=np.float64))
    elif k_emb == 256:
        assert param.k_tau == 3
        assert param.k_phi == 10
        assert np.all(param.get_taus() == np.asarray([1, (R + 1) / 2, R], dtype=np.float64))
    elif k_emb == 512:
        assert param.k_tau == 4
        assert param.k_phi == 16
        assert np.all(param.get_taus() == np.asarray([1, 1 + ((R - 1) / 3), 1 + 2*((R - 1) / 3),  R], dtype=np.float64))



@pytest.mark.parametrize("k_emb", [32, 64, 128, 256])
def test_new_parameters(k_emb):
    R = 5
    param = digw.DigraphwaveHyperparameters.create(num_nodes=1000, num_edges=1000, R=R, k_emb=k_emb,
                                                   arctan_log_transform=True, n_flag=False,
                                                   t_flag=True, a_flag=True)
    new_param = param.new_modified(arctan_log_transform=False, R=3)
    assert (not new_param.arctan_log_transform) and param.arctan_log_transform
    assert (new_param.R == 3) and param.R == 5

    with pytest.raises(AssertionError):
        new_param = param.new_modified(k_emb=16)

    with pytest.raises(AssertionError):
        new_param = param.new_modified(aggregate_neighbors=False)


@pytest.mark.parametrize("k_emb, alt", [(32, True), (64, False), (128, True), (256, False)])
def test_digraphwave(k_emb, alt):
    d = 3
    beta = 3
    ell = 3
    R = 5
    adj = tgraphs.source_star_adj(d, beta, ell, directed=True)
    param = digw.DigraphwaveHyperparameters.create(num_nodes=adj.shape[0], num_edges=adj.nnz, R=R, k_emb=k_emb,
                                                   arctan_log_transform=alt, n_flag=False, t_flag=True, a_flag=True)

    embeddings = digw.digraphwave(adj, param)
    assert embeddings.shape[0] == adj.shape[0]
    assert embeddings.shape[1] <= k_emb

    embeddings_selected = digw.digraphwave(adj, param, node_indices=[2, 10, 3])
    assert np.allclose(embeddings[[2, 10, 3], :], embeddings_selected)


@pytest.mark.parametrize("k_emb, alt", [(32, True), (64, False), (128, True), (256, False)])
def test_digraphwave_n(k_emb, alt):
    d = 3
    beta = 3
    ell = 3
    R = 5
    adj = tgraphs.source_star_adj(d, beta, ell, directed=True)
    param = digw.DigraphwaveHyperparameters.create(num_nodes=adj.shape[0], num_edges=adj.nnz, R=R, k_emb=k_emb,
                                                   arctan_log_transform=alt, n_flag=True, t_flag=True, a_flag=True)

    embeddings = digw.digraphwave(adj, param)
    assert embeddings.shape[0] == adj.shape[0]
    assert embeddings.shape[1] <= k_emb

    embeddings_selected = digw.digraphwave(adj, param, node_indices=[2, 10, 3])
    assert np.allclose(embeddings[[2, 10, 3], :], embeddings_selected)


@pytest.mark.parametrize("k_emb, alt", [(32, True), (64, False), (128, True), (256, False)])
def test_digraphwave_dense(k_emb, alt):
    d = 3
    beta = 3
    ell = 3
    R = 5
    adj = tgraphs.source_star_adj(d, beta, ell, directed=True)
    param = digw.DigraphwaveHyperparameters.create(num_nodes=adj.shape[0], num_edges=adj.nnz, R=R, k_emb=k_emb,
                                                   arctan_log_transform=alt, n_flag=False, t_flag=True, a_flag=True)

    embeddings = digw.digraphwave(torch.from_numpy(adj.toarray()).to(torch.bool), param)
    assert embeddings.shape[0] == adj.shape[0]
    assert embeddings.shape[1] <= k_emb

    embeddings_selected = digw.digraphwave(torch.from_numpy(adj.toarray()), param,
                                           node_indices=torch.tensor([2, 10, 3]))
    assert np.allclose(embeddings[[2, 10, 3], :], embeddings_selected)
