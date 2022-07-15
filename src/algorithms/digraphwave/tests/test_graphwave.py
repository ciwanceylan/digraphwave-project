import pytest
import numpy as np
import torch

import digraphwave.test_graphs as tgraphs
import digraphwave.graphwave as gw


@pytest.mark.parametrize("k_emb,parmtype", [(32, "digraphwave"), (64, "digraphwave"), (128, "digraphwave"),
                                            (64, "legacy"), (128, "legacy")])
def test_graphwave(k_emb, parmtype):
    d = 3
    beta = 3
    ell = 3
    R = 5
    adj = tgraphs.source_star_adj(d, beta, ell, directed=True)
    if parmtype == "digraphwave":
        param = gw.GraphwaveHyperparameters.as_digraphwave(num_nodes=adj.shape[0], num_edges=adj.nnz,
                                                           R=R,
                                                           k_emb=k_emb,
                                                           arctan_log_transform=True,
                                                           a_flag=True
                                                           )
    else:
        param = gw.GraphwaveHyperparameters.legacy(num_nodes=adj.shape[0], num_edges=adj.nnz, k_emb=k_emb)

    embeddings = gw.graphwave(adj, param)
    assert embeddings.shape[0] == adj.shape[0]
    assert embeddings.shape[1] <= k_emb

    embeddings_selected = gw.graphwave(adj, param, node_indices=[2, 10, 3])
    assert np.allclose(embeddings[[2, 10, 3], :], embeddings_selected)


@pytest.mark.parametrize("k_emb,parmtype", [(32, "digraphwave"), (64, "digraphwave"), (128, "digraphwave"),
                                            (64, "legacy"), (128, "legacy")])
def test_graphwave_dense(k_emb, parmtype):
    d = 3
    beta = 3
    ell = 3
    R = 5
    adj = tgraphs.source_star_adj(d, beta, ell, directed=True)
    if parmtype == "digraphwave":
        param = gw.GraphwaveHyperparameters.as_digraphwave(num_nodes=adj.shape[0], num_edges=adj.nnz,
                                                           R=R,
                                                           k_emb=k_emb,
                                                           arctan_log_transform=True,
                                                           a_flag=True
                                                           )
    else:
        param = gw.GraphwaveHyperparameters.legacy(num_nodes=adj.shape[0], num_edges=adj.nnz, k_emb=k_emb)

    embeddings = gw.graphwave(torch.from_numpy(adj.toarray()), param)
    assert embeddings.shape[0] == adj.shape[0]
    assert embeddings.shape[1] <= k_emb

    embeddings_selected = gw.graphwave(torch.from_numpy(adj.toarray()), param, node_indices=[2, 10, 3])
    assert np.allclose(embeddings[[2, 10, 3], :], embeddings_selected)
