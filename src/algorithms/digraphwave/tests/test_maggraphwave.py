import pytest
import numpy as np

import digraphwave.test_graphs as tgraphs
import digraphwave.maggraphwave as maggw


@pytest.mark.parametrize("k_emb,parmtype",
                         [(32, "digraphwave"), (64, "digraphwave"), (128, "digraphwave"), (64, "furutani_et_al"),
                          (128, "furutani_et_al")])
def test_maggraphwave(k_emb, parmtype):
    d = 3
    beta = 3
    ell = 3
    R = 5
    adj = tgraphs.source_star_adj(d, beta, ell, directed=True)
    if parmtype == "digraphwave":
        param = maggw.MaggraphwaveHyperparameters.as_digraphwave(num_nodes=adj.shape[0], num_edges=adj.nnz,
                                                                 R=R,
                                                                 k_emb=k_emb,
                                                                 batch_size=32,
                                                                 a_flag=True
                                                                 )
    else:
        param = maggw.MaggraphwaveHyperparameters.furutani_et_al(num_nodes=adj.shape[0], num_edges=adj.nnz, k_emb=k_emb)

    embeddings = maggw.maggraphwave(adj, param)
    assert embeddings.shape[0] == adj.shape[0]
    assert embeddings.shape[1] <= k_emb
