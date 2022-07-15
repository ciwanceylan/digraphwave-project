import numpy as np
from scipy import sparse as sp

import rolxrefex.core as core
from rolxrefex.core import sparse2symmetric_adj_dict, sparse2adj_dict


def refex_sparse(adj: sp.spmatrix, param: core.RefexParam):
    """ Computes ReFeX features.
    Feature pruning uses QR factorisation and removes feature columns for which the diagonal element of R is smaller
    than `prune_tol * diag(R).max()`.

    Args:
        adj: Weighted adjacency matrix with A[i,j] indicating an edge j -> i. Assumes non-negative weights.
        param: refex parameters as a RefexParam object
    Returns:
        features: Matrix of features. [n x 10] if use_weights, else  [n x 5]
    """
    base_features = extract_base_features(adj, use_weights=param.use_weights)
    num_base_features = base_features.shape[1]
    max_steps = int(np.log2(1 + (float(param.max_emb_size) / num_base_features))) - 1

    adj_dict, _ = sparse2symmetric_adj_dict(adj, return_weights=False, as_numba_dict=True)
    features, columns_keeped = core.do_feature_recursion(base_features, adj_dict=adj_dict, max_steps=max_steps,
                                                         tol=param.prune_tol, columns2keep=param.columns2keep)
    reprod_param = core.RefexParam(max_emb_size=param.max_emb_size,
                                   use_weights=param.use_weights,
                                   prune_tol=param.prune_tol,
                                   columns2keep=columns_keeped)

    return features, reprod_param


def extract_base_features(adj: sp.spmatrix, use_weights=True):
    """ Extract node and egonet features for each node in the graph.

    Args:
        adj: Adjacency matrix with A[i,j] indicating an edge j -> i
        use_weights: Append features using weights
    Returns:
        features: Matrix of features. [n x 10] if use_weights, else  [n x 5]
    """
    node_feat = extract_node_features(adj, use_weights=use_weights)
    ego_feat = extract_egonet_features(adj, use_weights=use_weights)

    return np.concatenate((node_feat, ego_feat), axis=1)


def extract_node_features(adj: sp.spmatrix, use_weights=True):
    """ Extract node features for each node in the graph. Node features are in and out degrees, and their weighted
    counterparts is `use_weights`.

    Args:
        adj: Adjacency matrix with A[i,j] indicating an edge j -> i
        use_weights: Append weighted node degrees
    Returns:
        features: Matrix of node features. [n x 4] if use_weights, else  [n x 2]
    """
    num_nodes = adj.shape[0]
    nz_row, nz_col = adj.nonzero()

    out_degrees_index, out_degrees_ = np.unique(nz_col, return_counts=True)
    out_degrees = np.zeros(num_nodes, dtype=np.float64)
    out_degrees[out_degrees_index] = out_degrees_

    in_degrees_index, in_degrees_ = np.unique(nz_row, return_counts=True)
    in_degrees = np.zeros(num_nodes, dtype=np.float64)
    in_degrees[in_degrees_index] = in_degrees_

    if use_weights:
        out_sum_weight = adj.sum(axis=0).A1
        in_sum_weight = adj.sum(axis=1).A1
        features = np.stack((out_degrees, in_degrees, out_sum_weight, in_sum_weight), axis=1)
    else:
        features = np.stack((out_degrees, in_degrees), axis=1)
    return features


def extract_egonet_features(adj, use_weights=True):
    """ Extract egonet features for each node in the network.
    The egonet features are the number of edges in the egonet and the number of incoming and outgoing edges of the
    egonet. Sum of weights are used if `use_weights` is `True`.


    Args:
        adj: Adjacency matrix with A[i,j] indicating an edge j -> i
        use_weights: Append weighted egonet features.

    Returns:
        features: Egonet features. [n x 6] if use_weights else [n x 3]
    """
    out_adj_dict, out_weights = sparse2adj_dict(adj, return_weights=use_weights, as_numba_dict=True)
    in_adj_dict, in_weights = sparse2adj_dict(adj, use_in_degrees=True, return_weights=use_weights, as_numba_dict=True)

    if use_weights:
        features = core._extract_egonet_features(out_adj_dict, in_adj_dict, out_weights, in_weights)
    else:
        features = core._extract_egonet_features_no_weights(out_adj_dict, in_adj_dict)
    return features
