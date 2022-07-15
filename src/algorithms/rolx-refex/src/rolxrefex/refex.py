import numpy as np
import scipy.sparse as sp
from typing import Union

from rolxrefex.core import RefexParam
import rolxrefex.refex_sparse as refex_sparse
import rolxrefex.refex_dense as refex_dense


def refex(adj: Union[sp.spmatrix, np.ndarray], param: RefexParam):
    """ Computes ReFeX features.
    Feature pruning uses QR factorisation and removes feature columns for which the diagonal element of R is smaller
    than `prune_tol * diag(R).max()`.

    Args:
        adj: Weighted adjacency matrix with A[i,j] indicating an edge j -> i. Assumes non-negative weights.
        param: refex parameters as a RefexParam object
    Returns:
        features: Matrix of features. [n x 10] if use_weights, else  [n x 5]
    """
    if isinstance(adj, np.ndarray):
        out = refex_dense.refex_dense(adj, param)
    elif isinstance(adj, sp.spmatrix):
        out = refex_sparse.refex_sparse(adj, param)
    else:
        raise TypeError(f"Unexpected type for adj {type(adj)}")

    return out