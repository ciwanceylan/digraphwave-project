import dataclasses as dc
import json
import warnings

import numba as nb
import numpy as np
from scipy import sparse as sp


@dc.dataclass(frozen=True)
class RefexParam:
    """
    Args:
        max_emb_size: Maximum size of refex embeddings. Used to calculate max recursion steps.
        use_weights: Append features using weights
        prune_tol: The feature pruning tolerance.
        columns2keep: Specification of which columns to keep in pruning step.
            Needed for reproducing refex feature on new graphs after having run it on a 'training' graph.
    """
    max_emb_size: int
    use_weights: bool
    prune_tol: float = 1e-5
    columns2keep: nb.types.DictType(nb.int64, nb.bool_[::1]) = None

    def save(self, path):
        data = dict(max_emb_size=self.max_emb_size, use_weights=self.use_weights, prune_tol=self.prune_tol)
        data['columns2keep'] = {i: val.tolist() for i, val in self.columns2keep.items()}
        with open(path, 'w') as fp:
            json.dump(data, fp)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as fp:
            data = json.load(fp)
        columns2keep = nb.typed.Dict.empty(
            key_type=nb.int64,
            value_type=nb.bool_[::1],
        )
        for i, cols2keep in data['columns2keep'].items():
            columns2keep[int(i)] = np.asarray(cols2keep, dtype=bool, order='C')
        data['columns2keep'] = columns2keep
        return cls(**data)


def do_feature_recursion(features: np.ndarray, adj_dict: nb.typed.Dict, max_steps: int, tol: float,
                         columns2keep: nb.types.DictType(nb.int64, nb.bool_[::1]) = None):
    """ Enhance features through recursive aggregation using mean and sum of neighbour features.
    Features are only added if they are linearly independent of existing features.
    Args:
        features: Base features to enhance
        adj_dict: Numba dict specifying neighbours of each node.
        max_steps: The maximum number of recursion steps
        tol: Tolerance for adding new features
        columns2keep:

    Returns:
        features: Matrix of enhanced features, [n x *]
    """
    # adj_dict, _ = sparse2adj_dict(adj.maximum(adj.T), as_numba_dict=True)
    if columns2keep is None:
        features, columns2keep = _feature_recursion(features.astype(np.float64), adj_dict, max_steps, tol)
    else:
        features = _repeat_feature_recursion(features.astype(np.float64), adj_dict, columns2keep)
    return features, columns2keep


@nb.jit(nb.float64[:](nb.float64[:, :]), nopython=True, nogil=True, parallel=True)
def nb_max(array):
    out = np.zeros(array.shape[0])
    for i in nb.prange(array.shape[0]):
        out[i] = np.max(array[i])

    return out


with warnings.catch_warnings():
    # Ignore type safety caused by casting uint64 to int64
    warnings.simplefilter('ignore', category=nb.NumbaTypeSafetyWarning)


    @nb.jit(nb.types.Tuple((nb.float64[:, :], nb.types.DictType(nb.int64, nb.bool_[::1])))(
        nb.float64[:, :], nb.types.DictType(nb.int64, nb.int64[::1]), nb.int64, nb.float64
    ), nopython=True, nogil=True, parallel=True)
    def _feature_recursion(features: np.ndarray, adj_dict: nb.typed.Dict, max_steps: int, tol: float):
        num_nodes = features.shape[0]
        columns_to_keep = {}
        for step in range(max_steps):
            num_new_features = 2 * features.shape[1]
            if features.shape[1] + num_new_features > num_nodes:
                break
            new_features = np.zeros((num_nodes, num_new_features))

            for v in nb.prange(num_nodes):
                num_neigh = len(adj_dict[v])
                if num_neigh == 0:
                    continue
                new_features[v] = np.concatenate((np.sum(features[adj_dict[v], :], axis=0) / num_neigh,
                                                  np.sum(features[adj_dict[v], :], axis=0)))
            new_features = new_features / np.sqrt(np.sum(np.power(new_features, 2), axis=0))

            features = np.concatenate((features, new_features), axis=1)
            _, r = np.linalg.qr(features)
            dependence_coeff = np.abs(np.diag(r))
            thresholds = tol * nb_max(np.abs(r).T)
            # thresholds = tol * dependence_coeff.max()

            keep_columns = dependence_coeff > thresholds
            if np.all(keep_columns[-num_new_features:] == 0):
                features = features[:, :-num_new_features]
                break  # Break if no new features are added, keeping only the previous features

            features = features[:, keep_columns]
            columns_to_keep[step] = keep_columns
        return features, columns_to_keep


    @nb.jit((nb.float64[:, :])(
        nb.float64[:, :], nb.types.DictType(nb.int64, nb.int64[::1]), nb.types.DictType(nb.int64, nb.bool_[::1])
    ), nopython=True, nogil=True, parallel=True)
    def _repeat_feature_recursion(features: np.ndarray, adj_dict: nb.typed.Dict, columns2keep: nb.typed.Dict):
        num_nodes = features.shape[0]
        for step, cols2keep in columns2keep.items():
            num_new_features = 2 * features.shape[1]
            new_features = np.zeros((num_nodes, num_new_features))

            for v in nb.prange(num_nodes):
                num_neigh = len(adj_dict[v])
                if num_neigh == 0:
                    continue
                new_features[v] = np.concatenate((np.sum(features[adj_dict[v], :], axis=0) / num_neigh,
                                                  np.sum(features[adj_dict[v], :], axis=0)))
            new_features = new_features / np.sqrt(np.sum(np.power(new_features, 2), axis=0))

            features = np.concatenate((features, new_features), axis=1)

            features = features[:, cols2keep]
        return features


@nb.jit(nb.float64[:, :](nb.types.DictType(nb.int64, nb.int64[::1]), nb.types.DictType(nb.int64, nb.int64[::1]),
                         nb.types.DictType(nb.int64, nb.float64[::1]), nb.types.DictType(nb.int64, nb.float64[::1])),
        nopython=True, nogil=True)
def _extract_egonet_features(out_adj_dict: nb.typed.Dict, in_adj_dict: nb.typed.Dict,
                             out_weights: nb.typed.Dict, in_weights: nb.typed.Dict):
    num_nodes = len(out_adj_dict)

    features = np.zeros((num_nodes, 6), dtype=np.float64)

    for v in range(num_nodes):
        egonet = set(np.unique(np.concatenate((out_adj_dict[v], in_adj_dict[v]))))
        egonet.add(v)
        num_internal_edges = 0
        num_out_edges = 0
        num_in_edges = 0

        internal_weight_sum = 0
        out_weight_sum = 0
        in_weight_sum = 0
        for neigh in egonet:
            for neigh_neigh, weight in zip(out_adj_dict[neigh], out_weights[neigh]):
                if neigh_neigh in egonet:
                    num_internal_edges += 1
                    internal_weight_sum += weight
                else:
                    num_out_edges += 1
                    out_weight_sum += weight

            for neigh_neigh, weight in zip(in_adj_dict[neigh], in_weights[neigh]):
                if neigh_neigh in egonet:
                    num_internal_edges += 1
                    internal_weight_sum += weight
                else:
                    num_in_edges += 1
                    in_weight_sum += weight

        features[v, 0] = num_internal_edges / 2
        features[v, 1] = num_out_edges
        features[v, 2] = num_in_edges

        features[v, 3] = internal_weight_sum / 2
        features[v, 4] = out_weight_sum
        features[v, 5] = in_weight_sum
    return features


@nb.jit(nb.float64[:, :](nb.types.DictType(nb.int64, nb.int64[::1]), nb.types.DictType(nb.int64, nb.int64[::1])),
        nopython=True, nogil=True)
def _extract_egonet_features_no_weights(out_adj_dict: nb.typed.Dict, in_adj_dict: nb.typed.Dict):
    num_nodes = len(out_adj_dict)

    features = np.zeros((num_nodes, 3), dtype=np.float64)

    for v in range(num_nodes):
        egonet = set(np.unique(np.concatenate((out_adj_dict[v], in_adj_dict[v]))))
        egonet.add(v)
        num_internal_edges = 0
        num_out_edges = 0
        num_in_edges = 0

        for neigh in egonet:
            for neigh_neigh in out_adj_dict[neigh]:
                if neigh_neigh in egonet:
                    num_internal_edges += 1
                else:
                    num_out_edges += 1

            for neigh_neigh in in_adj_dict[neigh]:
                if neigh_neigh in egonet:
                    num_internal_edges += 1
                else:
                    num_in_edges += 1

        features[v, 0] = num_internal_edges / 2
        features[v, 1] = num_out_edges
        features[v, 2] = num_in_edges
    return features


def dense2adj_dict(array: np.ndarray, return_values=True):
    return dense2row_adj_dict(np.maximum(array, array.T), return_values=return_values)


def dense2row_adj_dict(array: np.ndarray, return_values=True):
    array = array.astype(np.float64)
    adj_dict = _dense2row_adj_dict(array)
    values = None
    if return_values:
        values = _dense_get_weights(array, adj_dict)
    return adj_dict, values


@nb.jit((nb.types.DictType(nb.int64, nb.int64[::1]))(nb.types.Array(nb.types.float64, 2, 'A')),
        nopython=True, nogil=True)
def _dense2row_adj_dict(array: np.ndarray):
    num_nodes = array.shape[0]
    adj_dict = {}
    for i in range(num_nodes):
        adj_dict[i] = np.nonzero(array[i, :])[0]
    return adj_dict


@nb.jit((nb.types.DictType(nb.int64, nb.float64[::1]))(
    nb.types.Array(nb.types.float64, 2, 'A', readonly=True),
    nb.types.DictType(nb.int64, nb.int64[::1])),
    nopython=True, nogil=True)
def _dense_get_weights(array: np.ndarray, adj_dict):
    num_nodes = array.shape[0]
    weights = {}
    for i in range(num_nodes):
        weights[i] = array[i][adj_dict[i]]
    return weights


def sparse2symmetric_adj_dict(adj: sp.spmatrix, return_weights=False, as_numba_dict=False):
    return sparse2adj_dict(adj.maximum(adj.T), return_weights=return_weights, as_numba_dict=as_numba_dict)


def sparse2adj_dict(adj: sp.spmatrix, return_weights=False, use_in_degrees=False, as_numba_dict=False):
    """ Get a adjacency dictionary representation of the graph defined by adj.

    Args:
        adj: Adjacency matrix with A[i,j] indicating an edge j -> i
        return_weights: Return a dictionary of the weights associated with each edge.
        use_in_degrees: Create the adjacency dict using incoming edges instead of outgoing.
        as_numba_dict: Create a numba Dict which can be used with nopython numba functions.

    Returns:
        targets: Dict of adjacent nodes for each node in the graph
        weights: Dict of edge weights for adjacent node. None if return_weights=False.
    """
    num_nodes = adj.shape[0]
    if use_in_degrees:
        adj = adj.tocsr()
    else:
        adj = adj.tocsc()

    adj.indices = adj.indices.astype(np.int64)
    adj.indptr = adj.indptr.astype(np.int64)

    if as_numba_dict:
        targets = nb.typed.Dict.empty(nb.int64, nb.int64[::1])
        for i in range(num_nodes):
            targets[i] = np.ascontiguousarray(adj.indices[adj.indptr[i]:adj.indptr[i + 1]])
    else:
        targets = {i: adj.indices[adj.indptr[i]:adj.indptr[i + 1]] for i in range(num_nodes)}

    if return_weights:
        adj.data = adj.data.astype(np.float64)
        if as_numba_dict:

            weights = nb.typed.Dict.empty(nb.int64, nb.float64[::1])
            for i in range(num_nodes):
                weights[i] = np.ascontiguousarray(adj.data[adj.indptr[i]:adj.indptr[i + 1]])
        else:
            weights = {i: adj.data[adj.indptr[i]:adj.indptr[i + 1]] for i in range(num_nodes)}
    else:
        weights = None
    return targets, weights
