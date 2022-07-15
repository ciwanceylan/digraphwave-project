from typing import Union
import warnings
import numpy as np
import numba as nb
import scipy.sparse as sp
import torch

from digraphwave.utils import sparse2dict


class FeatureAggregator:
    def __init__(self, adj: Union[sp.spmatrix, torch.Tensor], selected_nodes: np.ndarray = None):
        self.adj_dict = get_adj_dict(adj)
        self.selected_nodes = selected_nodes
        if selected_nodes is not None:
            self.required_node_indices = np.asarray(list(get_all_neighbours(self.adj_dict, selected_nodes)))
            self.node2featureindex = nb.typed.Dict.empty(nb.int64, nb.int64)
            for i, n in enumerate(self.required_node_indices):
                self.node2featureindex[n] = i
        else:
            self.required_node_indices = None
            self.node2featureindex = None

    def create_enhanced_features(self, features: np.ndarray):
        """ Create feature aggregation using mean of neighbour features.
        Args:
            features: Base features to enhance [num_nodes x D]
            adj_dict: Adjacency dictionary
            selected_node_indices: Nodes with the corresponding features

        Returns:
            : Matrix of aggregated features. Same shape as `features`
        """
        if self.selected_nodes is None:
            new_features = _feature_aggregation(features.astype(np.float64), self.adj_dict)
            features = np.concatenate((features, new_features), axis=1)
        else:
            new_features = _feature_aggregation_selected(features.astype(np.float64), self.adj_dict,
                                                         self.selected_nodes, self.node2featureindex)
            feature_indices = [self.node2featureindex[n] for n in self.selected_nodes]
            features = np.concatenate((features[feature_indices, :], new_features), axis=1)
        return features


with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=nb.NumbaTypeSafetyWarning)


    @nb.jit(nb.float64[:, :](nb.float64[:, :], nb.types.DictType(nb.int64, nb.int64[::1])),
            nopython=True, nogil=True, parallel=True)
    def _feature_aggregation(features: np.ndarray, adj_dict: nb.typed.Dict):
        num_nodes = features.shape[0]

        new_features = np.zeros((num_nodes, features.shape[1]))

        for v in nb.prange(num_nodes):
            num_neigh = len(adj_dict[v])
            if num_neigh == 0:
                continue
            new_features[v] = np.sum(features[adj_dict[v], :], axis=0) / num_neigh

        return new_features


    @nb.jit(nb.float64[:, :](nb.float64[:, :], nb.types.DictType(nb.int64, nb.int64[::1]),
                             nb.int64[:], nb.types.DictType(nb.int64, nb.int64)),
            nopython=True, nogil=True, parallel=True)
    def _feature_aggregation_selected(features: np.ndarray, adj_dict: nb.typed.Dict,
                                      selected_node_indices: np.ndarray, node2featureindex: nb.typed.Dict):
        num_nodes_to_enhance = len(selected_node_indices)

        new_features = np.zeros((selected_node_indices.shape[0], features.shape[1]))

        for v in nb.prange(num_nodes_to_enhance):
            node_index = selected_node_indices[v]
            num_neigh = len(adj_dict[node_index])
            if num_neigh == 0:
                continue
            feature_indices = np.asarray([node2featureindex[n] for n in adj_dict[node_index]])
            new_features[v] = np.sum(features[feature_indices, :], axis=0) / num_neigh

        return new_features


def get_adj_dict(adj: Union[sp.spmatrix, torch.Tensor]):
    if isinstance(adj, torch.Tensor):
        return dense2adj_dict(adj.numpy())[0]
    else:
        return sparse2dict(adj.maximum(adj.T), as_numba_dict=True)[0]


@nb.jit((nb.types.Set(nb.int64))(nb.types.DictType(nb.int64, nb.int64[::1]), nb.int64[::1]), nopython=True, nogil=True)
def get_all_neighbours(adj_dict: nb.typed.Dict, indices: np.ndarray):
    all_neighbours = set(indices)
    for i in indices:
        all_neighbours.update(set(adj_dict[i]))
    return all_neighbours


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
