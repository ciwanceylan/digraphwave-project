from typing import Dict, Sequence
import warnings
import scipy.sparse as sp
import numpy as np
import dataclasses as dc
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kendalltau
from sklearn.preprocessing import StandardScaler


@dc.dataclass(frozen=True)
class MaxsimCorrelation:
    kendal_euclidean: float
    kendal_cosine: float


def maxsim_structural_distances(adj: sp.spmatrix,
                                metrics=('euclidean', 'cosine'), unweighted: bool = False,
                                as_squareform: bool = False, **pdist_kwargs):
    embs = maxsim_embeddings(adj, unweighted)
    structural_distances = {}
    for metric in metrics:
        structural_distances_ = pdist(embs, metric=metric, **pdist_kwargs)
        if as_squareform:
            structural_distances_ = squareform(structural_distances_)
        structural_distances[metric] = structural_distances_
    return structural_distances


def maxsim_embeddings(adj: sp.spmatrix, unweighted: bool = False) -> np.ndarray:
    num_nodes = adj.shape[0]
    geodesic_distances = sp.csgraph.shortest_path(adj, unweighted=unweighted)
    with np.errstate(divide='ignore'):
        recip = 1. / geodesic_distances

    embs = []
    for i in range(num_nodes):
        col = np.sort(recip[:, i])[:-1]  # Remove last element since it is inf
        row = np.sort(recip[i, :])[:-1]  # Remove last element since it is inf
        embs.append(np.concatenate((col, row)))
    embs = np.stack(embs, axis=0)
    return embs


def evaluate_maxsim_kendal_coeffs(maxsim_euclidean_distances: np.ndarray, maxsim_cosine_distances: np.ndarray,
                                  embeddings: np.ndarray, which_nodes: Sequence[int]):
    embeddings = StandardScaler().fit_transform(embeddings)[which_nodes, :]

    emb_distances = pdist(embeddings, metric="euclidean")
    with warnings.catch_warnings():
        # Ignore overflow warning for calculation of p-value which is not used
        warnings.simplefilter('ignore', category=RuntimeWarning)
        tau_euclidean, _ = kendalltau(maxsim_euclidean_distances, emb_distances)

    emb_distances = pdist(embeddings, metric="cosine")
    with warnings.catch_warnings():
        # Ignore overflow warning for calculation of p-value which is not used
        warnings.simplefilter('ignore', category=RuntimeWarning)
        tau_cosine, _ = kendalltau(maxsim_cosine_distances, emb_distances)
    return MaxsimCorrelation(kendal_euclidean=tau_euclidean, kendal_cosine=tau_cosine)
