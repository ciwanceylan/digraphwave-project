from typing import Union, Dict, Sequence
import numpy as np
import scipy.sparse as sp
import scipy.special as sspecial
import torch
import torch_sparse as tsp
import joblib
import tqdm.auto as tqdm

import digraphwave.utils as utils
import digraphwave.search as search


def log_source_star_expm(distances: np.ndarray, outdegrees: np.ndarray, taus: Union[np.ndarray, float],
                         beta: float, d: float) -> np.ndarray:
    """
    Compute the heat diffusion for a source star graph using the analytical expression from 'ref'.
    Args:
        distances: Directed shortest path distances from the central node to other nodes
        outdegrees: Out-degree of other nodes
        taus: Time values tau to evaluate
        beta: Branch factor to use
        d: Out-degree of the central node

    Returns:
        expm_values: Matrix with source-star expm values. [len(taus) x len(distances)]
    """
    beta = max(beta, 1)  # Avoid nan in case where beta = 0
    taus = np.atleast_1d(taus)
    num_taus = len(taus)
    if d < 1:
        expm_values = np.zeros(shape=(len(taus), len(distances)), dtype=np.float64)
    else:
        expm_values = _log_source_star_expm(distances, outdegrees, taus, beta, d)

    if num_taus == 1:
        expm_values = expm_values[0]
    return expm_values


def _log_source_star_expm(distances: np.ndarray, outdegrees: np.ndarray, taus: Union[np.ndarray, float],
                          beta: float, d: float) -> np.ndarray:
    """ Compute the heat diffusion for a source star graph using the analytical expression from 'ref'.
    Args:
        distances: Directed shortest path distances from the central node to other nodes
        outdegrees: Out-degree of other nodes
        taus: Time values tau to evaluate
        beta: Branch factor to use
        d: Out-degree of the central node

    Returns:
        expm_values: Matrix with source-star expm values. [len(taus) x len(distances)]
    """
    unreachable = np.isinf(distances)
    source_node_mask = (distances == 0) & (outdegrees > 0)
    sink_node_mask = (outdegrees == 0) & (distances > 0) & ~unreachable
    source_and_sink = (outdegrees == 0) & (distances == 0)
    rest_mask = ~(source_node_mask | sink_node_mask | source_and_sink | unreachable)

    expm_values = np.empty(shape=(len(taus), len(distances)), dtype=np.float64)
    expm_values[:, unreachable] = -np.inf
    expm_values[:, source_and_sink] = 0
    expm_values[:, source_node_mask] = -taus
    expm_values[:, sink_node_mask] = (
            np.log(sspecial.gammainc(distances[sink_node_mask], taus[:, None]))
            - (distances[sink_node_mask] - 1) * np.log(beta) - np.log(d)
    )
    expm_values[:, rest_mask] = (
            utils.log_poly_exp_factorial(taus, distances[rest_mask])
            - (distances[rest_mask] - 1) * np.log(beta) - np.log(d)
    )
    return expm_values


def _calc_log_normalization_values(adj: sp.spmatrix, tau: float,
                                   sources: Sequence[int], targets_per_source: Dict[int, np.ndarray],
                                   n_jobs: int = -2):
    adj_dict, _ = utils.sparse2dict(adj, as_numba_dict=True)
    outdegrees = utils.calc_out_degrees(adj, weighted=False)
    beta_i = utils.calc_ss_beta_per_node(adj)

    values = joblib.Parallel(n_jobs=n_jobs, prefer='threads')(
        joblib.delayed(_log_norm_values_child)(
            adj_dict, s, targets_per_source[s], outdegrees[targets_per_source[s]],
            tau=tau, beta=beta_i[s], d=outdegrees[s])
        for s in tqdm.tqdm(sources)
    )
    values = dict(zip(sources, values))

    values_adj = utils.mapping2sparse(targets_per_source, weights=values, num_targets=adj.shape[0],
                                      sources=sources)
    return values_adj


def _log_norm_values_child(adj_dict, source, targets, outdegrees, tau, beta, d):
    distances = search.shortest_path_with_targets(adj_dict, source, targets, max_degree=None)
    return log_source_star_expm(distances, outdegrees, tau, beta, d).astype(np.float64)


def calc_log_normalization_values(psi: Union[tsp.SparseTensor, torch.Tensor], adj: sp.spmatrix,
                                  tau: float, sources: Sequence[int]):
    targets_per_source, _ = utils.sparse2dict(psi.cpu().to_scipy(layout='csc', dtype=np.float64), sources=sources)

    log_norm_psi = _calc_log_normalization_values(adj, tau, sources, targets_per_source)
    log_norm_psi = tsp.SparseTensor.from_scipy(log_norm_psi)

    return log_norm_psi


def normalize(psi: Union[tsp.SparseTensor, torch.Tensor], adj: sp.spmatrix, tau: float,
              sources: Sequence[int]):
    num_sources = psi.size(1)
    if len(sources) != num_sources:
        raise ValueError(f"'sources' with {len(sources)} should be the same length as number "
                         f"of columns in 'psi' {num_sources}")

    is_dense = not isinstance(psi, tsp.SparseTensor)
    if is_dense:
        device = psi.device
        dtype = psi.dtype
        psi = tsp.SparseTensor.from_dense(psi)
    else:
        device = psi.device()
        dtype = psi.dtype()

    log_norm_psi = calc_log_normalization_values(psi, adj, tau, sources)
    log_norm_psi = log_norm_psi.to(device)

    normalized_psi = _normalize_sparse(psi, log_norm_psi)

    if is_dense:
        normalized_psi = normalized_psi.to_dense(dtype)

    return normalized_psi


def _normalize_sparse(psi: tsp.SparseTensor, log_norm_psi: tsp.SparseTensor, copy: bool = False) -> tsp.SparseTensor:
    if copy:
        psi = psi.set_value(torch.exp(torch.log(psi.storage.value()) - log_norm_psi.storage.value()), layout='coo')
    else:
        psi = psi.set_value_(torch.exp(torch.log(psi.storage.value()) - log_norm_psi.storage.value()), layout='coo')
    return psi
