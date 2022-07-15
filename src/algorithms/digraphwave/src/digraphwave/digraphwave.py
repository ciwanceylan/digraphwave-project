from typing import Sequence, Union
from functools import partialmethod
import dataclasses as dc
import numpy as np
import scipy.sparse as sp
import torch
# import torch.nn as nn
# import torch_sparse as tsp
import tqdm.auto as tqdm
import joblib

import digraphwave.utils as utils
import digraphwave.expm as expm
import digraphwave.wavelets2embeddings as w2e
import digraphwave.aggregation as diagg
import digraphwave.ss_norm as ss_norm


@dc.dataclass(frozen=True)
class DigraphwaveHyperparameters(utils.Parameters):
    arctan_log_transform: bool
    ss_normalize: bool
    transpose_embeddings: bool

    def __post_init__(self):
        assert self.k_tau > 0
        assert self.k_phi > 0
        assert (self.k_emb == 2 * self.k_tau * self.k_phi * (2 ** self.aggregate_neighbors) *
                (2 ** self.transpose_embeddings))
        # arclog transform forbidden needed when normalising
        assert (self.ss_normalize and not self.arctan_log_transform) or not self.ss_normalize

    @classmethod
    def create(cls, *, num_nodes: int, num_edges: int, R: int, k_emb: int, arctan_log_transform: bool = False,
               t_flag: bool = True, a_flag: bool = True, n_flag: bool = False,
               order: int = 40, available_memory: int = 8, batch_size: int = None,
               dtype: torch.dtype = None):
        if dtype is None:
            dtype = torch.float64

        emb_dim_multiplier = 2 * (2 ** t_flag) * (2 ** a_flag)
        if R > 1:
            k_tau = int(
                ((k_emb / emb_dim_multiplier) ** (1 / 3)) + 0.01)  # add 0.01 to avoid flooring due to inprecision
        else:
            k_tau = 1

        k_phi = int(k_emb / emb_dim_multiplier / k_tau)
        k_emb = emb_dim_multiplier * k_tau * k_phi

        tau_start = 1
        tau_stop = R

        if batch_size is None:
            bytes_per_element = torch.ones(1, dtype=dtype).element_size()
            batch_size = cls.get_auto_batch_size(memory_available=available_memory, num_nodes=num_nodes,
                                                 num_edges=num_edges,
                                                 k_tau=k_tau, k_emb=k_emb, bytes_per_element=bytes_per_element)

        arctan_log_transform = arctan_log_transform and not n_flag  # arclog transform forbidden needed when normalising

        out = cls(
            thresholds="digraphwave",
            R=R,
            tau_start=tau_start,
            tau_stop=tau_stop,
            k_tau=k_tau,
            k_phi=k_phi,
            k_emb=k_emb,
            num_nodes=num_nodes,
            num_edges=num_edges,
            order=order,
            arctan_log_transform=arctan_log_transform,
            ss_normalize=n_flag,
            transpose_embeddings=t_flag,
            aggregate_neighbors=a_flag,
            batch_size=batch_size,
            dtype=dtype,
            char_fun_step=1.
        )
        return out


def digraphwave(adj: Union[sp.spmatrix, torch.Tensor], param: DigraphwaveHyperparameters,
                node_indices: Sequence[int] = None, device_ids: Sequence[int] = None, verbose: bool = False):
    tqdm.tqdm.__init__ = partialmethod(tqdm.tqdm.__init__, disable=not verbose)

    feature_aggregator = None
    node_indices = np.asarray(node_indices) if node_indices is not None else None
    if param.aggregate_neighbors:
        feature_aggregator = diagg.FeatureAggregator(adj, selected_nodes=node_indices)
        node_indices = feature_aggregator.required_node_indices

    embeddings = digraphwave_core(adj, param, node_indices, device_ids=device_ids).cpu().numpy()

    if param.transpose_embeddings:
        adjT = adj.T.tocsc() if isinstance(adj, sp.spmatrix) else adj.t()
        t_embeddings = digraphwave_core(adjT, param, node_indices, device_ids=device_ids).cpu().numpy()
        embeddings = np.concatenate((embeddings, t_embeddings), axis=1)

    if feature_aggregator is not None:
        embeddings = feature_aggregator.create_enhanced_features(embeddings)

    return embeddings


def digraphwave_core(adj: Union[sp.spmatrix, torch.Tensor], param: DigraphwaveHyperparameters,
                     node_indices: Sequence[int] = None, thresholds: torch.Tensor = None,
                     device_ids: Sequence[int] = None):
    devices = utils.device_ids2devices(device_ids)
    node_indices = utils.check_node_indices(param.num_nodes, node_indices)

    if thresholds is None:
        thresholds = param.get_thresholds(adj)

    X = expm.make_polynomial_term(adj, dtype=param.dtype)
    taus = param.get_taus()
    coeffs = torch.stack(
        [expm._taylor_expm_coeffs(tau, param.order, device=devices[0], dtype=param.dtype) for tau in taus],
        dim=0)
    expm_obj = expm.ExpmObject(X=X, coeffs=coeffs, thresholds=thresholds)

    if len(devices) == 1:
        embeddings = _run_on_single_device(expm_obj, adj, param, device=devices[0], node_indices=node_indices)
    else:
        meta_batches = torch.tensor_split(node_indices, len(devices))
        embeddings = joblib.Parallel(prefer='threads', n_jobs=len(devices))(
            joblib.delayed(_run_on_single_device)(expm_obj, adj, param=param, device=device, node_indices=meta_batch)
            for meta_batch, device in tqdm.tqdm(zip(meta_batches, devices))
        )
        embeddings = torch.cat([emb.to(devices[0]) for emb in embeddings], dim=0)

    return embeddings


def _run_on_single_device(expm_obj: expm.ExpmObject, adj: sp.spmatrix, param: DigraphwaveHyperparameters,
                          device: torch.device, node_indices):
    batches = utils.make_batches(node_indices, batch_size=param.batch_size)
    expm_obj = expm_obj.to(device)
    embeddings = []
    for batch in tqdm.tqdm(batches):
        batch = batch.to(device)
        embeddings.append(
            _digraphwave_core_child(expm_obj, batch=batch, adj=adj, param=param, device=device))
    return torch.cat(embeddings, dim=0)


def _digraphwave_core_child(expm_obj, batch: torch.LongTensor,
                            adj: sp.spmatrix, param: DigraphwaveHyperparameters, device: torch.device):
    use_sparse = True  # This seems to be the fastest option both for cpu and gpu
    maxval = 10. if param.ss_normalize else 1.  # TODO solve maxval when normalising.
    minval = 0.
    char_fun_t = param.char_fun_step * torch.arange(1, param.k_phi + 1, step=1, device=device, dtype=param.dtype)
    taus = param.get_taus()
    embeddings = []

    expm_batch = expm._taylor_approximation(expm_obj.X, batch_indices=batch, coeffs=expm_obj.coeffs)

    for k in range(param.k_tau):
        psi = w2e.threshold_batch_inplace(expm_batch[k], expm_obj.thresholds[batch], as_sparse=use_sparse)
        if param.ss_normalize:
            psi = ss_norm.normalize(psi, adj, tau=taus[k], sources=batch.cpu().numpy())

        psi = w2e.transform(psi, maxval=maxval, minval=minval, log_arctan=param.arctan_log_transform)
        embeddings.append(w2e.wavelets2embeddings(psi, char_fun_t))

    embeddings = torch.cat(embeddings, dim=1)
    return embeddings
