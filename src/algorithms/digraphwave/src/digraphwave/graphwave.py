from typing import Sequence, Union
from functools import partialmethod
import dataclasses as dc
import numpy as np
import scipy.sparse as sp
import torch
import tqdm.auto as tqdm
import joblib

import digraphwave.utils as utils
import digraphwave.expm as expm
import digraphwave.wavelets2embeddings as w2e
import digraphwave.aggregation as diagg


@dc.dataclass(frozen=True)
class GraphwaveHyperparameters(utils.Parameters):
    arctan_log_transform: bool

    def __post_init__(self):
        assert self.k_tau > 0
        assert self.k_phi > 0
        assert self.k_emb == 2 * self.k_tau * self.k_phi * (2 ** self.aggregate_neighbors)

    @classmethod
    def legacy(cls, *, num_nodes: int, num_edges: int, k_emb: int, order: int = 30,
               available_memory: int = 8, batch_size: int = None, dtype: torch.dtype = None):

        eta_max = 0.95
        eta_min = 0.80

        k_tau = 2
        l1 = 1.0 / num_nodes
        smax = -np.log(eta_min) * np.sqrt(0.5 / l1)
        smin = -np.log(eta_max) * np.sqrt(0.5 / l1)

        k_phi = int(k_emb / 2 / k_tau)
        if batch_size is None:
            bytes_per_element = torch.ones(1, dtype=dtype).element_size()
            batch_size = cls.get_auto_batch_size(memory_available=available_memory, num_nodes=num_nodes,
                                                 num_edges=num_edges,
                                                 k_tau=k_tau, k_emb=k_emb, bytes_per_element=bytes_per_element)

        out = cls(
            thresholds="flat",
            R=None,
            tau_start=smin,
            tau_stop=smax,
            k_tau=k_tau,
            k_phi=k_phi,
            k_emb=k_emb,
            num_nodes=num_nodes,
            num_edges=num_edges,
            order=order,
            arctan_log_transform=False,
            aggregate_neighbors=False,
            batch_size=batch_size,
            dtype=dtype,
            char_fun_step=1.
        )
        return out

    @classmethod
    def as_digraphwave(cls, *, num_nodes: int, num_edges: int, R: int, k_emb: int, arctan_log_transform: bool = False,
                       a_flag: bool = True, order: int = 40, available_memory: int = 8, batch_size: int = None,
                       dtype: torch.dtype = None):
        if dtype is None:
            dtype = torch.float64

        emb_dim_multiplier = 2 * (2 ** a_flag)
        if R > 1:
            k_tau = int(
                ((k_emb / emb_dim_multiplier) ** (1 / 3)) + 0.01
            )  # add 0.01 to avoid flooring due to inprecision
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
            aggregate_neighbors=a_flag,
            batch_size=batch_size,
            dtype=dtype,
            char_fun_step=1.
        )
        return out


def graphwave(adj: Union[sp.spmatrix, torch.Tensor], param: GraphwaveHyperparameters,
              node_indices: Sequence[int] = None, device_ids: Sequence[int] = None, verbose: bool = False):
    tqdm.tqdm.__init__ = partialmethod(tqdm.tqdm.__init__, disable=not verbose)

    feature_aggregator = None
    node_indices = np.asarray(node_indices) if node_indices is not None else None
    if param.aggregate_neighbors:
        feature_aggregator = diagg.FeatureAggregator(adj, selected_nodes=node_indices)
        node_indices = feature_aggregator.required_node_indices

    embeddings = graphwave_core(adj, param, node_indices, device_ids=device_ids)
    embeddings = embeddings.cpu().numpy()

    if feature_aggregator is not None:
        embeddings = feature_aggregator.create_enhanced_features(embeddings)

    return embeddings


def graphwave_core(adj: Union[sp.spmatrix, torch.Tensor], param: GraphwaveHyperparameters,
                   node_indices: Sequence[int] = None, thresholds: torch.Tensor = None,
                   device_ids: Sequence[int] = None):
    devices = utils.device_ids2devices(device_ids)
    node_indices = utils.check_node_indices(param.num_nodes, node_indices)

    if thresholds is None:
        thresholds = param.get_thresholds(adj)

    X = expm.make_polynomial_term(adj, dtype=param.dtype)
    taus = param.get_taus()

    coeffs = torch.stack(
        [expm._chebyshev_expm_coeffs(tau, param.order, device=devices[0], dtype=param.dtype) for tau in taus],
        dim=0)
    expm_obj = expm.ExpmObject(X=X, coeffs=coeffs, thresholds=thresholds)

    if len(devices) == 1:
        embeddings = _run_on_single_device(expm_obj, param, device=devices[0], node_indices=node_indices)
    else:
        meta_batches = torch.tensor_split(node_indices, len(devices))
        embeddings = joblib.Parallel(prefer='threads', n_jobs=len(devices))(
            joblib.delayed(_run_on_single_device)(expm_obj, param=param, device=device, node_indices=meta_batch)
            for meta_batch, device in zip(meta_batches, devices)
        )
        embeddings = torch.cat([emb.to(devices[0]) for emb in embeddings], dim=0)
    return embeddings


def _run_on_single_device(expm_obj: expm.ExpmObject, param: GraphwaveHyperparameters,
                          device: torch.device, node_indices: Sequence[int] = None):
    node_indices = utils.check_node_indices(param.num_nodes, node_indices)
    batches = utils.make_batches(node_indices, batch_size=param.batch_size)
    expm_obj = expm_obj.to(device)
    embeddings = []
    for batch in tqdm.tqdm(batches):
        batch = batch.to(device)
        embeddings.append(
            _graphwave_core_child(expm_obj, batch=batch, param=param, device=device))
    return torch.cat(embeddings, dim=0)


def _graphwave_core_child(expm_obj: expm.ExpmObject, batch: torch.LongTensor, param: GraphwaveHyperparameters,
                          device: torch.device):
    use_sparse = True  # This seems to be the fastest option both for cpu and gpu
    maxval = 1.
    minval = 0.
    char_fun_t = param.char_fun_step * torch.arange(1, param.k_phi + 1, step=1, device=device, dtype=param.dtype)
    embeddings = []

    expm_batch = expm._chebyshev_approximation(expm_obj.X, batch_indices=batch, coeffs=expm_obj.coeffs)

    for k in range(param.k_tau):
        psi = w2e.threshold_batch_inplace(expm_batch[k], expm_obj.thresholds[batch], as_sparse=use_sparse)
        psi = w2e.transform(psi, maxval=maxval, minval=minval, log_arctan=param.arctan_log_transform)
        embeddings.append(w2e.wavelets2embeddings(psi, char_fun_t))

    embeddings = torch.cat(embeddings, dim=1)
    return embeddings
