""" Currently only to be used for comparison """
from typing import Sequence
from functools import partialmethod
import dataclasses as dc
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import torch
import tqdm.auto as tqdm

import digraphwave.utils
import digraphwave.utils as utils
import digraphwave.expm as expm
import digraphwave.wavelets2embeddings as w2e
import digraphwave.aggregation as diagg


@dc.dataclass(frozen=True)
class MaggraphwaveHyperparameters(utils.Parameters):
    q: float

    def __post_init__(self):
        assert self.tau_start > 0
        assert self.tau_stop >= self.tau_start
        assert self.k_tau > 0
        assert self.k_phi > 0
        assert self.k_emb == 2 * self.k_tau * self.k_phi * (2 ** self.aggregate_neighbors)

    @classmethod
    def furutani_et_al(cls, *, num_nodes: int, num_edges: int, k_emb: int, order: int = 30, available_memory: int = 8,
                       batch_size: int = None):
        q = 0.02
        k_phi = 10

        tau_start = 2
        tau_stop = 20
        # k_tau = 181
        k_tau = int(k_emb / 2 / k_phi)

        k_emb = 2 * k_tau * k_phi

        if batch_size is None:
            batch_size = cls.get_auto_batch_size(memory_available=available_memory, num_nodes=num_nodes,
                                                 num_edges=num_edges,
                                                 k_tau=k_tau, k_emb=k_emb, bytes_per_element=16)

        out = cls(
            thresholds="flat",
            R=None,
            q=q,
            tau_start=tau_start,
            tau_stop=tau_stop,
            k_tau=k_tau,
            k_phi=k_phi,
            k_emb=k_emb,
            num_nodes=num_nodes,
            num_edges=num_edges,
            order=order,
            aggregate_neighbors=False,
            batch_size=batch_size,
            dtype=torch.complex128,
            char_fun_step=1.
        )
        return out

    @classmethod
    def as_digraphwave(cls, *, num_nodes: int, num_edges: int, R: int, k_emb: int, a_flag: bool, q: float = 0.02,
                       order: int = 40, available_memory: int = 8, batch_size: int = None):
        emb_dim_multiplier = 2 * (2 ** a_flag)
        k_tau = int(((k_emb / emb_dim_multiplier) ** (1 / 3)) + 0.01)  # add 0.01 to avoid flooring due to inprecision
        k_phi = int(k_emb / emb_dim_multiplier / k_tau)
        k_emb = emb_dim_multiplier * k_tau * k_phi

        tau_start = 1
        tau_stop = R

        if batch_size is None:
            batch_size = cls.get_auto_batch_size(memory_available=available_memory, num_nodes=num_nodes,
                                                 num_edges=num_edges,
                                                 k_tau=k_tau, k_emb=k_emb, bytes_per_element=16)

        out = cls(
            thresholds="digraphwave",
            R=R,
            q=q,
            tau_start=tau_start,
            tau_stop=tau_stop,
            k_tau=k_tau,
            k_phi=k_phi,
            k_emb=k_emb,
            num_nodes=num_nodes,
            num_edges=num_edges,
            order=order,
            aggregate_neighbors=a_flag,
            batch_size=batch_size,
            dtype=torch.complex128,
            char_fun_step=1.
        )
        return out


def maggraphwave(adj: sp.spmatrix, param: MaggraphwaveHyperparameters, node_indices: Sequence[int] = None,
                 verbose: bool = False):
    tqdm.tqdm.__init__ = partialmethod(tqdm.tqdm.__init__, disable=not verbose)
    feature_aggregator = None
    node_indices = np.asarray(node_indices) if node_indices is not None else None
    if param.aggregate_neighbors:
        feature_aggregator = diagg.FeatureAggregator(adj, selected_nodes=node_indices)
        node_indices = feature_aggregator.required_node_indices

    embeddings = maggraphwave_core(adj, param, node_indices)
    embeddings = embeddings.cpu().numpy()

    if feature_aggregator is not None:
        embeddings = feature_aggregator.create_enhanced_features(embeddings)

    return embeddings


def maggraphwave_core(adj: sp.spmatrix, param: MaggraphwaveHyperparameters, node_indices: Sequence[int] = None,
                      thresholds: torch.Tensor = None, device_ids=None):
    node_indices = digraphwave.utils.check_node_indices(adj.shape[0], node_indices)
    batches = digraphwave.utils.make_batches(node_indices, batch_size=param.batch_size)
    lap = utils.hermitian_laplacian(adj, q=param.q)
    if thresholds is None:
        thresholds = param.get_thresholds(adj)

    devices = utils.device_ids2devices(device_ids)
    if len(devices) == 1:
        embeddings = []
        for batch in tqdm.tqdm(batches):
            embeddings.append(
                _maggraphwave_core_child(lap, batch=batch, thresholds=thresholds, param=param, device=devices[0]))
    else:
        raise NotImplementedError

    return torch.cat(embeddings, dim=0)


def _maggraphwave_core_child(lap: sp.spmatrix, batch: torch.LongTensor, thresholds: torch.Tensor,
                             param: MaggraphwaveHyperparameters, device: torch.device):
    use_sparse = True
    maxval = np.pi
    minval = 0.
    char_fun_t = param.char_fun_step * np.arange(1, param.k_phi + 1, step=1, dtype=np.float64)
    embeddings = []
    n_nodes = lap.shape[0]

    B = expm._create_batch(n_nodes, batch, dtype=torch.float64).numpy()
    expm_batch = sp.linalg.expm_multiply(-lap, B=B, start=param.tau_start, stop=param.tau_stop, num=param.k_tau)
    expm_batch = torch.from_numpy(expm_batch)

    for k in range(param.k_tau):
        psi = w2e.threshold_batch_inplace(expm_batch[k], thresholds[batch], as_sparse=use_sparse)
        psi = w2e.transform(psi, maxval=maxval, minval=minval, log_arctan=False)
        embeddings.append(w2e.wavelets2embeddings(psi, char_fun_t))

    embeddings = torch.cat(embeddings, dim=1)
    return embeddings
