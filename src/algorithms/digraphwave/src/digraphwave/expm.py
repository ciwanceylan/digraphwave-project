from typing import Sequence, Union
import dataclasses as dc
import numpy as np
import scipy.sparse as sp
import torch
import torch_sparse as tsp

import digraphwave.utils as utils


@dc.dataclass(frozen=True)
class ExpmObject:
    X: tsp.SparseTensor
    coeffs: torch.Tensor
    thresholds: torch.Tensor

    def to(self, device: torch.device):
        return ExpmObject(self.X.to(device), self.coeffs.to(device), self.thresholds.to(device))


# @torch.jit.script
def _create_batch(n_nodes: int, node_indices: torch.LongTensor, dtype: torch.dtype):
    if len(node_indices.shape) == 1:
        node_indices = node_indices.unsqueeze(0)
    v = torch.zeros((n_nodes, node_indices.shape[1]), device=node_indices.device, dtype=dtype)
    v.scatter_(dim=0, index=node_indices, value=1.)
    return v


def _taylor_expm_coeffs(tau: float, order: int, device: torch.device = None,
                        dtype: torch.dtype = None):
    """ Compute Taylor coefficients for f(x)=exp(- tau x) using FFT.
        See :meth:`???` for further details.

    The pytorch implementation of FFT returns both the positive and negative frequencies. Only the positive frequencies
    are needed for the Taylor polynomials, so the negative are discarded.

    Args:
        tau (float): The scale parameter `s` in exp(- tau x)
        order (int): The index of the last coefficient to return
        device (torch.device): Device to use. cpu or cuda
        dtype (torch.dtype): Which float dtype to use

    Returns:
        coeffs (torch.Tensor): The first `order + 1` Taylor coefficients
    """
    if device is None:
        device = torch.device("cpu")

    if dtype is None:
        dtype = torch.float64

    M = 2 * order + 1  # Compute twice the number of order and only take the positive frequencies
    pi2_k_M = 2 * np.pi * torch.arange(start=0, end=M, device=device, dtype=dtype) / M
    xx = torch.exp(1j * pi2_k_M) + 1
    fft_coeffs = torch.exp(- tau * xx)
    coeffs = torch.real(torch.fft.fft(fft_coeffs, M)) / M
    return coeffs[:order]


# @torch.jit.script
def _taylor_approximation(X: tsp.SparseTensor, batch_indices: torch.LongTensor,
                          coeffs: torch.Tensor):
    """ Compute Taylor approximation of a maxtrix function f(X) defined by the provided coefficients.
    Using :meth:`taylor_expm_coeffs` will result in the matrix exponential exp(- tau X) .

    See paper "ref" for more details.

    Args:
        X (tsp.SparseTensor): The input matrix
        batch_indices (torch.LongTensor): Which columns in f(X) to compute.
        coeffs (torch.Tensor): The Taylor coefficients defining f(X) and the order of the approximation polynomial

    Returns:
        out (torch.Tensor): The `batch_indices` columns of f(X)
    """
    if isinstance(X, tsp.SparseTensor):
        mm_pkg = tsp
        device = X.device()
        dtype = X.dtype()
    else:
        mm_pkg = torch
        device = X.device
        dtype = X.dtype
    order = coeffs.shape[1] - 1
    num_tau = coeffs.shape[0]
    n_nodes = X.size(0)
    monome = _create_batch(n_nodes, batch_indices, dtype=dtype)
    # out = [torch.zeros_like(monome) for _ in range(num_tau)]
    out = torch.zeros((num_tau, monome.shape[0], monome.shape[1]), dtype=dtype, device=device)

    for k in range(order + 1):
        # for k_tau in range(num_tau):
        #     out[k_tau] += coeffs[k_tau, k] * monome
        out += coeffs[:, k][:, None, None] * monome[None, :, :]
        monome = mm_pkg.matmul(X, monome)
    return out


def _chebyshev_expm_coeffs(tau: float, order: int, device: torch.device = None,
                           dtype: torch.dtype = None):
    """ Compute Chebyshev coefficients for f(x)=exp(- tau x) using FFT.
        See :meth:`???` for further details.

    The pytorch implementation of FFT returns both the positive and negative frequencies. Only the positive frequencies
    are needed for the Chebyshev polynomials, so the negative are discarded.

    Args:
        tau (float): The scale parameter `s` in exp(- tau x)
        order (int): The index of the last coefficient to return
        device (torch.device): Device to use. cpu or cuda
        dtype (torch.dtype): Which float dtype to use

    Returns:
        coeffs (torch.Tensor): The first `order + 1` Chebyshev coefficients
    """
    if device is None:
        device = torch.device("cpu")

    if dtype is None:
        dtype = torch.float64

    M = 2 * order + 1  # Compute twice the number of order and only take the positive frequencies
    pi2_k_M = 2 * np.pi * torch.arange(start=0, end=M, device=device, dtype=dtype) / M
    xx = torch.cos(pi2_k_M) + 1
    fft_coeffs = torch.exp(- tau * xx)
    coeffs = 2 * torch.real(torch.fft.fft(fft_coeffs, M)) / M
    coeffs[0] = coeffs[0] / 2
    return coeffs[:order]


def _chebyshev_approximation(X: tsp.SparseTensor, batch_indices: torch.LongTensor, coeffs: torch.Tensor):
    """ Compute Chebyshev approximation of a maxtrix function f(X) defined by the provided coefficients.
    Using :meth:`_chebyshev_expm_coeffs` will result in the matrix exponential exp(- tau X) .

    Warning, X should only have real eigenvalues for Chebyshev to be accurate.
    See paper "ref" for more details.

    Args:
        X (tsp.SparseTensor): The input matrix. Warning, should only have real eigenvalues.
        batch_indices (torch.LongTensor): Which columns in f(X) to compute.
        coeffs (torch.Tensor): The Chebyshev coefficients defining f(X) and the order of the approximation polynomial

    Returns:
        out (torch.Tensor): The `batch_indices` columns of f(X)
    """
    if isinstance(X, tsp.SparseTensor):
        mm_pkg = tsp
        device = X.device()
        dtype = X.dtype()
    else:
        mm_pkg = torch
        device = X.device
        dtype = X.dtype
    order = coeffs.shape[1] - 1
    num_tau = coeffs.shape[0]
    n_nodes = X.size(0)
    B = _create_batch(n_nodes, batch_indices, dtype=dtype)
    monome_k_2 = B
    monome_k_1 = mm_pkg.matmul(X, B)

    # out = [torch.zeros_like(B) for _ in range(num_tau)]
    out = torch.zeros((num_tau, B.shape[0], B.shape[1]), dtype=dtype, device=device)

    for k_tau in range(num_tau):
        out[k_tau] += coeffs[k_tau, 1] * monome_k_1 + coeffs[k_tau, 0] * monome_k_2

    for k in range(2, order + 1):
        next_monome = 2 * mm_pkg.matmul(X, monome_k_1) - monome_k_2
        # for k_tau in range(num_tau):
        #     out[k_tau] += coeffs[k_tau, k] * next_monome
        out += coeffs[:, k][:, None, None] * next_monome[None, :, :]
        monome_k_2 = monome_k_1
        monome_k_1 = next_monome
    return out


def make_polynomial_term(adj: Union[sp.spmatrix, torch.Tensor], device: torch.device = None, dtype: torch.dtype = None):
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float64

    if isinstance(adj, torch.Tensor):
        X = _make_polynomial_term_dense(adj, device, dtype)
    elif isinstance(adj, sp.spmatrix):
        X = _make_polynomial_term_sp_sparse(adj, device, dtype)
    else:
        raise TypeError(f"adj must be either {sp.spmatrix} or {torch.Tensor}, not {type(adj)}")
    return X


# def make_polynomial_term(adj: sp.spmatrix, device: torch.device = None, dtype: torch.dtype = None):
#     num_nodes = adj.shape[0]
#     if device is None:
#         device = torch.device("cpu")
#     if dtype is None:
#         dtype = torch.float64
#     np_dtype = utils.torch_to_numpy_dtype_dict[dtype]
#     X = tsp.SparseTensor.from_scipy(
#         utils.rw_laplacian(adj, data_dtype=np_dtype) - sp.eye(num_nodes, dtype=np_dtype, format=adj.getformat())
#     ).to(device=device)
#     return X
#
#
# def make_polynomial_term_dense(adj: torch.Tensor, device: torch.device = None, dtype: torch.dtype = None):
#     num_nodes = adj.shape[0]
#     if device is None:
#         device = torch.device("cpu")
#     if dtype is None:
#         dtype = torch.float64
#     X = utils.rw_laplacian_dense(adj).to(device=device, dtype=dtype) - torch.eye(num_nodes, dtype=dtype, device=device)
#     return X

def _make_polynomial_term_sp_sparse(adj: sp.spmatrix, device: torch.device, dtype: torch.dtype):
    np_dtype = utils.torch_to_numpy_dtype_dict[dtype]
    X = tsp.SparseTensor.from_scipy(
        utils.rw_laplacian(adj, data_dtype=np_dtype) - sp.eye(adj.shape[0], dtype=np_dtype, format=adj.getformat())
    ).to(device=device)
    return X


def _make_polynomial_term_dense(adj: torch.Tensor, device: torch.device, dtype: torch.dtype):
    return (utils.rw_laplacian_dense(adj).to(device=device, dtype=dtype) -
            torch.eye(adj.shape[0], dtype=dtype, device=device))


def _concatenate_expm_batches(expm_batches):
    num_taus = len(expm_batches[0])

    heats = {k_tau: [] for k_tau in range(num_taus)}
    for heats_ in expm_batches:
        for k_tau, heat_ in enumerate(heats_):
            heats[k_tau].append(heat_)

    for k_tau in range(num_taus):
        heats[k_tau] = torch.cat(heats[k_tau], dim=1)
    return heats


def lapexpm_taylor(adj: Union[sp.spmatrix, torch.Tensor], taus: Sequence[float], order: int, batch_size: int = None,
                   node_indices: Sequence[int] = None,
                   device: torch.device = None, dtype: torch.dtype = None):
    """ Computes the matrix expoential f(x)=expm(- tau L) using the rw normalised Laplacian.

    TODO add documentation
    Reference paper

    Args:
        adj: The adjacency matrix of the graph
        taus:
        order:
        batch_size:
        node_indices:
        device:
        dtype:

    Returns:

    """
    num_nodes = adj.shape[0]
    if batch_size is None:
        batch_size = num_nodes
    node_indices = utils.check_node_indices(adj.shape[0], node_indices)
    batches = utils.make_batches(node_indices, batch_size=batch_size)

    X = make_polynomial_term(adj, device=device, dtype=dtype)

    coeffs = torch.stack([_taylor_expm_coeffs(tau, order, device, dtype) for tau in taus], dim=0)
    expm_batches = []
    for batch in batches:
        expm_batches.append(_taylor_approximation(X, batch_indices=batch, coeffs=coeffs))

    return _concatenate_expm_batches(expm_batches)


def lapexpm_chebyshev(adj: Union[sp.spmatrix, torch.Tensor], taus: Sequence[float], order: int, batch_size: int = None,
                      node_indices: Sequence[int] = None,
                      device: torch.device = None, dtype: torch.dtype = None):
    """ Computes the matrix expoential f(x)=expm(- tau L) using the rw normalised Laplacian.

    TODO add documentation
    Reference paper

    Args:
        adj: The adjacency matrix of the graph
        taus:
        order:
        batch_size:
        node_indices:
        device:
        dtype:

    Returns:

    """
    num_nodes = adj.shape[0]
    if batch_size is None:
        batch_size = num_nodes
    node_indices = utils.check_node_indices(adj.shape[0], node_indices)
    batches = utils.make_batches(node_indices, batch_size=batch_size)

    X = make_polynomial_term(adj, device=device, dtype=dtype)

    # if (adj.T != adj).nnz > 0:
    #     warnings.warn("Using Chebyshev approximation for directed graph will likely result in large errors.")

    coeffs = torch.stack([_chebyshev_expm_coeffs(tau, order, device, dtype) for tau in taus], dim=0)

    expm_batches = []
    for batch in batches:
        expm_batches.append(_chebyshev_approximation(X, batch_indices=batch, coeffs=coeffs))

    return _concatenate_expm_batches(expm_batches)
