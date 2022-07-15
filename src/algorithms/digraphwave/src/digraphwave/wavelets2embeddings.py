from typing import Union, Optional
import numpy as np
import torch
import torch_sparse as tsp


def _linear_transform(x: torch.Tensor, maxval: float, minval: float):
    """ Scales x to the range [0, PI] using minval and maxval """
    return np.pi * (x - minval) / (maxval - minval)


def _log_transform(x: torch.Tensor, minval: float, maxval: float = 1.):
    """ Take the log of x and scale to the range [0, PI] using `_linear_transform` """
    return _linear_transform(torch.log(x), maxval=np.log(maxval), minval=np.log(minval))


def _log_arctan_transform(x: torch.Tensor, maxval: float, minval: float = 0):
    """ Take the log of x and scale to the range [0, PI] using arctan.
    If x has a maximal value this is scaled to be inf using tan. """
    x = x.to(dtype=torch.float64)
    if maxval is not None:
        x = torch.tan(0.5 * _linear_transform(x, maxval=maxval, minval=minval))
    return torch.arctan(torch.log(x)) + (np.pi / 2)


def transform(x: Union[tsp.SparseTensor, torch.Tensor], maxval: Optional[float], minval: float = 0,
              log_arctan: bool = False):
    if isinstance(x, tsp.SparseTensor):
        return transform_sparse(x, maxval, minval, log_arctan=log_arctan, copy=False)
    else:
        return transform_dense(x, maxval, minval, log_arctan=log_arctan)


def transform_sparse(x: tsp.SparseTensor, maxval: Optional[float], minval: float = 0, log_arctan: bool = False,
                     copy: bool = False):
    if maxval is None and not log_arctan:
        raise ValueError("Maxval is required if not using 'log_arctan'")
    if log_arctan:
        value = _log_arctan_transform(x.storage.value(), maxval=maxval, minval=minval)
    else:
        value = _linear_transform(x.storage.value(), maxval=maxval, minval=minval)

    if copy:
        x = x.set_value(value, layout='coo')
    else:
        x = x.set_value_(value, layout='coo')

    return x


def transform_dense(x: torch.Tensor, maxval: Optional[float], minval: float = 0, log_arctan: bool = False):
    if maxval is None and not log_arctan:
        raise ValueError("Maxval is required if not using 'log_arctan'")
    if log_arctan:
        x = _log_arctan_transform(x, maxval=maxval, minval=minval)
    else:
        x = _linear_transform(x, maxval=maxval, minval=minval)
    return x


def threshold_batch_inplace(expm_batch: torch.Tensor, theta: torch.Tensor, as_sparse: bool):
    theta = torch.atleast_2d(theta)
    expm_batch[torch.lt(torch.abs(expm_batch), theta)] = 0
    if as_sparse:
        expm_batch = tsp.SparseTensor.from_dense(expm_batch)
    return expm_batch


def _wavelets2embeddings(psi: tsp.SparseTensor, time_points: torch.Tensor):
    num_total_nodes = psi.size(0)
    batch_size = psi.size(1)

    prop_zeros = torch.zeros(batch_size, device=psi.device(), dtype=torch.float64)
    nonzero_cols, num_nonzero = torch.unique(psi.storage.col(), return_counts=True)
    prop_zeros[nonzero_cols] = (num_total_nodes - num_nonzero.to(dtype=torch.float64)) / num_total_nodes

    tmp: tsp.SparseTensor = psi.copy().to(torch.complex128)
    values = tmp.storage.value()
    res = []
    for i, t in enumerate(time_points):
        # tmp.set_value_(torch.cos(values * t), layout="coo")
        # real_part = tmp.sum(dim=0) / num_total_nodes

        tmp.set_value_(torch.exp(1j * values * t), layout="coo")
        res_ = tmp.sum(dim=0) / num_total_nodes
        real_part = res_.real

        real_part = real_part + prop_zeros
        res.append(real_part)

        # tmp.set_value_(torch.sin(values * t), layout="coo")
        # im_part = tmp.sum(dim=0) / num_total_nodes

        im_part = res_.imag
        res.append(im_part)

    return torch.stack(res, dim=0).T


def _wavelets2embeddings_dense(psi: torch.Tensor, time_points: torch.Tensor):
    num_total_nodes = psi.size(0)
    res = []
    for i, t in enumerate(time_points):
        res_ = torch.exp(1j * psi * t).sum(dim=0) / num_total_nodes
        # real_part = torch.cos(psi * t).sum(dim=0) / num_total_nodes
        # res.append(real_part)
        res.append(res_.real)

        # im_part = torch.sin(psi * t).sum(dim=0) / num_total_nodes
        # res.append(im_part)
        res.append(res_.imag)

    return torch.stack(res, dim=0).T


def wavelets2embeddings(psi: Union[torch.Tensor, tsp.SparseTensor], time_points: torch.Tensor):
    if isinstance(psi, tsp.SparseTensor):
        return _wavelets2embeddings(psi, time_points)
    else:
        return _wavelets2embeddings_dense(psi, time_points)
