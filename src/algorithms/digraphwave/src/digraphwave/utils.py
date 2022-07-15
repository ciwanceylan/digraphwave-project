from typing import Union, Mapping, Sequence, Tuple, Dict, Optional, List
import warnings
import hashlib
import dataclasses as dc
import numpy as np
import scipy.sparse as sp
import scipy.special as sspecial
import numba as nb
import torch


@dc.dataclass(frozen=True)
class Parameters:
    thresholds: str
    R: Optional[int]
    tau_start: float
    tau_stop: float
    k_tau: int
    k_phi: int
    k_emb: int
    num_nodes: int
    num_edges: int
    # beta: float
    # avg_branch_factor: float
    order: int
    aggregate_neighbors: bool
    batch_size: int
    dtype: torch.dtype
    char_fun_step: float

    def get_thresholds(self, adj):
        if self.thresholds == "digraphwave":
            theta = calculate_thetas(adj, self.R)
            theta = torch.from_numpy(theta).to(dtype=torch.float64, device=torch.device("cpu"))
        elif self.thresholds == "flat":
            theta = torch.tensor(self.num_nodes * [1e-4 / self.num_nodes], dtype=torch.float64,
                                 device=torch.device('cpu'))
        else:
            raise ValueError(f"Unknown thresholds option '{self.thresholds}'")
        return theta

    def get_taus(self):
        return np.linspace(self.tau_start, self.tau_stop, self.k_tau, dtype=np.float64)

    def to_hash(self):
        # tuple_rep = (self.thresholds, self.q, self.tau_start, self.tau_stop,
        #              self.k_tau, self.k_phi, self.k_emb,
        #              self.num_nodes, self.beta, self.avg_branch_factor,
        #              self.order,
        #              self.aggregate_neighbors,
        #              self.batch_size,
        #              self.dtype)
        md5_hash = hashlib.md5(str(self).encode('utf-8')).hexdigest()
        return md5_hash

    @staticmethod
    def get_auto_batch_size(*, memory_available: int, num_nodes: int, num_edges: int, k_tau: int, k_emb: int,
                            bytes_per_element=8, cuda_overhead: float = 1.2):
        capacity = (memory_available - cuda_overhead) * (1024 ** 3) / bytes_per_element
        needed_other = (3 * num_edges + num_nodes * k_emb)  # Factors 4 and 2 are heuristic to account for some overhead
        # needed_dense_batch: k_tau * n_nodes in expm, + 1 copy during computation, + 1 copy with complex number in w2e
        # 3.5 accounts for some overhead during computations
        needed_dense_batch = 3.5 * num_nodes * (k_tau + 1 + 2)
        batch_size = (capacity - needed_other) / needed_dense_batch
        num_needed_batches = np.ceil(num_nodes / batch_size)
        batch_size = int(np.ceil(num_nodes / num_needed_batches))
        return batch_size

    def new_modified(self, **kwargs):
        old_obj = dc.asdict(self)
        old_obj.update(kwargs)
        return type(self)(**old_obj)


numpy_to_torch_dtype_dict = {
    bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128
}
# Dict of torch dtype -> NumPy dtype
torch_to_numpy_dtype_dict = {value: key for (key, value) in numpy_to_torch_dtype_dict.items()}


def np_torch_dtypes(dtype):
    if dtype not in torch_to_numpy_dtype_dict and dtype not in numpy_to_torch_dtype_dict:
        raise TypeError(f"Unrecognized dtype {dtype}")
    np_dtype = torch_to_numpy_dtype_dict[dtype] if dtype in torch_to_numpy_dtype_dict else dtype
    torch_dtype = numpy_to_torch_dtype_dict[dtype] if dtype in numpy_to_torch_dtype_dict else dtype
    return np_dtype, torch_dtype


def calc_out_degrees(adj: Union[sp.spmatrix, torch.Tensor], weighted: bool):
    return calc_degrees(adj, weighted=weighted, in_degrees=False)


def calc_in_degrees(adj: Union[sp.spmatrix, torch.Tensor], weighted: bool):
    return calc_degrees(adj, weighted=weighted, in_degrees=True)


def calc_degrees(adj: Union[sp.spmatrix, torch.Tensor], weighted: bool, in_degrees: bool):
    if weighted and isinstance(adj, sp.spmatrix):
        return calc_weighted_degrees(adj, in_degrees=in_degrees)
    elif not weighted and isinstance(adj, sp.spmatrix):
        return calc_unweighted_degrees(adj, in_degrees=in_degrees)
    elif weighted and isinstance(adj, torch.Tensor):
        return calc_weighted_degrees_dense(adj, in_degrees=in_degrees)
    elif not weighted and isinstance(adj, torch.Tensor):
        return calc_unweighted_degrees_dense(adj, in_degrees=in_degrees)
    else:
        raise NotImplementedError(f"out degrees for weighted={weighted} not implemented for type {type(adj)}")


def calc_weighted_degrees(adj: sp.spmatrix, in_degrees: bool):
    if in_degrees:
        return np.squeeze(np.asarray(adj.sum(axis=1))).astype(np.float64)
    else:
        return np.squeeze(np.asarray(adj.sum(axis=0))).astype(np.float64)


def calc_unweighted_degrees(adj: sp.spmatrix, in_degrees: bool):
    num_nodes = adj.shape[1]
    nz_row, nz_col = adj.nonzero()

    if in_degrees:
        degrees_index, degrees_ = np.unique(nz_row, return_counts=True)
    else:
        degrees_index, degrees_ = np.unique(nz_col, return_counts=True)
    degrees = np.zeros(num_nodes, dtype=np.float64)
    degrees[degrees_index] = degrees_.astype(np.float64)
    return degrees


def calc_weighted_degrees_dense(adj: torch.Tensor, in_degrees: bool):
    if in_degrees:
        return torch.squeeze(adj.sum(dim=1)).to(torch.float64)
    else:
        return torch.squeeze(adj.sum(dim=0)).to(torch.float64)


def calc_unweighted_degrees_dense(adj: torch.Tensor, in_degrees: bool):
    num_nodes = adj.shape[1]
    nonzero = adj.nonzero()
    if in_degrees:
        degrees_index, degrees_ = torch.unique(nonzero[:, 0], return_counts=True)
    else:
        degrees_index, degrees_ = torch.unique(nonzero[:, 1], return_counts=True)
    degrees = torch.zeros(num_nodes, dtype=torch.float64)
    degrees[degrees_index] = degrees_.to(dtype=torch.float64)
    return degrees


def calc_avg_branch_factor(adj: sp.spmatrix):
    """ Calculate the average branch factor for the graph """
    out_degrees_ = calc_out_degrees(adj, weighted=False)
    beta = np.mean(out_degrees_)
    return beta


def calc_source_star_beta(adj: sp.spmatrix):
    """ Calculate source star beta parameter for the whole graph. Same as average branch factor except node with
     out degree zero are ignored. """
    out_degrees_ = calc_out_degrees(adj, weighted=False)
    beta = np.mean(out_degrees_[out_degrees_ > 0])
    return beta


def calc_ss_beta_per_node(adj: sp.spmatrix):
    """ Calculate source star beta parameters per node in the graph """
    # num_nodes = adj.shape[0]
    out_degrees_ = np.asarray(calc_out_degrees(adj, weighted=False))
    num_nodes_with_out_edge = len(out_degrees_[out_degrees_ > 0])

    beta = (out_degrees_[out_degrees_ > 0]).mean()
    beta_i = (
            (num_nodes_with_out_edge * beta - out_degrees_) /
            np.maximum(num_nodes_with_out_edge - (out_degrees_ > 0), 1)
    )
    return beta_i


def log_poly_exp_factorial(tau: Union[float, np.ndarray], ell: Union[float, np.ndarray]):
    """ Compute the logarithm of tau^ell * exp(-tau) / (ell)! """

    tau = np.atleast_1d(tau)
    ell = np.atleast_1d(ell)
    res = ell * np.log(tau[:, None]) - tau[:, None] - sspecial.loggamma(ell + 1)
    return res.squeeze()


def poly_exp_factorial(tau: Union[float, np.ndarray], ell: Union[float, np.ndarray]):
    """ Compute the tau^ell * exp(-tau) / (ell)! """
    return np.exp(log_poly_exp_factorial(tau, ell))


# def laplacian(adj):
#     """
#     Using the definition that element adj_ij means indicates the edge j -> i
#     Args:
#         adj:
#
#     Returns:
#
#     """
#     n_nodes, _ = adj.shape
#     format = adj.getformat()
#     out_degs = out_degrees(adj)
#     diag = sp.diags(out_degs, 0, adj.shape, format=format)
#     return diag - adj


def remove_self_loops(adj: sp.spmatrix):
    """ WARNING: THIS CAN BE SLOW FOR LARGE GRAPHS. BETTER AVOID IF SELF-LOOPS ARE NOT PRESENT.
    Remove self-loops by setting diagonal to zero. """
    with warnings.catch_warnings():
        # This produces a SparseEfficiencyWarning if adj is csr or csc
        # However, conversion to lil is also expensive
        warnings.simplefilter('ignore', category=sp.SparseEfficiencyWarning)
        if np.any(adj.diagonal() > 0):
            adj.setdiag(0)
        adj.eliminate_zeros()
    return adj


def normalized_laplacian(adj: sp.spmatrix, index_dtype=np.int64, data_dtype=np.float64):
    """
    Compute normalised Laplacian from adjacency matrix.
    Using the definition that element adj[i, j] means indicates the edge j -> i.
    Currently only to be used with undirected graphs. Behaviour for directed graphs with Digraphwave is unknown.

    Args:
        adj: Weighted adjacency matrix
        index_dtype: dtype for index
        data_dtype: dtype for data
    Returns:
        lap: Normalised Laplacian
    """
    # adj = remove_self_loops(adj)
    n_nodes, _ = adj.shape
    format = adj.getformat()

    out_degs = calc_out_degrees(adj, weighted=True)
    out_degs_inv = sp.diags(sqrtinversewithzero(out_degs), 0, adj.shape, format=format)

    in_degs = calc_in_degrees(adj, weighted=True)
    in_degs_inv = sp.diags(sqrtinversewithzero(in_degs), 0, adj.shape, format=format)

    eye = sp.diags((in_degs > 0) & (out_degs > 0), format=format, dtype=data_dtype)
    dad = in_degs_inv.dot(adj.dot(out_degs_inv))
    lap = eye - dad
    lap.indices = lap.indices.astype(index_dtype, copy=False)
    lap.indptr = lap.indptr.astype(index_dtype, copy=False)
    lap.data = lap.data.astype(data_dtype, copy=False)
    return lap


def rw_laplacian(adj: sp.spmatrix, use_out_degrees=True, index_dtype=np.int64, data_dtype=np.float64):
    """
    Compute random walk normalised Laplacian from adjacency matrix.
    Using the definition that element adj[i, j] means indicates the edge j -> i.

    Args:
        adj: Weighted adjacency matrix
        use_out_degrees: Normalise using out degrees. Otherwise in degrees.
        index_dtype: dtype for index
        data_dtype: dtype for data

    Returns:
        lap: Random walk normalised Laplacian

    """
    # adj = remove_self_loops(adj)
    n_nodes, _ = adj.shape
    format = adj.getformat()
    if use_out_degrees:
        out_degs = calc_out_degrees(adj, weighted=True)
        out_degs_inv = sp.diags(inversewithzero(out_degs), 0, adj.shape, format=format)
        ad = adj.dot(out_degs_inv)
        eye = sp.diags(out_degs > 0, format=format, dtype=data_dtype)
        lap = eye - ad

    else:
        in_degs = calc_in_degrees(adj, weighted=True)
        in_degs_inv = sp.diags(inversewithzero(in_degs), 0, adj.shape, format=format)
        da = in_degs_inv.dot(adj)
        eye = sp.diags(in_degs > 0, format=format, dtype=data_dtype)
        lap = eye - da

    lap.indices = lap.indices.astype(index_dtype, copy=False)
    lap.indptr = lap.indptr.astype(index_dtype, copy=False)
    lap.data = lap.data.astype(data_dtype, copy=False)
    return lap


def rw_laplacian_dense(adj: torch.Tensor):
    """
    Compute random walk normalised Laplacian from dense adjacency matrix.
    Using the definition that element adj[i, j] means indicates the edge j -> i.

    Args:
        adj: Weighted adjacency matrix

    Returns:
        lap: Random walk normalised Laplacian

    """
    # adj = adj.fill_diagonal_(0.)
    out_degs = calc_weighted_degrees_dense(adj, in_degrees=False)
    out_degs_inv = torch.zeros_like(out_degs)
    out_degs_inv[out_degs > 0] = torch.reciprocal(out_degs[out_degs > 0])
    ad = adj.to(dtype=torch.float64).matmul(torch.diag(out_degs_inv))
    eye = torch.diag(out_degs > 0).to(torch.float64)
    lap = eye - ad
    return lap


def _hermitian_lap_help(adj: sp.spmatrix, q: float):
    adj = adj.astype(np.complex128)
    adj_s = 0.5 * (adj + adj.T)
    not_equal = (adj != adj.T).tocoo()

    phase_mat = adj_s.copy()
    phase_mat.data = np.ones_like(phase_mat.data)
    for r, c in zip(not_equal.row, not_equal.col):
        phase_mat[r, c] = np.exp(1j * 2 * np.pi * q * (adj[r, c] - adj[c, r]))

    # phase_mat = (adj.T - adj)
    # phase_mat.data = np.exp(1j * 2 * np.pi * q * phase_mat.data)

    return adj_s, phase_mat


def hermitian_laplacian(adj: sp.spmatrix, q: float):
    # adj = remove_self_loops(adj)
    adj_s, phase_mat = _hermitian_lap_help(adj, q)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=np.ComplexWarning)
        degrees = calc_out_degrees(adj_s, weighted=True).astype(np.float64)
    D = sp.diags(degrees, format=adj.getformat())

    hem_lap = D - adj_s.multiply(phase_mat)
    return hem_lap


def normalised_hermitian_laplacian(adj: sp.spmatrix, q: float):
    # adj = remove_self_loops(adj)
    adj_s, phase_mat = _hermitian_lap_help(adj, q)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=np.ComplexWarning)
        degrees = calc_out_degrees(adj_s, weighted=True).astype(np.float64)
    D_half = sp.diags(sqrtinversewithzero(degrees), format=adj.getformat())
    eye = sp.diags(degrees > 0, format=adj.getformat(), dtype=np.complex128)

    hem_lap = eye - D_half.dot(adj_s).dot(D_half).multiply(phase_mat)
    return hem_lap


@nb.vectorize([nb.float64(nb.float64)])
def _sqrtinversewithzero(a):
    return 1. / np.sqrt(a) if a > 0. else 0.


def sqrtinversewithzero(a):
    # Warning supression to solve this issue: https://github.com/numba/numba/issues/4793
    with np.errstate(divide='ignore'):
        return _sqrtinversewithzero(a)


@nb.vectorize([nb.float64(nb.float64)])
def _inversewithzero(a):
    return 1. / a if a > 0. else 0.


def inversewithzero(a):
    # Warning supression to solve this issue: https://github.com/numba/numba/issues/4793
    with np.errstate(divide='ignore'):
        return _inversewithzero(a)


def one_matrix_norm(mat: Union[sp.spmatrix, np.ndarray]):
    """ Max sum over rows matrix norm """
    return np.abs(mat).sum(axis=0).max()


def inf_matrix_norm(mat: Union[sp.spmatrix, np.ndarray]):
    """ Max sum over columns matrix norm """
    return np.abs(mat).sum(axis=1).max()


def gamma_k_n(n, K, dtype=np.float64):
    """ Round-off error term from Higham """
    u = np.finfo(dtype).eps
    nku = n * K * u
    return nku / (1 - nku)


def one_taylor_bound(n, K, tau, dtype=np.float64):
    """ Taylor approximation error bound for matrix exponential """
    log_approx_bound = (K + 1) * np.log(tau) + tau - sspecial.loggamma(K + 2)
    approx_bound = np.exp(log_approx_bound)
    bound = (gamma_k_n(n, K, dtype=dtype) + approx_bound) * np.exp(tau)
    return bound


def one_taylor_bound_rel(n, K, tau, dtype=np.float64):
    """ Taylor approximation relative error bound for matrix exponential """
    return one_taylor_bound(n, K, tau, dtype) * np.exp(2 * tau)


def calculate_thetas(adj: sp.spmatrix, radius: float, min_value: float = 1e-6):
    out_degrees_ = np.asarray(calc_out_degrees(adj, weighted=False))
    beta_i = calc_ss_beta_per_node(adj)
    mask = ~((beta_i < 1e-8) | (out_degrees_ < 1e-8))
    if radius < 1e-8:
        theta = np.full(shape=out_degrees_.shape, fill_value=min_value, dtype=np.float64)
    else:
        log_theta = np.full(shape=out_degrees_.shape, fill_value=-radius, dtype=np.float64)

        log_other = np.minimum(-radius + np.log(radius) - np.log(out_degrees_[mask]),
                               -1 - np.log(out_degrees_[mask]) - (radius - 1) * np.log(
                                   beta_i[mask]) - sspecial.loggamma(radius + 1)
                               )
        log_theta[mask] = np.minimum(log_theta[mask], log_other)

        theta = np.maximum(np.exp(log_theta).astype(np.float64), min_value)  # TODO maybe change this threshold
    return theta


def sparse2dict(spmat: sp.spmatrix, return_weights: bool = False,
                use_rows: bool = False, as_numba_dict: bool = False,
                sources: Sequence[int] = None) -> Tuple[Dict[int, np.ndarray], Optional[Dict[int, np.ndarray]]]:
    """ Convert a sparse matrix to dictionary representation. Useful for numba functions not supporting scipy sparse
    matrices.
    By using `sources` the dictionary keys can be specified for each column (or row if use_rows=True).
    
    Args:
        spmat: A Scipy sparse matrix where sp[i,j] is interpreted as an edge j -> i
        return_weights: Return a dictionary of the weights associated with each edge.
        use_rows: Create the adjacency dict using incoming edges instead of outgoing.
        as_numba_dict: Create a numba Dict which can be used with nopython numba functions.
        sources: A sequence of integers specifying aliases for the sources in `spmat`

    Returns:
        targets: Dict of adjacent nodes for each source in the matrix
        weights: Dict of element values for each source-target. None if return_weights=False.
    """
    if not (isinstance(spmat, sp.csr_matrix) or isinstance(spmat, sp.csc_matrix)):
        raise TypeError(f"'spmat' needs to be csr or csc matrix, not {type(spmat)}.")

    if use_rows:
        spmat = spmat.tocsr()
        num_sources = spmat.shape[0]
    else:
        spmat = spmat.tocsc()
        num_sources = spmat.shape[1]

    if sources is None:
        sources = range(num_sources)

    if len(sources) != num_sources:
        raise ValueError(f"'sources' with {len(sources)} should be the same length as number of sources in {spmat}, "
                         f"which should be {num_sources}")
    # assert len(sources) == num_sources  # Source aliases should be specified for every source if used

    spmat.indices = spmat.indices.astype(np.int64)
    spmat.indptr = spmat.indptr.astype(np.int64)

    if as_numba_dict:
        targets = nb.typed.Dict.empty(nb.int64, nb.int64[::1])
        for i, s in enumerate(sources):
            targets[s] = spmat.indices[spmat.indptr[i]:spmat.indptr[i + 1]]
    else:
        targets = {s: spmat.indices[spmat.indptr[i]:spmat.indptr[i + 1]] for i, s in enumerate(sources)}

    if return_weights:
        spmat.data = spmat.data.astype(np.float64)
        if as_numba_dict:

            weights = nb.typed.Dict.empty(nb.int64, nb.float64[::1])
            for i, s in enumerate(sources):
                weights[s] = spmat.data[spmat.indptr[i]:spmat.indptr[i + 1]]
        else:
            weights = {s: spmat.data[spmat.indptr[i]:spmat.indptr[i + 1]] for i, s in enumerate(sources)}
    else:
        weights = None
    return targets, weights


def mapping2sparse(targets: Mapping[int, Sequence[int]], weights: Mapping[int, Sequence[float]],
                   num_targets: int, sources: Sequence[int] = None, sources_are_rows: bool = False) -> sp.spmatrix:
    """ Converts a mapping representation of a sparse matrix to a scipy sparse matrix. The inverse of `sparse2dict`.

    Args:
        targets: Dict of adjacent nodes for each source in the matrix
        weights: Dict of element values for each source-target.
        num_targets: The number of rows (or columns if `sources_are_rows=True`)
        sources: Specifies in which order to put the sources in the matrix.
        sources_are_rows: Sources are on the rows of the sparse matrix.

    Returns:
        spmat: A csc (or csr if sources_are_rows=True) matrix representation of the mappings
    """
    n_nodes = len(targets)
    if sources is None:
        sources = range(n_nodes)

    data = []
    indices = []
    indptr = [0]

    for i, s in enumerate(sources):
        targets_ = targets[s]
        indptr.append(indptr[i] + len(targets_))
        indices.append(targets_)
        data.append(weights[s])

    indices = np.concatenate(indices)
    data = np.concatenate(data)
    if sources_are_rows:
        return sp.csr_matrix((data, indices, indptr), shape=(len(sources), num_targets))
    else:
        return sp.csc_matrix((data, indices, indptr), shape=(num_targets, len(sources)))


def auto_find_batch_size(num_nodes, num_edges, k_emb, k_tau, available_memory, safety_factor=2):
    n = num_nodes
    num_adj_elements = 2 * 3 * num_edges
    num_embeddings_elements = k_emb * n
    total_elements = 1024 ** 3 * (available_memory / 8)
    total_elements /= safety_factor
    estimated_bs = (total_elements - num_adj_elements - num_embeddings_elements) / (2 * k_tau * n)
    estimated_bs = max(2 ** (int(np.log2(estimated_bs))), 1)
    return estimated_bs


def device_ids2devices(device_ids: Optional[Union[List[int], int]]):
    if device_ids is None:
        devices = [torch.device('cpu')]
    else:
        try:
            _ = len(device_ids)
        except TypeError:
            device_ids = [device_ids]
        devices = [torch.device(id_) for id_ in device_ids]
    return devices


def check_node_indices(num_nodes, node_indices: Sequence[int] = None):
    if node_indices is None:
        node_indices = torch.arange(num_nodes, dtype=torch.int64)
    elif isinstance(node_indices, torch.Tensor):
        node_indices = node_indices.to(dtype=torch.int64)
    else:
        node_indices = torch.tensor(node_indices, dtype=torch.int64)
    if node_indices.max() >= num_nodes:
        raise ValueError("Node index cannot be larger than number of nodes.")
    return node_indices


def make_batches(node_indices: torch.Tensor, batch_size: int):
    assert batch_size is not None
    assert batch_size > 0

    batches = [
        node_indices[b:b + batch_size].to(dtype=torch.int64).long()
        for b in range(0, len(node_indices), batch_size)
    ]
    return batches
