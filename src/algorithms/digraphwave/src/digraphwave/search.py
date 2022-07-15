import scipy.sparse as sp
import numpy as np
import numba as nb
# import ray
import tqdm.auto as tqdm
import joblib
from digraphwave.utils import sparse2dict


# @ray.remote
# def ray_sp_to_targets(adj: sp.spmatrix, node: int, targets: np.ndarray):
#     adj_dict, _ = adj2adj_dict(adj, as_numba_dict=True)
#     return shortest_path_with_targets(adj_dict=adj_dict, node=node, targets=targets, max_degree=None)
#
#
# def ray_parallel_sp_to_targets(adj: sp.spmatrix, sources, targets_per_source):
#     futures = [ray_sp_to_targets.remote(adj, sources[i], targets_per_source[i]) for i in range(len(sources))]
#     return ray.get(futures)


def joblib_parallel_sp_to_targets(adj: sp.spmatrix, sources, targets_per_source, n_jobs=-2, max_degree=None):
    adj_dict, _ = sparse2dict(adj, as_numba_dict=True)
    r = joblib.Parallel(n_jobs=n_jobs, prefer='threads')(
        joblib.delayed(shortest_path_with_targets)(adj_dict, s, targets_per_source[s], max_degree=max_degree)
        for s in sources
    )
    return r


@nb.jit(nb.float64[:](
    nb.types.DictType(nb.int64, nb.int64[::1]),
    nb.int64,
    nb.types.Array(nb.types.int64, 1, 'C', readonly=True),
    nb.optional(nb.int64)
),
    nopython=True, nogil=True)
def shortest_path_with_targets(adj_dict: nb.typed.Dict, node: int, targets: np.ndarray, max_degree: int = None):
    """
    Extract a the shortest path distances to the target nodes. Performs a BFS from the starting node until all nodes in
    `adj_targets` have been found or the graph has been exhausted.
    The BFS stops at nodes which degree >= max_degree. This prevents the distances to be measured through very high
     degree nodes.
    Args:
        adj_dict: The graph represented by a adjacency list using a numba.Dict and numpy.arrays
        node: The starting node.
        targets: Target nodes to find
        max_degree: Degree at which the BFS stops. If None, no restriction is applied.

    Returns:
        dists: List of shortest path distances to the targets
    """
    num_nodes = len(adj_dict)
    targets_distances = nb.typed.Dict.empty(key_type=nb.int64, value_type=nb.float64)
    for target in targets:
        targets_distances[target] = np.inf
    num_reached_targets = 0
    queued_nodes = [node]
    if max_degree is None:
        max_degree = num_nodes + 1
    node_is_checked = np.zeros(num_nodes, dtype=np.bool_)
    node_is_checked[node] = True
    current_dist = 0
    while len(queued_nodes) > 0 and num_reached_targets < len(targets):
        new_queue = []
        for node_ in queued_nodes:
            if node_ in targets_distances:
                targets_distances[node_] = current_dist
                num_reached_targets += 1
            neigh = adj_dict[node_]
            if len(neigh) < max_degree:
                for neigh_ in neigh:
                    if not node_is_checked[neigh_]:
                        new_queue.append(neigh_)
                        node_is_checked[neigh_] = True
        queued_nodes = new_queue
        current_dist += 1
    return np.asarray([targets_distances[target] for target in targets])
