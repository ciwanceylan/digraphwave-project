import numpy as np
import numba as nb
import graph_tool as gt
import graph_tool.spectral as gts
import pandas as pd

import umap
import hdbscan

import digraphwave.utils as utils


@nb.jit(nb.types.UniTuple(nb.int64[:], 2)(
    nb.types.DictType(nb.int64, nb.int64[:]), nb.int64, nb.optional(nb.int64), nb.optional(nb.int64)),
    nopython=True, nogil=True)
def generalized_egonet(adj_dict: nb.typed.Dict, node: int, max_depth: int, max_degree: int = None):
    """
    Extract a generalized (restricted) egonet. Performs a BFS from the starting node and collects all nodes up to the
    provided depth. The BFS stops at nodes which degree >= max_degree. This prevents the subgraphs from becoming very
    large due to the presence of highly connected nodes in the graph.
    Args:
        adj_dict: The graph represented by a adjacency list using a numba.Dict and numpy.arrays
        node: The starting node.
        max_depth: The largest number of steps from the starting node. If None, the whole graph may be searched.
        max_degree: Degree at which the BFS stops. If None, no restriction is applied.

    Returns:
        subgraph_nodes: List of nodes in the generalized egonet subgraph
        dists: List of shortest path distances in the subgraph
    """
    num_nodes = len(adj_dict)
    subgraph_nodes = []
    queued_nodes = [node]
    if max_degree is None:
        max_degree = num_nodes + 1
    if max_depth is None:
        max_depth = num_nodes + 1
    node_is_checked = np.zeros(num_nodes, dtype=np.bool_)
    node_is_checked[node] = True
    dists = []
    # dists = np.full(num_nodes, fill_value=num_nodes + 1, dtype=np.int64)
    current_dist = 0
    while len(queued_nodes) > 0 and current_dist < max_depth:
        new_queue = []
        for node_ in queued_nodes:
            subgraph_nodes.append(node_)
            dists.append(current_dist)
            neigh = adj_dict[node_]
            if len(neigh) < max_degree:
                for neigh_ in neigh:
                    if not node_is_checked[neigh_]:
                        new_queue.append(neigh_)
                        node_is_checked[neigh_] = True
        queued_nodes = new_queue
        current_dist += 1
    return np.asarray(subgraph_nodes), np.asarray(dists)


def gt2adj_dict(graph: gt.Graph, neigh_type: str = 'out', weights: gt.EdgePropertyMap = None,
                as_numba_dict: bool = False):
    if neigh_type == 'out':
        adj = gts.adjacency(graph, weight=weights)
    elif neigh_type == 'in':
        adj = gts.adjacency(graph, weight=weights).T
    elif neigh_type == 'all':
        adj = gts.adjacency(gt.GraphView(graph, directed=False), weight=weights)
    else:
        raise ValueError(f"Unknown neigh_type {neigh_type}. Must be 'out', 'in' or 'all'.")
    return_weights = weights is not None
    return utils.sparse2dict(adj, return_weights=return_weights, use_rows=False, as_numba_dict=as_numba_dict)


def extract_egonet_gt(graph: gt.Graph, node: int, max_depth: int, neigh_type: str = 'all'):
    """Extract a r-egonet of around a node in a graph-tool graph.

    Args:
        graph (gt.Graph): Graph adjacency matrix as scipy sparse matrix
        node (int): Node index
        max_depth (int): Max number of hops of the extracted egonet

    Returns:
        egonets (List[gt.Graph]): List of egonets as graph-tool GraphViews of the main graph
    """
    adj_dict, adj_weights = gt2adj_dict(graph, neigh_type=neigh_type, as_numba_dict=True)
    egonet_nodes, dists = generalized_egonet(adj_dict, node, max_depth + 1, None)
    egonets = []

    for d in range(2, max_depth + 2):
        vfilt = graph.new_vp('bool', False)
        vfilt.a[egonet_nodes[dists < d]] = True
        egonets.append(gt.GraphView(graph, vfilt=vfilt))
    return egonets


def standardize(embeddings):
    return (embeddings - embeddings.mean(axis=0)) / embeddings.std(axis=0)


def make_2d_umap(embeddings):
    mapper = umap.UMAP(tqdm_kwds={'disable': False}).fit(standardize(embeddings))
    return mapper


def make_hdbscan(embeddings) -> hdbscan.HDBSCAN:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15).fit(standardize(embeddings))
    return clusterer


def make_vis_data(embeddings, graph: gt.Graph):
    print('Fit Umap')
    mapper = make_2d_umap(embeddings)
    print('Fit Hdbscan')
    clusterer = make_hdbscan(embeddings)

    vis_data = pd.DataFrame(mapper.embedding_, columns=['umap_x', 'umap_y'])
    vis_data["degree"] = graph.degree_property_map("total").a
    vis_data["outlier_score"] = clusterer.outlier_scores_
    # TODO Add cluster id
    return vis_data
