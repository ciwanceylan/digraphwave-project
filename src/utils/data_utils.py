from collections import namedtuple
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx


# DataFile = namedtuple("DataFile", ["name", "format", "pd_kwargs"])

def try_read_num_nodes(filename):
    num_nodes = None
    comment_char = None
    with open(filename) as f:
        first_line = f.readline()
    if first_line[0] in "#%":
        comment_char = first_line[0]
        try:
            num_nodes = int(first_line.strip(comment_char + "\n "))
        except ValueError:
            pass
    return num_nodes, comment_char


def read_to_nx(filename, filetype, is_weighted, directed, num_nodes=None, remove_self_loops=True, **pd_kwargs):
    num_nodes_ = None
    comment_char = None
    if filetype in {"csv", "tsv", "edgelist"}:
        num_nodes_, comment_char = try_read_num_nodes(filename)
    num_nodes = num_nodes_ if num_nodes_ is not None else num_nodes

    if comment_char is not None:
        pd_kwargs["comment"] = comment_char
    edges, weights = read_edges(filename, filetype, is_weighted, remove_self_loops=remove_self_loops, **pd_kwargs)
    if num_nodes is None:
        num_nodes = np.max(edges) + 1
    return edges2nx(edges, weights, num_nodes=num_nodes, directed=directed)


def read_to_spmat(filename, filetype, is_weighted, directed, num_nodes=None, remove_self_loops=True, **pd_kwargs):
    num_nodes_ = None
    if filetype in {"csv", "tsv", "edgelist"}:
        num_nodes_, comment_char = try_read_num_nodes(filename)
    num_nodes = num_nodes_ if num_nodes_ is not None else num_nodes
    edges, weights = read_edges(filename, filetype, is_weighted, remove_self_loops=remove_self_loops, **pd_kwargs)
    if num_nodes is None:
        num_nodes = np.max(edges) + 1
    return edges2spmat(edges, weights, num_nodes=num_nodes, directed=directed)


def read_edges(filename, filetype, is_weighted, remove_self_loops=True, **pd_kwargs):
    if "index_col" not in pd_kwargs:
        pd_kwargs["index_col"] = False

    if "header" not in pd_kwargs:
        pd_kwargs["header"] = None

    if "comment" not in pd_kwargs:
        pd_kwargs["comment"] = '%'

    if filetype in {'csv', 'tsv', 'edgelist', 'txt'}:
        if filetype == "csv":
            pd_kwargs["sep"] = ","
        elif filetype == "tsv" or filetype == "edgelist" or filetype == "twitter_txt":
            pd_kwargs["sep"] = "\s+"

        edges, weights = _read_tex_to_edges(filename, is_weighted, remove_self_loops=remove_self_loops, **pd_kwargs)
    elif filetype == 'parquet':
        edges, weights = _read_parquet_to_edges(filename, is_weighted, remove_self_loops=remove_self_loops, **pd_kwargs)
    else:
        raise NotImplementedError

    if weights is None:
        weights = np.ones(edges.shape[0], dtype=np.float64)
    return edges, weights


def _read_parquet_to_edges(filename, is_weighted, remove_self_loops=True, **pd_kwargs):
    df = pd.read_parquet(filename)
    return _post_load(df, is_weighted, remove_self_loops)


def _read_tex_to_edges(filename, is_weighted, remove_self_loops=True, **pd_kwargs):
    df = pd.read_csv(filename, **pd_kwargs)
    return _post_load(df, is_weighted, remove_self_loops)


def _post_load(df, is_weighted, remove_self_loops=True):
    if remove_self_loops:
        df = df.loc[df.iloc[:, 0] != df.iloc[:, 1], :]
    edges = df.iloc[:, [0, 1]].to_numpy().astype(np.int64)
    if is_weighted and df.shape[1] > 2:
        weights = df.iloc[:, 2].to_numpy().astype(np.float64)
    else:
        weights = None
    return edges, weights


def edges2spmat(edges, weights, num_nodes, directed):
    mat = sp.coo_matrix((weights, (edges[:, 1], edges[:, 0])), shape=[num_nodes, num_nodes]).tocsc()

    if not directed:
        mat = mat.maximum(mat.T)

    return mat


def edges2nx(edges, weights, num_nodes, directed):
    # type: (np.ndarray, np.ndarray, int, bool) -> nx.Graph
    if directed:
        graph = nx.DiGraph()
    else:
        graph = nx.Graph()

    graph.add_nodes_from(range(num_nodes))
    edges = [(u, v, w) for (u, v), w in zip(edges, weights)]
    graph.add_weighted_edges_from(edges)
    return graph
