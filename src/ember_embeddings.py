from __future__ import print_function
import hashlib
import time
from collections import namedtuple

import numpy as np
import networkx as nx

import algorithms.EMBER.src.main as ember_main
import algorithms.EMBER.src.rep_method as ember_rep_method

import utils.data_utils as datautils
import common

EmberParam = namedtuple("EmberParam", ["p", "alpha", "gamma"])


def get_param_hash(ember_param):
    md5_hash = hashlib.md5(str(ember_param).encode('utf-8')).hexdigest()
    return md5_hash


def save_embeddings(out_file, nx_graph, weighted, directed, param, verbose=False):
    start = time.time()
    representations = _call_ember(nx_graph, weighted, directed, param=param, verbose=verbose)
    duration = time.time() - start
    np.save(out_file, representations, allow_pickle=False)

    return duration


def _call_ember(nx_graph, weighted, directed, param, verbose=False):
    num_nodes = nx_graph.number_of_nodes()
    adj_matrix = nx.adjacency_matrix(nx_graph, nodelist=range(num_nodes))  # .todense()
    if not weighted:
        adj_matrix[np.nonzero(adj_matrix)] = 1

    graph = ember_main.Graph(adj_matrix, weighted=weighted, directed=directed)

    nodes_to_embed = list(range(num_nodes))

    rep_method = ember_rep_method.RepMethod(method="xnetmf", max_layer=2, num_buckets=2, p=param.p,
                                            alpha=param.alpha, gamma=param.gamma,
                                            sampling_method="degree", sampling_prob="top", normalize=True)
    graph.compute_node_features(rep_method.binning_features)
    representations = ember_main.get_representations(graph, rep_method, verbose=verbose, nodes_to_embed=nodes_to_embed)

    return representations


def get_args(parser):
    parser.description = "EMBER: Inferring professional roles in email networks."
    parser.add_argument('--p', type=int, help='Number of landmarks, also the embedding dimension')
    parser.add_argument('--alpha', type=int, default=0.1, help='Decay factor')
    parser.add_argument('--gamma', type=int, default=1.0, help='Similarity constant')
    args = parser.parse_args()
    return args


def main():
    parser = common.get_common_parser()
    args = get_args(parser)
    common.check_save_dir_exists(args.output_file)
    filetype = common.get_filetype(args.input_file)

    nx_graph = datautils.read_to_nx(args.input_file, filetype=filetype, is_weighted=args.weighted,
                                    directed=not args.undirected)
    param = EmberParam(p=args.p, alpha=args.alpha, gamma=args.gamma)

    args.output_file = common.get_npy_filename(args.output_file,
                                               include_time=args.include_time,
                                               include_hash=args.include_hash,
                                               param_hash=get_param_hash(param))
    duration = save_embeddings(args.output_file, nx_graph, args.weighted, directed=not args.undirected, param=param,
                               verbose=args.verbose)
    print("::".join(["completed", args.output_file, str(duration)]))


if __name__ == "__main__":
    main()
