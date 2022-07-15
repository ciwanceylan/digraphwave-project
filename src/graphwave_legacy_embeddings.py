from __future__ import print_function
import hashlib
import time
from collections import namedtuple

import numpy as np

import algorithms.graphwave.graphwave.graphwave as gw

import utils.data_utils as datautils
import common

LegacyGwParam = namedtuple("LegacyGwParam", ["k_emb"])


def get_param_hash(gw_param):
    md5_hash = hashlib.md5(str(gw_param).encode('utf-8')).hexdigest()
    return md5_hash


def save_embeddings(out_file, nx_graph, param, verbose=False):
    start = time.time()
    representations = _call_graphwave(nx_graph, param=param, verbose=verbose)
    duration = time.time() - start
    np.save(out_file, representations, allow_pickle=False)

    return duration


def _call_graphwave(nx_graph, param, verbose=False):
    num_timepoints = int(param.k_emb / 2. / gw.NB_FILTERS)
    time_pnts = np.linspace(0, 100, num_timepoints)
    embeddings, _, _ = gw.graphwave_alg(nx_graph, time_pnts=time_pnts, verbose=verbose)

    return embeddings


def get_args(parser):
    parser.description = "Graphwave implementation from https://github.com/snap-stanford/graphwave."
    parser.add_argument('--kemb', type=int,
                        help="Desired embedding dimension. Actual dimension may be smaller.")
    args = parser.parse_args()
    return args


def main():
    parser = common.get_common_parser()
    args = get_args(parser)
    common.check_save_dir_exists(args.output_file)
    filetype = common.get_filetype(args.input_file)

    nx_graph = datautils.read_to_nx(args.input_file, filetype=filetype, is_weighted=args.weighted,
                                    directed=not args.undirected)
    param = LegacyGwParam(k_emb=args.kemb)

    args.output_file = common.get_npy_filename(args.output_file,
                                               include_time=args.include_time,
                                               include_hash=args.include_hash,
                                               param_hash=get_param_hash(param))
    duration = save_embeddings(args.output_file, nx_graph, param=param, verbose=args.verbose)
    print("::".join(["completed", args.output_file, str(duration)]))


if __name__ == "__main__":
    main()
