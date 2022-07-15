import argparse
import time

import numpy as np

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import digraphwave.graphwave as gw

import utils.data_utils as datautils
import common


def save_embeddings(out_file, adj, param, device_ids, verbose):
    start = time.time()
    embeddings = gw.graphwave(adj=adj, param=param, device_ids=device_ids, verbose=verbose)
    duration = time.time() - start
    np.save(out_file, embeddings, allow_pickle=False)
    return duration


def args2parameters(adj, args):
    if args.use_legacy:
        param = gw.GraphwaveHyperparameters.legacy(num_nodes=adj.shape[0], num_edges=adj.nnz, k_emb=args.kemb,
                                                   batch_size=args.batch_size, order=args.order)
    else:
        param = gw.GraphwaveHyperparameters.as_digraphwave(num_nodes=adj.shape[0], num_edges=adj.nnz, R=args.radius,
                                                           k_emb=args.kemb,
                                                           batch_size=args.batch_size,
                                                           arctan_log_transform=not args.no_transform,
                                                           a_flag=args.aggregate,
                                                           order=args.order)
    return param


def get_args(parser: argparse.ArgumentParser):
    parser.add_argument("-r", "--radius", type=int,
                        help="Hyperparameter R."
                        )
    parser.add_argument("-k", "--kemb", type=int,
                        help="Desired embedding dimension. Actual dimension may be smaller."
                        )
    parser.add_argument("-b", "--batch_size", type=int, default=None,
                        help="Desired embedding dimension. Actual dimension may be smaller."
                        )
    parser.add_argument("--order", type=int, default=40)
    parser.add_argument("--no_transform", action="store_true", default=False,
                        help="Do not apply arctan log transform."
                        )
    parser.add_argument("--aggregate", action="store_true", default=False,
                        help="Enhance embeddings by also computing aggregations of neighbours embeddings."
                        )
    parser.add_argument("--num_gpu", type=int, default=0,
                        help="Number of gpus to use. 0 means cpu."
                        )
    parser.add_argument("--use_legacy", action="store_true", default=False,
                        help="Use parameter to mimic choices by Graphwave authors."
                        )
    args = parser.parse_args()
    return args


def main():
    parser = common.get_common_parser()
    args = get_args(parser)
    common.check_save_dir_exists(args.output_file)
    filetype = common.get_filetype(args.input_file)

    adj = datautils.read_to_spmat(args.input_file, filetype=filetype, is_weighted=args.weighted,
                                  directed=not args.undirected)
    param = args2parameters(adj, args)

    args.output_file = common.get_npy_filename(args.output_file,
                                               include_time=args.include_time,
                                               include_hash=args.include_hash,
                                               param_hash=param.to_hash())
    device_ids = list(range(args.num_gpu)) if args.num_gpu > 0 else None
    duration = save_embeddings(args.output_file, adj, param, device_ids=device_ids, verbose=args.verbose)
    print("::".join(["completed", args.output_file, str(duration)]))


if __name__ == "__main__":
    main()
