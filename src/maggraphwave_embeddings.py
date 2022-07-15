import argparse
import time

import numpy as np

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import digraphwave.maggraphwave as maggw

import utils.data_utils as datautils
import common


def save_embeddings(out_file, adj, param):
    start = time.time()
    embeddings = maggw.maggraphwave(adj=adj, param=param)
    duration = time.time() - start
    np.save(out_file, embeddings, allow_pickle=False)
    return duration


def args2parameters(adj, args):
    if args.use_furutani:
        param = maggw.MaggraphwaveHyperparameters.furutani_et_al(num_nodes=adj.shape[0], num_edges=adj.nnz,
                                                                 k_emb=args.kemb, batch_size=args.batch_size,
                                                                 order=args.order
                                                                 )
    else:
        param = maggw.MaggraphwaveHyperparameters.as_digraphwave(num_nodes=adj.shape[0], num_edges=adj.nnz,
                                                                 R=args.radius, k_emb=args.kemb,
                                                                 batch_size=args.batch_size,
                                                                 a_flag=args.aggregate,
                                                                 order=args.order
                                                                 )
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
    parser.add_argument("--no_transform", action="store_false", default=True,
                        help="Do not apply arctan log transform."
                        )
    parser.add_argument("--aggregate", action="store_true", default=False,
                        help="Enhance embeddings by also computing aggregations of neighbours embeddings."
                        )
    parser.add_argument("--use_furutani", action="store_true", default=False,
                        help="Use parameters to mimic choices by Furutani et al (authors of 'MagGraphwave')."
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
    duration = save_embeddings(args.output_file, adj, param)
    print("::".join(["completed", args.output_file, str(duration)]))


if __name__ == "__main__":
    main()
