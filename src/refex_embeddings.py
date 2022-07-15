import rolxrefex.refex as refex

import hashlib
import time
from collections import namedtuple

import numpy as np

import utils.data_utils as datautils
import common

RefexParam = namedtuple("RefexParam", ["kemb", "prune_tol"])


def get_param_hash(refex_param):
    md5_hash = hashlib.md5(str(refex_param).encode('utf-8')).hexdigest()
    return md5_hash


def save_embeddings(out_file, adj, weighted, directed, param, verbose=False):
    param = refex.RefexParam(max_emb_size=param.kemb, use_weights=weighted, prune_tol=param.prune_tol)
    start = time.time()
    representations, _ = refex.refex(adj, param)
    duration = time.time() - start
    np.save(out_file, representations, allow_pickle=False)

    return duration


def get_args(parser):
    parser.description = "ReFex: Recursive structural features"
    parser.add_argument("-k", "--kemb", type=int,
                        help="Desired embedding dimension. Actual dimension may be smaller."
                        )
    parser.add_argument('--tol', type=float, default=0.001, help='Feature independence tolerance')
    args = parser.parse_args()
    return args


def main():
    parser = common.get_common_parser()
    args = get_args(parser)
    common.check_save_dir_exists(args.output_file)
    filetype = common.get_filetype(args.input_file)

    adj = datautils.read_to_spmat(args.input_file, filetype=filetype, is_weighted=args.weighted,
                                  directed=not args.undirected)
    param = RefexParam(kemb=args.kemb, prune_tol=args.tol)

    args.output_file = common.get_npy_filename(args.output_file,
                                               include_time=args.include_time,
                                               include_hash=args.include_hash,
                                               param_hash=get_param_hash(param))
    duration = save_embeddings(args.output_file, adj, args.weighted, directed=not args.undirected, param=param,
                               verbose=args.verbose)
    print("::".join(["completed", args.output_file, str(duration)]))


if __name__ == "__main__":
    main()
