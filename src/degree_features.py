import rolxrefex.refex_sparse as refex_core
import time
import numpy as np

import utils.data_utils as datautils
import common


def save_embeddings(out_file, adj, weighted, directed, param, verbose=False):
    start = time.time()
    representations = refex_core.extract_node_features(adj, use_weights=weighted)
    duration = time.time() - start
    np.save(out_file, representations, allow_pickle=False)

    return duration


def get_args(parser):
    parser.description = "In and out degrees."
    args = parser.parse_args()
    return args


def main():
    parser = common.get_common_parser()
    args = get_args(parser)
    common.check_save_dir_exists(args.output_file)
    filetype = common.get_filetype(args.input_file)

    adj = datautils.read_to_spmat(args.input_file, filetype=filetype, is_weighted=args.weighted,
                                  directed=not args.undirected)

    args.output_file = common.get_npy_filename(args.output_file,
                                               include_time=args.include_time,
                                               include_hash=args.include_hash,
                                               param_hash=str(args.weighted))
    duration = save_embeddings(args.output_file, adj, args.weighted, directed=not args.undirected, param=None,
                               verbose=args.verbose)
    print("::".join(["completed", args.output_file, str(duration)]))


if __name__ == "__main__":
    main()
