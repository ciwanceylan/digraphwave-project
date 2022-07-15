import argparse
import os
import time


def check_save_dir_exists(out_file, fix=True):
    if not os.path.exists(os.path.dirname(out_file)):
        if fix:
            os.makedirs(os.path.dirname(out_file), exist_ok=True)
        else:
            raise OSError("Directory of " + out_file + " does not exist.")


def get_filetype(input_file):
    filetype = input_file.split(".")[-1]
    if filetype not in {"tsv", "csv", "parquet", "edgelist"}:
        raise ValueError("File type " + filetype + " not supported")
    return filetype


def get_npy_filename(output_file, include_time, include_hash, param_hash):
    if output_file[-4:] == ".npy":
        output_file = output_file[:-4]
    if include_time:
        output_file += "_time" + str(int(1000 * time.time()))
    if include_hash:
        output_file += "_param" + param_hash
    output_file += ".npy"
    return output_file


def get_common_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", type=str,
                        help="File containing the graph. Either tsv or csv. "
                             "Optionally with number of nodes as comment on the first line.",
                        )
    parser.add_argument("output_file", type=str,
                        help="Where embeddings will be saved. Numpy .npy files are used.",
                        )
    parser.add_argument("--undirected", action="store_true", default=False,
                        help="Treat the graph as undirected")
    parser.add_argument("--weighted", action="store_true", default=False,
                        help="Use edge weights.")
    parser.add_argument("--include_time", action="store_true", default=False,
                        help="Append current time to output file name")
    parser.add_argument("--include_hash", action="store_true", default=False,
                        help="Append hash of parameters to output file name")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Currently not implemented for all algorithms ")
    return parser
