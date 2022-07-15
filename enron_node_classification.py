from typing import Sequence, Literal
import time
import dataclasses as dc
import os
import argparse
import warnings

import pandas as pd
import tqdm.auto as tqdm
from sklearn.exceptions import ConvergenceWarning

import src.common as common
from synthetic_data_benchmarks import get_embeddings, BenchmarkAlg
from src.utils.node_classification import evaluate_multiple_embeddings
import src.utils.enron_data as enrondata

LabelType = Literal['email_type', 'role']


def run_eval(algs: Sequence[BenchmarkAlg], label_type: LabelType, undirected: bool = False, weighted: bool = False,
             simplified: bool = True, workdir: str = "./"):
    if label_type == "email_type":
        data_dir = f"data/enron/parsed_enron"
    elif label_type == "role":
        data_dir = f"data/enron/parsed_enron/internal_subgraph"
    else:
        raise ValueError(f"Unknown label type {label_type}")

    if undirected:
        data_file = os.path.join(data_dir, "enron_edges_undirected.tsv")
    else:
        data_file = os.path.join(data_dir, "enron_edges.tsv")

    if simplified:
        labels, unique_labels = enrondata.load_simplified_labels(data_dir=data_dir, label_type=label_type)
    else:
        labels, unique_labels = enrondata.load_labels(data_dir=data_dir, label_type=label_type)
    results = []

    for alg in tqdm.tqdm(algs):
        embeddings, duration, outcome, error_out = get_embeddings(data_file, alg, workdir, undirected=undirected,
                                                                  weighted=weighted)
        if outcome in {"timeout", "oom"}:
            continue
        elif outcome != "completed":
            raise RuntimeError(f"Getting embeddings failed with outcome {outcome}. Error: {error_out}")

        data = {"name": alg.name,
                "kemb": embeddings.shape[1],
                "duration": duration,
                "label_type": label_type}
        with warnings.catch_warnings():
            # Ignore convergence warnings caused by identical embedding vectors
            warnings.simplefilter('ignore', category=ConvergenceWarning)
            scores = evaluate_multiple_embeddings(embeddings, labels, unique_labels=unique_labels,
                                                  rskf_random_state=113)
        results.extend([dict(dc.asdict(scores_), **data) for scores_ in scores])

    return results


def main(kemb: int, undirected: bool, weighted: bool, simplified: bool, workdir: str = "./", debug: bool = False):
    label_types = ['role', 'email_type']
    file_name = f"node_classification_{'_'.join(label_types)}_{kemb}_" \
                f"{'weighted' if weighted else 'unweighted'}_" \
                f"{'undirected' if undirected else 'directed'}_" \
                f"{'simplified_' if simplified else ''}" \
                f"{'debug_' if debug else ''}" \
                f"{int(100 * time.time())}.json"

    file_name = f"node_classification_radius2_{'_'.join(label_types)}_{kemb}_" \
                f"{'weighted' if weighted else 'unweighted'}_" \
                f"{'undirected' if undirected else 'directed'}_" \
                f"{'simplified_' if simplified else ''}" \
                f"{'debug_' if debug else ''}" \
                f"{int(100 * time.time())}.json"

    results_path_roles = os.path.join(workdir, "results/enron", file_name)
    print(f"Checking results path {results_path_roles}")
    common.check_save_dir_exists(results_path_roles)

    if undirected:
        algs = [
            BenchmarkAlg.refex(kemb=kemb, tol=0.001),
            BenchmarkAlg.refex(kemb=kemb, tol=1e-6),
            BenchmarkAlg.ember(kemb=kemb),
            BenchmarkAlg.graphwave(kemb=kemb, radius=2, arctan_tranform=False, aggregate=True, num_gpus=3),
            BenchmarkAlg.graphwave(kemb=kemb, radius=3, arctan_tranform=False, aggregate=True, num_gpus=3),
            BenchmarkAlg.degree_features()
        ]
    else:
        algs = [
            BenchmarkAlg.refex(kemb=kemb, tol=0.001),
            BenchmarkAlg.refex(kemb=kemb, tol=1e-6),
            BenchmarkAlg.ember(kemb=kemb),
            BenchmarkAlg.digraphwave(kemb=kemb, radius=2, arctan_tranform=False, aggregate=True, transpose=True,
                                     normalize=False, num_gpus=3),
            BenchmarkAlg.digraphwave(kemb=kemb, radius=3, arctan_tranform=False, aggregate=True, transpose=True,
                                     normalize=False, num_gpus=3),
            BenchmarkAlg.degree_features()
        ]
    if debug:
        algs = [BenchmarkAlg.refex(kemb=kemb, tol=0.001)]

    results = []
    for label_type in label_types:
        results.extend(
            run_eval(algs, label_type=label_type, undirected=undirected,
                     weighted=weighted, simplified=simplified, workdir=workdir)
        )
    pd.DataFrame(results).to_json(results_path_roles, indent=2, orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kemb", type=int, help="Embedding sizes.")
    parser.add_argument("--weighted", action="store_true", help="Use weights")
    parser.add_argument("--undirected", action="store_true", help="Remove directions.")
    parser.add_argument("--simplified", action="store_true", help="Use simplified labels.")

    parser.add_argument("-w", "--workdir", type=str, default="./", help="Path to file where results will be saved.")
    parser.add_argument("--debug", action="store_true", help="Use debug settings.")
    args = parser.parse_args()
    main(args.kemb, undirected=args.undirected, weighted=args.weighted, simplified=args.simplified,
         workdir=args.workdir, debug=args.debug)
