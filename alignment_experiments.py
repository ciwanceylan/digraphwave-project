from typing import Sequence, Literal
import argparse
import time
import os
import tempfile
import tqdm.auto as tqdm

import pandas as pd

import src.common as common
from synthetic_data_benchmarks import get_embeddings, BenchmarkAlg

import src.utils.alignment as alignmentutils
import src.utils.data_utils as datautils

ExperimentLiteral = Literal['arxiv',
                            'enron_all',
                            'enron_internal',
                            'enron_external',
                            'enron_all_internal',
                            'enron_all_external']


def parse_experiment_name(experiment: ExperimentLiteral):
    g1_name = None
    g2_name = None
    if experiment == "arxiv":
        dataset = "arxiv"
        g1_name = "arxiv"
    else:
        dataset = "enron"
        out = experiment.split("_")
        if len(out) > 2:
            g1_name = out[1]
            g2_name = out[2]
        else:
            g1_name = out[1]
            g2_name = None

    return dataset, g1_name, g2_name


def load_arxiv():
    num_nodes, comment_char = datautils.try_read_num_nodes("./data/arxiv/arxiv_edges.tsv")
    edges, _ = datautils.read_edges("./data/arxiv/arxiv_edges.tsv", filetype="tsv", is_weighted=False,
                                    comment=comment_char)

    return alignmentutils.SimpleGraph(num_nodes=num_nodes, edges=edges)


def load_enron(split: str, weighted: bool, undirected: bool = False):
    undir_str = "_undirected" if undirected else ""
    datadir = "data/enron/parsed_enron"
    if split == "internal":
        datadir = os.path.join(datadir, "internal_subgraph")
    elif split == "external":
        datadir = os.path.join(datadir, "external_subgraph")

    edge_file = os.path.join(datadir, f"enron_edges{undir_str}.tsv")

    num_nodes, comment_char = datautils.try_read_num_nodes(edge_file)
    edges, weights = datautils.read_edges(edge_file, filetype="tsv", is_weighted=True, comment=comment_char)
    if not weighted:
        weights = None
    return alignmentutils.SimpleGraph(num_nodes=num_nodes, edges=edges, weights=weights)


def load_enron_alignment(g2_name: str):
    datadir = "data/enron/parsed_enron"
    if g2_name == "internal":
        datadir = os.path.join(datadir, "internal_subgraph")
    elif g2_name == "external":
        datadir = os.path.join(datadir, "external_subgraph")
    else:
        raise ValueError(f"g2_name = {g2_name} not supported. Only 'internal' or 'external' ")
    alignment_obj = alignmentutils.AlignedGraphs.load_from_file(os.path.join(datadir, f"node_index_alignment.csv"))
    return alignment_obj


def get_alignment_data(dataset: str, g1_name: str, g2_name: str, weighted: bool, undirected: bool, noise_p: float):
    actual_p = 0.
    if dataset == "enron":
        g1 = load_enron(g1_name, weighted=weighted, undirected=undirected)
        if g2_name is not None:
            g2 = load_enron(g2_name, weighted=weighted)
            alignment_obj = load_enron_alignment(g2_name)
        else:
            g2, alignment_obj = alignmentutils.create_permuted(g1)
    elif dataset == "arxiv":
        g1 = load_arxiv()
        g2, alignment_obj = alignmentutils.create_permuted(g1)
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    merged_g = alignmentutils.SimpleGraph.union(g1, g2)
    if noise_p > 0.:
        merged_g, actual_p = merged_g.add_noise_edges(merged_g, noise_p)
    return merged_g, alignment_obj, actual_p


def run_eval(experiment: ExperimentLiteral, algs: Sequence[BenchmarkAlg], undirected: bool = False,
             weighted: bool = False, workdir: str = "./", debug: bool = False):
    dataset, g1_name, g2_name = parse_experiment_name(experiment)

    num_reps = 5
    noise_levels = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1]

    if debug:
        num_reps = 1
        noise_levels = [0.01]
    all_results = []
    bad_outcomes_count = {alg.name: 0 for alg in algs}

    for noise_p in tqdm.tqdm(noise_levels):
        for rep in tqdm.trange(num_reps):
            graph, alignment_obj, noise_p = get_alignment_data(dataset, g1_name, g2_name,
                                                               weighted=weighted, undirected=undirected,
                                                               noise_p=noise_p)
            with tempfile.NamedTemporaryFile(suffix='.tsv', dir=workdir, delete=True) as fp:
                graph.save(fp)
                for alg in algs:
                    fp.seek(0)
                    if bad_outcomes_count[alg.name] > 4:
                        continue
                    embeddings, duration, outcome, error_out = get_embeddings(fp.name, alg=alg, temp_dir=workdir,
                                                                              undirected=undirected, weighted=weighted)
                    data = {"name": alg.name,
                            "duration": duration,
                            "noise_p": noise_p,
                            "outcome": outcome
                            }
                    if embeddings is not None:
                        results = alignmentutils.eval_topk_sim(embeddings, alignment_obj)
                        data.update({f"k@{k}": val for k, val in results.items()})
                    else:
                        print(f"Outcome {outcome} for {alg.name}. Error {error_out}")
                        bad_outcomes_count[alg.name] += 1
                    all_results.append(data)
    return all_results


def main(experiment: ExperimentLiteral, undirected: bool, weighted: bool, workdir: str, debug: bool = False):
    if experiment == "arxiv":
        weighted = False
        undirected = True
    file_name = f"alignment_{experiment}_" \
                f"{'weighted' if weighted else 'unweighted'}_" \
                f"{'undirected' if undirected else 'directed'}_" \
                f"{int(100 * time.time())}.json"
    results_path_roles = os.path.join(workdir, "results/alignment", file_name)
    print(f"Checking results path {results_path_roles}")
    common.check_save_dir_exists(results_path_roles)

    if undirected:
        algs = [
            BenchmarkAlg.refex(kemb=128, tol=0.001),
            BenchmarkAlg.refex(kemb=128, tol=1e-6),
            BenchmarkAlg.ember(kemb=128),
            BenchmarkAlg.graphwave(kemb=128, radius=2, arctan_tranform=False, aggregate=True, num_gpus=3),
            BenchmarkAlg.graphwave(kemb=128, radius=3, arctan_tranform=False, aggregate=True, num_gpus=3),
            BenchmarkAlg.degree_features()
        ]
    else:
        algs = [
            BenchmarkAlg.refex(kemb=128, tol=0.001),
            BenchmarkAlg.refex(kemb=128, tol=1e-6),
            BenchmarkAlg.ember(kemb=128),
            BenchmarkAlg.digraphwave(kemb=128, radius=2, arctan_tranform=False, aggregate=True, transpose=True,
                                     normalize=False, num_gpus=3),
            BenchmarkAlg.digraphwave(kemb=128, radius=3, arctan_tranform=False, aggregate=True, transpose=True,
                                     normalize=False, num_gpus=3),
            BenchmarkAlg.degree_features()
        ]
    if debug:
        algs = [BenchmarkAlg.refex(kemb=128, tol=1e-3), BenchmarkAlg.refex(kemb=128, tol=1e-3)]

    results = run_eval(experiment=experiment, algs=algs, undirected=undirected, weighted=weighted, workdir=workdir,
                       debug=debug)
    pd.DataFrame(results).to_json(results_path_roles, indent=2, orient="records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Which experiment.")
    parser.add_argument("--weighted", action="store_true", help="Use weights")
    parser.add_argument("--undirected", action="store_true", help="Remove directions.")

    parser.add_argument("-w", "--workdir", type=str, default="./", help="Path to file where results will be saved.")
    parser.add_argument("--debug", action="store_true", help="Use debug settings.")
    args = parser.parse_args()
    main(args.experiment, undirected=args.undirected, weighted=args.weighted, workdir=args.workdir, debug=args.debug)
