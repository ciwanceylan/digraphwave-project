from typing import Sequence
import time
import os
import tempfile
import argparse
import dataclasses as dc
import warnings

import numpy as np
import pandas as pd
import scipy.spatial.distance as spdist
from sklearn.exceptions import ConvergenceWarning

import tqdm.auto as tqdm

import src.common as common
import src.utils.node_classification as evalutil
import src.utils.synthetic_dataset as synthdata
import src.utils.data_utils as data_utils
from src.utils.misc_utils import mk_digraphwave_name, mk_graphwave_name, run_command
import src.algorithms.maxsim as maxsim


@dc.dataclass(frozen=True)
class BenchmarkAlg:
    name: str
    conda_env_name: str
    run_file: str
    optional_args: str

    def get_command_list(self, in_file_name: str, out_file_name: str, undirected: bool = False, weighted: bool = False):
        # if out_file_name is None:
        #     out_file_name = "./tmp/benchmark_outputs"

        undirected = "--undirected" if undirected else ""
        weighted = "--weighted" if weighted else ""

        conda_command = f"conda run -n {self.conda_env_name}".split()
        python_command = f"python {self.run_file}".split()
        args = f"{in_file_name} {out_file_name} {self.optional_args.strip(' ')} {undirected} {weighted}".split()
        command = conda_command + python_command + args
        return command

    @classmethod
    def graphwave(cls, kemb: int, radius=3, arctan_tranform=True, aggregate=True, num_gpus=0):
        conda_env_name = "digraphwave_experiment_python3_env"
        run_file = "src/graphwave_embeddings.py"
        name = mk_graphwave_name(kemb=kemb, radius=radius, arctan_tranform=arctan_tranform, aggregate=aggregate,
                                 num_gpus=num_gpus)
        arctan_transform_s = "--no_transform" if not arctan_tranform else ""
        aggregate = "--aggregate" if aggregate else ""
        optional_args = f"-r {radius} --kemb {kemb} {arctan_transform_s} {aggregate} --verbose --num_gpu {num_gpus}"
        return cls(name, conda_env_name, run_file, optional_args)

    @classmethod
    def maggraphwave(cls, kemb: int):
        conda_env_name = "digraphwave_experiment_python3_env"
        run_file = "src/maggraphwave_embeddings.py"
        optional_args = f"--kemb {kemb} --use_furutani"
        return cls("furutani_maggraphwave", conda_env_name, run_file, optional_args)

    @classmethod
    def digraphwave(cls, kemb: int, radius=3, arctan_tranform=True, aggregate=True, transpose=False, normalize=False,
                    num_gpus=0):
        conda_env_name = "digraphwave_experiment_python3_env"
        run_file = "src/digraphwave_embeddings.py"
        name = mk_digraphwave_name(kemb=kemb, radius=radius, arctan_tranform=arctan_tranform, aggregate=aggregate,
                                   transpose=transpose,
                                   normalize=normalize, num_gpus=num_gpus)
        arctan_transform_s = "--no_transform" if not arctan_tranform else ""
        aggregate_s = "--aggregate" if aggregate else ""
        transpose_s = "--transpose" if transpose else ""
        normalize_s = "--normalize" if normalize else ""
        optional_args = f"-r {radius} --kemb {kemb} {arctan_transform_s} {aggregate_s} {transpose_s} {normalize_s} --num_gpu {num_gpus}"
        return cls(name, conda_env_name, run_file, optional_args)

    @classmethod
    def graphwave_legacy(cls, kemb: int):
        conda_env_name = "digraphwave_experiment_python2_env"
        run_file = "src/graphwave_legacy_embeddings.py"
        optional_args = f"--kemb {kemb}"
        return cls("graphwave2018", conda_env_name, run_file, optional_args)

    @classmethod
    def ember(cls, kemb: int):
        conda_env_name = "digraphwave_experiment_python2_env"
        run_file = "src/ember_embeddings.py"
        optional_args = f"--p {kemb}"
        return cls(f"ember{kemb}", conda_env_name, run_file, optional_args)

    @classmethod
    def refex(cls, kemb: int, tol: float):
        conda_env_name = "digraphwave_experiment_python3_env"
        run_file = "src/refex_embeddings.py"
        optional_args = f"--kemb {2 * kemb}  --tol {tol}"
        log10tol = int(np.log10(tol))
        return cls(f"refexnew_{kemb}_{log10tol}", conda_env_name, run_file, optional_args)

    @classmethod
    def degree_features(cls):
        conda_env_name = "digraphwave_experiment_python3_env"
        run_file = "src/degree_features.py"
        optional_args = ""
        return cls(f"degreefeatures", conda_env_name, run_file, optional_args)


def get_embeddings(data_file: str, alg: BenchmarkAlg, temp_dir: str, undirected=False, weighted=False):
    embeddings = None
    duration = None
    with tempfile.NamedTemporaryFile(suffix='.npy', dir=temp_dir, delete=True) as f:
        command = alg.get_command_list(in_file_name=data_file, out_file_name=f.name,
                                       undirected=undirected, weighted=weighted)
        duration, outcome, error_out = run_command(command, timeout_time=2000)
        if outcome == "completed":
            f.seek(0)
            embeddings = np.load(f)

    return embeddings, duration, outcome, error_out


def run_eval_on_composed_graph(algs: Sequence[BenchmarkAlg], num_connecting_edges: int, rep: int,
                               as_undirected: bool = False, workdir: str = "./", ):
    data_dir = f"data/synthetic/composed/cycle_main_10_{num_connecting_edges}-{rep}/"

    data_file = os.path.join(data_dir, "composed_graph.edgelist")
    train_labels = pd.read_json(os.path.join(data_dir, "train_labels.json"), typ="series")
    test_labels = pd.read_json(os.path.join(data_dir, "test_labels.json"), typ="series")
    combinded_labels = pd.concat((train_labels, test_labels), axis=0)
    unique_labels = synthdata.get_unique_labels()

    results = []
    adj = data_utils.read_to_spmat(data_file, "edgelist", is_weighted=False, directed=True)

    maxsim_embeddings = maxsim.maxsim_embeddings(adj, unweighted=True)
    maxsim_euclidean_distances = spdist.pdist(maxsim_embeddings[combinded_labels.index, :], metric="euclidean")
    maxsim_cosine_distances = spdist.pdist(maxsim_embeddings[combinded_labels.index, :], metric="cosine")

    for alg in tqdm.tqdm(algs):
        embeddings, duration, outcome, error_out = get_embeddings(data_file, alg, workdir, undirected=as_undirected)
        if outcome in {"timeout", "oom"}:
            continue
        elif outcome != "completed":
            raise RuntimeError(f"Getting embeddings failed with outcome {outcome}. Error: {error_out}")
        maxsim_correlations = maxsim.evaluate_maxsim_kendal_coeffs(
            maxsim_euclidean_distances=maxsim_euclidean_distances,
            maxsim_cosine_distances=maxsim_cosine_distances,
            embeddings=embeddings,
            which_nodes=combinded_labels.index
        )
        data = {"name": alg.name if not as_undirected else alg.name + '::undirected',
                "kemb": embeddings.shape[1],
                "duration": duration,
                "num_connecting_edges": num_connecting_edges}
        data.update(dc.asdict(maxsim_correlations))

        with warnings.catch_warnings():
            # Ignore convergence warnings caused by identical embedding vectors
            warnings.simplefilter('ignore', category=ConvergenceWarning)
            classification_scores = evalutil.evaluate_emb_logistic_regression(embeddings, train_labels, test_labels,
                                                                              unique_labels=unique_labels)
            data.update(dc.asdict(classification_scores))
            clustering_scores = evalutil.evaluate_emb_kmeans(embeddings, labels=combinded_labels)
            data.update(dc.asdict(clustering_scores))

        results.append(data)

    return results


def run_eval_on_all_composed_graphs(results_path: str, algs: Sequence[BenchmarkAlg],
                                    num_reps: int, algs_undirected: Sequence[BenchmarkAlg] = None,
                                    workdir: str = "./", ):
    all_results = []
    for num_connecting_edges in tqdm.trange(6):
        for rep in range(num_reps):
            results = run_eval_on_composed_graph(algs, num_connecting_edges=num_connecting_edges, rep=rep,
                                                 as_undirected=False, workdir=workdir)
            all_results.extend(results)
            if algs_undirected is not None:
                results = run_eval_on_composed_graph(algs_undirected, num_connecting_edges=num_connecting_edges,
                                                     rep=rep, as_undirected=True, workdir=workdir)
                all_results.extend(results)

            df = pd.DataFrame(all_results)
            df.to_json(results_path, indent=2, orient="records")


def main_compare_algorithms(workdir: str = "./", debug: bool = False):
    results_path = os.path.join(workdir, f"results/synthetic/alg_compare_{int(100 * time.time())}.json")
    print(f"Checking results path {results_path}")
    common.check_save_dir_exists(results_path)

    algs = [
        BenchmarkAlg.digraphwave(kemb=128, radius=3, arctan_tranform=False, aggregate=True, transpose=True,
                                 normalize=False, num_gpus=0),
        BenchmarkAlg.graphwave_legacy(kemb=128),
        BenchmarkAlg.ember(kemb=128),
        BenchmarkAlg.maggraphwave(kemb=128),
        BenchmarkAlg.refex(kemb=128, tol=1e-6),
        BenchmarkAlg.degree_features()
    ]
    num_reps = 5

    if debug:
        algs = [BenchmarkAlg.degree_features()]
        num_reps = 1

    run_eval_on_all_composed_graphs(results_path, algs, num_reps=num_reps, workdir=workdir)


def main_ablation(workdir: str = "./", debug: bool = False):
    results_path = os.path.join(workdir, f"results/synthetic/digw_ablation_{int(100 * time.time())}.json")
    print(f"Checking results path {results_path}")
    common.check_save_dir_exists(results_path)

    algs = [
        BenchmarkAlg.graphwave(kemb=128, radius=3, arctan_tranform=False, aggregate=False, num_gpus=0),
        BenchmarkAlg.graphwave(kemb=128, radius=3, arctan_tranform=False, aggregate=True, num_gpus=0),
        BenchmarkAlg.digraphwave(kemb=128, radius=3, arctan_tranform=False, aggregate=False, transpose=False,
                                 num_gpus=0),
        BenchmarkAlg.digraphwave(kemb=128, radius=3, arctan_tranform=False, aggregate=True, transpose=False,
                                 num_gpus=0),
        BenchmarkAlg.digraphwave(kemb=128, radius=3, arctan_tranform=False, aggregate=False, transpose=True,
                                 num_gpus=0),
        BenchmarkAlg.digraphwave(kemb=128, radius=3, arctan_tranform=False, aggregate=True, transpose=True, num_gpus=0),
    ]

    algs_undirected = [
        BenchmarkAlg.graphwave(kemb=128, radius=3, arctan_tranform=False, aggregate=False, num_gpus=0),
        BenchmarkAlg.graphwave(kemb=128, radius=3, arctan_tranform=False, aggregate=True, num_gpus=0),
    ]

    num_reps = 5

    if debug:
        algs = [BenchmarkAlg.digraphwave(kemb=128, arctan_tranform=True, aggregate=True, transpose=True, num_gpus=0)]
        num_reps = 1
        algs_undirected = []

    run_eval_on_all_composed_graphs(results_path, algs, num_reps=num_reps, algs_undirected=algs_undirected,
                                    workdir=workdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("main", type=str, default=None, help="Which main function to run.")
    parser.add_argument("-w", "--workdir", type=str, default="./", help="Path to file where results will be saved.")
    parser.add_argument("--debug", action="store_true", help="Use debug settings.")
    args = parser.parse_args()
    if args.main.lower() == "algs":
        main_compare_algorithms(workdir=args.workdir, debug=args.debug)
    elif args.main.lower() == "ablation":
        main_ablation(workdir=args.workdir, debug=args.debug)
