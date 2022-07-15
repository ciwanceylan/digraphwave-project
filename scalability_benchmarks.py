from typing import Sequence
import time
import os
import tempfile
import argparse
import dataclasses as dc

import numpy as np
import pandas as pd
import graph_tool.generation as gtgen

from tqdm.auto import tqdm

import src.common as common
from src.utils.misc_utils import mk_digraphwave_name, mk_graphwave_name, run_command


@dc.dataclass(frozen=True)
class BenchmarkAlg:
    name: str
    conda_env_name: str
    run_file: str
    optional_args: str

    def get_command_list(self, in_file_name: str, undirected: bool, weighted: bool = False, out_file_name: str = None):
        if out_file_name is None:
            out_file_name = "./tmp/benchmark_outputs"

        undirected = "--undirected" if undirected else ""
        weighted = "--weighted" if weighted else ""

        conda_command = f"conda run -n {self.conda_env_name}".split()
        python_command = f"python {self.run_file}".split()
        args = f"{in_file_name} {out_file_name} {self.optional_args.strip(' ')} {undirected} {weighted}".split()
        command = conda_command + python_command + args
        return command

    @classmethod
    def graphwave(cls, aggregate=True, num_gpus=0):
        conda_env_name = "digraphwave_experiment_python3_env"
        run_file = "src/graphwave_embeddings.py"
        name = mk_graphwave_name(kemb=64, radius=4, aggregate=aggregate, num_gpus=num_gpus)
        aggregate = "--aggregate" if aggregate else ""
        optional_args = f"-r 4 --kemb 64 {aggregate} --verbose --num_gpu {num_gpus}"
        return cls(name, conda_env_name, run_file, optional_args)

    @classmethod
    def maggraphwave(cls):
        conda_env_name = "digraphwave_experiment_python3_env"
        run_file = "src/maggraphwave_embeddings.py"
        optional_args = f"--kemb 64 --use_furutani"
        return cls("furutani_maggraphwave", conda_env_name, run_file, optional_args)

    @classmethod
    def digraphwave(cls, aggregate=True, transpose=False, normalize=False, num_gpus=0):
        conda_env_name = "digraphwave_experiment_python3_env"
        run_file = "src/digraphwave_embeddings.py"
        name = mk_digraphwave_name(kemb=64, radius=4, aggregate=aggregate, transpose=transpose, normalize=normalize,
                                   num_gpus=num_gpus)
        aggregate_s = "--aggregate" if aggregate else ""
        transpose_s = "--transpose" if transpose else ""
        normalize_s = "--normalize" if normalize else ""
        optional_args = f"-r 4 --kemb 64 {aggregate_s} {transpose_s} {normalize_s} --verbose --num_gpu {num_gpus}"
        return cls(name, conda_env_name, run_file, optional_args)

    @classmethod
    def graphwave_legacy(cls):
        conda_env_name = "digraphwave_experiment_python2_env"
        run_file = "src/graphwave_legacy_embeddings.py"
        optional_args = f"--kemb 64"
        return cls("graphwave2018", conda_env_name, run_file, optional_args)

    @classmethod
    def ember(cls):
        conda_env_name = "digraphwave_experiment_python2_env"
        run_file = "src/ember_embeddings.py"
        optional_args = f"--p 64"
        return cls("ember", conda_env_name, run_file, optional_args)

    @classmethod
    def refex(cls, tol: float):
        conda_env_name = "digraphwave_experiment_python3_env"
        run_file = "src/refex_embeddings.py"
        optional_args = f"--kemb 128  --tol {tol}"
        log10tol = int(np.log10(tol))
        return cls(f"refexnew_{log10tol}", conda_env_name, run_file, optional_args)


def generate_ba_graph(file, num_nodes: int, m: int):
    graph = gtgen.price_network(num_nodes, m=m, directed=False)
    edges = pd.DataFrame(({'source': int(s), 'target': int(t)} for s, t in graph.edges()))
    edges.to_csv(file, index=False, header=False)
    return graph.num_edges()


def _run_on_ba_graph(num_nodes: int, m: int, algs: Sequence[BenchmarkAlg], as_undirected: bool, timeout: float,
                     temp_dir: str = None):
    data = []
    timed_out = set()
    oom_set = set()
    with tempfile.NamedTemporaryFile(suffix='.csv', dir=temp_dir, delete=True) as f:
        num_edges = generate_ba_graph(f, num_nodes, m)
        f.seek(0)
        for alg in algs:
            command = alg.get_command_list(f.name, undirected=as_undirected)
            start = time.time()
            duration, outcome, error_out = run_command(command, timeout_time=timeout)
            subprocess_duration = time.time() - start
            data.append({"num_nodes": num_nodes, "m": m, "num_edges": num_edges, "alg": alg.name,
                         "duration": duration, "subprocess_duration": subprocess_duration,
                         "outcome": outcome, "error": error_out,
                         "undirected": as_undirected,
                         "command": " ".join(command)})
            if outcome.lower() == "timeout" or outcome.lower() == "oom":
                timed_out.add(alg.name)
            elif outcome.lower() == "oom":
                oom_set.add(alg.name)
        f.close()

    return data, timed_out, oom_set


def run_on_ba_graphs(num_nodes, m_values, num_reps, algs, timeout: float, results_path: str, tmpfile_dir: str = None):
    exclude_algs = set()
    all_data = []
    print(f"Starting benchmarks")
    for n in tqdm(num_nodes):
        for m in tqdm(m_values):
            for rep in range(num_reps):
                algs_to_run = [alg for alg in algs if alg.name not in exclude_algs]
                if algs_to_run:
                    data, timed_out, oom_set = _run_on_ba_graph(num_nodes=n, m=m, algs=algs_to_run, as_undirected=True,
                                                                timeout=timeout, temp_dir=tmpfile_dir)
                    if timed_out:
                        print(f"timed_out: {timed_out}")
                    if oom_set:
                        print(f"Out of memory: {oom_set}")
                    exclude_algs.update(timed_out)
                    exclude_algs.update(oom_set)
                    all_data.extend(data)
        df = pd.DataFrame(all_data)
        df.to_json(results_path, indent=2, orient="records")


def main_compare_algorithms(timeout: float, workdir: str = "./", debug: bool = False):
    results_path = os.path.join(workdir, f"results/scalability/alg_compare_{int(100 * time.time())}.json")
    print(f"Checking results path {results_path}")
    common.check_save_dir_exists(results_path)

    algs = [
        BenchmarkAlg.digraphwave(aggregate=True, transpose=False, normalize=False, num_gpus=0),
        BenchmarkAlg.digraphwave(aggregate=True, transpose=False, normalize=False, num_gpus=1),
        BenchmarkAlg.graphwave_legacy(),
        BenchmarkAlg.ember(),
        BenchmarkAlg.maggraphwave(),
        BenchmarkAlg.refex(tol=0.001),
        BenchmarkAlg.refex(tol=1e-6)
    ]
    num_nodes = [100, 1000, 10000, 100000, 1000000]
    m_values = [1, 5, 10]
    num_reps = 5
    if debug:
        num_nodes = [100]  # for debug
        m_values = [2]
        num_reps = 1

    run_on_ba_graphs(num_nodes, m_values, num_reps=num_reps, algs=algs, timeout=timeout, results_path=results_path,
                     tmpfile_dir=workdir)


def main_compare_digraphwave_settings(timeout: float, workdir: str = "./", debug: bool = False):
    results_path = os.path.join(workdir, f"results/scalability/digw_compare_{int(100 * time.time())}.json")
    print(f"Checking results path {results_path}")
    common.check_save_dir_exists(results_path)

    algs = [
        BenchmarkAlg.digraphwave(aggregate=False, transpose=False, normalize=False, num_gpus=0),
        BenchmarkAlg.digraphwave(aggregate=True, transpose=False, normalize=False, num_gpus=0),
        BenchmarkAlg.digraphwave(aggregate=True, transpose=True, normalize=False, num_gpus=0),
        BenchmarkAlg.digraphwave(aggregate=True, transpose=True, normalize=True, num_gpus=0),
        BenchmarkAlg.digraphwave(aggregate=False, transpose=False, normalize=False, num_gpus=1),
        BenchmarkAlg.digraphwave(aggregate=True, transpose=False, normalize=False, num_gpus=1),
        BenchmarkAlg.digraphwave(aggregate=True, transpose=True, normalize=False, num_gpus=1),
        BenchmarkAlg.digraphwave(aggregate=True, transpose=True, normalize=True, num_gpus=1),
        BenchmarkAlg.digraphwave(aggregate=False, transpose=False, normalize=False, num_gpus=2),
        BenchmarkAlg.digraphwave(aggregate=True, transpose=False, normalize=False, num_gpus=2),
        BenchmarkAlg.digraphwave(aggregate=True, transpose=True, normalize=False, num_gpus=2),
        BenchmarkAlg.digraphwave(aggregate=True, transpose=True, normalize=True, num_gpus=2),
    ]
    num_nodes = [100, 1000, 10000, 100000, 1000000]
    m_values = [1, 5, 10]
    num_reps = 4
    if debug:
        num_nodes = [100]  # for debug
        m_values = [2]
        num_reps = 1

    run_on_ba_graphs(num_nodes, m_values, num_reps=num_reps, algs=algs, timeout=timeout, results_path=results_path,
                     tmpfile_dir=workdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("main", type=str, default=None, help="Which main function to run.")
    parser.add_argument("-w", "--workdir", type=str, default="./", help="Path to file where results will be saved.")
    parser.add_argument("--debug", action="store_true", help="Use debug settings.")
    args = parser.parse_args()
    timeout = 3 if args.debug else 8 * 3600
    if args.main.lower() == "algs":
        main_compare_algorithms(timeout, workdir=args.workdir, debug=args.debug)
    elif args.main.lower() == "digw":
        main_compare_digraphwave_settings(timeout, workdir=args.workdir, debug=args.debug)
