# Digraphwave Experiments

A repo containing the experiments for the paper.

## Setup


### Prepare environments
Two different experiment environments are used. A python2 environment for Graphwave and EMBER, and a python3 environment for the other methods.
There is also an extra environment used for some visualisations (`digraphwave_python3_with_full_gt_env`).

Create all the environments so that they are available.
```bash
conda env create --file conda_env_files/digraphwave_experiment_python2_env.yml
conda env create --file conda_env_files/digraphwave_experiment_python3_env.yml
codda env create --file conda_env_files/digraphwave_python3_with_full_gt_env.yml
```

### Install Digraphwave and ReFeX in the python3 environment

```bash
conda activate digraphwave_experiment_python3_env
pip install ./algorithms/rolx-refex
pip install ./algorithms/digraphwave
```

### Experiments

All experiments restuls are available in `./results`. To rerun the experiments, first create a new results folder.
```bash
mkdir ./new_results
```
Then use the commands below to rerun all the experiments. 
Note that default parameter may expect certain hardware to be available, e.g.,
3 GPUs for the Enron experiments.

#### Scalability experiment

```bash
conda activate digraphwave_experiment_python3_env
python python scalability_benchmarks.py algs -w ./new_results
```

#### Ablation

```bash
conda activate digraphwave_experiment_python3_env
python synthetic_data_benchmarks.py ablation -w ./new_results
```


#### Quality comparison on synthetic data

```bash
conda activate digraphwave_experiment_python3_env
python synthetic_data_benchmarks.py algs -w ./new_results
```

#### Enron node classification
```bash
conda activate digraphwave_experiment_python3_env
python enron_node_classification.py 128 --weighted --simplified -w ./new_results
python enron_node_classification.py 128 --simplified -w ./new_results
python enron_node_classification.py 128 --weighted --undirected --simplified -w ./new_results
python enron_node_classification.py 128 --undirected --simplified -w ./new_results

```

#### Network alignment
python alignment_experiments.py $experiment $weighted $undirected -w $RESULTS_ROOT

```bash
conda activate digraphwave_experiment_python3_env
python alignment_experiments.py enron_all_internal --weighted -w ./new_results
python alignment_experiments.py enron_all_internal -w ./new_results
python alignment_experiments.py enron_all_internal --weighted --undirected -w ./new_results
python alignment_experiments.py enron_all_internal --undirected -w ./new_results
python alignment_experiments.py arxiv --undirected -w ./new_results

```

### Datasets

#### Synthetic dataset
The synthetic dataset is found in `./data/synthetic`.
The folders contain the edge list and automorphic identity labels for each graph.
Moreover, `./data/synthetic/composed` contains the composed graphs (circular and attached synthetic graphs) used for the experiments.

The file `./data/synthetic/create_synthetic_data.py` can be used to create the synthetic dataset from scratch:
```bash
conda activate digraphwave_python3_with_full_gt_env
cd ./some/workdir
python ./data/synthetic/create_synthetic_data.py
```
And the file The file `./src/utils/synthetic_dataset.py` can be used generate composed graphs.
