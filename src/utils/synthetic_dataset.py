from typing import Tuple, List
import os
import random
import dataclasses as dc
import pandas as pd
import graph_tool as gt
import graph_tool.generation as gtgen
import numpy as np
from tqdm.auto import trange

INCLUDED_COMPONENTS = {"B5", "C8", "D5", "DB55", "DS5", "H5", "L5", "PB5", "S5", "U5", "W5"}
ALL_UNIQUE_LABELS = []
for label in INCLUDED_COMPONENTS:
    ALL_UNIQUE_LABELS.append(f"{label}_0")
    ALL_UNIQUE_LABELS.append(f"{label}_1")


@dc.dataclass(frozen=True)
class SyntheticGraph:
    name: str
    core_name: str
    edges: pd.DataFrame
    labels: pd.DataFrame
    gt_graph: gt.Graph

    @classmethod
    def load(cls, root: str, name: str, reverse: bool):
        core_name = name
        name = f"{name}_{int(reverse)}"
        edge_file = os.path.join(root, core_name, f"{name}.edgelist")
        node_labels_file = os.path.join(root, core_name, f"{name}_nodes.txt")
        edges = pd.read_csv(edge_file, sep="\s+", index_col=False, names=["source", "target"])
        labels = pd.read_csv(node_labels_file, sep="\s+", index_col=False, names=["node_id", "label"])
        graph = gt.Graph(directed=True)
        graph.add_edge_list(edges.to_numpy())
        label_vp = graph.new_vp('int')
        label_vp.a = labels.loc[:, "label"].to_numpy()
        str_label_vp = graph.new_vp('string')
        for node in graph.vertices():
            str_label_vp[node] = f"{name}_{label_vp[node]}"
        graph.vp.label = label_vp
        graph.vp.str_label = str_label_vp
        return cls(name, core_name, edges, labels, graph)


def create_composed(root: str, num_repeated: int, num_connecting_edges_per_subgraph: int, included_components=None):
    # included_components = {"B5", "C8", "D5", "DB55", "DS5", "H5", "L5", "PB5", "S5", "U5", "W5"}
    # included_components = {"B5"}  # For debug
    if included_components is None:
        included_components = INCLUDED_COMPONENTS

    all_graphs = []

    for i in range(num_repeated):
        for comp_name in included_components:
            for reversed in [True, False]:
                graph = SyntheticGraph.load(root, comp_name, reversed)
                all_graphs.append(graph.gt_graph)

    num_main_nodes = max(5 * len(all_graphs), 10)
    main_graph = gtgen.circular_graph(num_main_nodes, directed=True)
    label_vp = main_graph.new_vp('int', val=-1)
    str_label_vp = main_graph.new_vp('string', val="main")
    subgraph_vp = main_graph.new_vp('int', val=-1)
    main_graph.vp.label = label_vp
    main_graph.vp.str_label = str_label_vp

    random.shuffle(all_graphs)
    num_nodes = main_graph.num_vertices()
    for i, g in enumerate(all_graphs):
        main_graph = gtgen.graph_union(main_graph, g, internal_props=True, include=True)
        subgraph_vp.a[num_nodes:num_nodes + g.num_vertices()] = i
        num_nodes += g.num_vertices()

    main_graph.vp.subgraph_label = subgraph_vp
    _ = add_connecting_edge(main_graph, num_connecting_edges_per_subgraph)
    return main_graph


def add_connecting_edge(graph, edges_per_subgraph):
    if edges_per_subgraph < 1:
        return np.asarray([[]])
    num_subgraphs = graph.vp.subgraph_label.a.max() + 1
    edges_to_add = []
    for s in range(num_subgraphs):
        nodes = np.nonzero(graph.vp.subgraph_label.a == s)[0]
        sources = np.random.choice(nodes, size=edges_per_subgraph)
        targets = np.random.randint(low=s * 5, high=(s + 1) * 5 - 1, size=edges_per_subgraph)
        edges_to_add.append(np.stack((sources, targets), axis=1))

    edges_to_add = np.unique(np.concatenate(edges_to_add, axis=0), axis=1)
    edges_to_add = np.concatenate((edges_to_add, edges_to_add[:, ::-1]), axis=0)

    graph.add_edge_list(edges_to_add)
    return edges_to_add


def get_train_test_labels(graph: gt.Graph, num_train: int):
    labels = pd.Series((graph.vp.str_label[i] for i in range(graph.num_vertices())))
    subgraph_df = labels.str.rsplit("_", n=1, expand=True).rename(columns={0: "sg_type"}).loc[:, ["sg_type"]]

    subgraph_df["sg_index"] = graph.vp.subgraph_label.a
    subgraph_df["node_index"] = subgraph_df.index
    main_graph_index = subgraph_df.loc[(subgraph_df["sg_index"] == -1)].index
    subgraph_df = subgraph_df.loc[subgraph_df["sg_index"] > -1, :]

    grouped = subgraph_df.groupby(["sg_type", "sg_index"])["node_index"].apply(list)

    train_node_index = []
    test_node_index = []
    for ind in grouped.index.get_level_values(0).unique():
        train_node_index.extend(grouped[ind].iloc[:num_train].sum())
        test_node_index.extend(grouped[ind].iloc[num_train:].sum())

    train_labels = labels[train_node_index]
    test_labels = labels[test_node_index]
    other_labels = labels[main_graph_index]

    # Sainity checks
    assert (len(train_node_index) + len(test_node_index) + len(main_graph_index)) == graph.num_vertices()
    assert len(np.unique(np.concatenate((train_node_index, test_node_index, main_graph_index)))) == graph.num_vertices()

    return train_labels, test_labels, other_labels


def save_datasets(save_root_dir="data/synthetic/composed"):
    num_reps = 10
    num_test = 5
    num_graph_samples = 5
    num_train = num_reps - num_test

    for num_connects in trange(6):
        for gs in trange(num_graph_samples):
            graph = create_composed("data/synthetic/", num_reps, num_connects)
            train_labels, test_labels, other_labels = get_train_test_labels(graph, num_train)

            save_dir = os.path.join(save_root_dir, f"cycle_main_{num_reps}_{num_connects}-{gs}")
            os.makedirs(save_dir, exist_ok=True)

            edges = pd.DataFrame(({'source': int(s), 'target': int(t)} for s, t in graph.edges()))
            pd.DataFrame(edges).to_csv(os.path.join(save_dir, f"composed_graph.edgelist"), sep=" ",
                                       index=False, header=False)

            train_labels.to_json(os.path.join(save_dir, "train_labels.json"), indent=2)
            test_labels.to_json(os.path.join(save_dir, "test_labels.json"), indent=2)
            other_labels.to_json(os.path.join(save_dir, "other_labels.json"), indent=2)


def get_unique_labels(data_dir=None):
    if data_dir is None:
        data_dir = f"data/synthetic/composed/cycle_main_10_0/"
    train_labels = pd.read_json(os.path.join(data_dir, "train_labels.json"), typ="series")
    test_labels = pd.read_json(os.path.join(data_dir, "test_labels.json"), typ="series")
    combinded_labels = pd.concat((train_labels, test_labels), axis=0)
    unique_labels = combinded_labels.unique()
    return unique_labels


if __name__ == "__main__":
    save_datasets()
