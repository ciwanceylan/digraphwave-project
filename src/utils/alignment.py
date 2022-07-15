from typing import Mapping
import dataclasses as dc
import numpy as np
import pandas as pd

from numpy.random import default_rng

from sklearn.neighbors import KDTree

from src.utils.empirical_distribution import EDF


@dc.dataclass(frozen=True)
class SimpleGraph:
    num_nodes: int
    edges: np.ndarray
    weights: np.ndarray = None

    @classmethod
    def union(cls, g1: 'SimpleGraph', g2: 'SimpleGraph'):
        g2_to_merged = AlignedGraphs.g2_to_merged(g1.num_nodes, g2.num_nodes)
        # new_sources = g2_to_merged[g2.edges[:, 0]]
        # new_targets = g2_to_merged[g2.edges[:, 1]]
        new_edges = np.stack((g2_to_merged[g2.edges[:, 0]], g2_to_merged[g2.edges[:, 1]]), axis=1)
        edges = np.concatenate((g1.edges, new_edges), axis=0)
        weights = None
        if g1.weights is not None and g2.weights is not None:
            weights = np.concatenate((g1.weights, g2.weights), axis=0)
        return cls(g1.num_nodes + g2.num_nodes, edges, weights)

    @classmethod
    def add_noise_edges(cls, g: 'SimpleGraph', p: float):
        num_noise_edges = int(p * g.edges.shape[0])
        noise_edges = get_noise_edges(g.edges, num_noise_edges, g.num_nodes, max_attepts=5)
        num_noise_edges = noise_edges.shape[0]
        actual_p = num_noise_edges / g.edges.shape[0]
        edges = np.concatenate((g.edges, noise_edges), axis=0)
        weights = None
        if g.weights is not None:
            edf = EDF(g.weights, min_value=0)
            noise_weights = edf.sample(num_noise_edges)
            weights = np.concatenate((g.weights, noise_weights), axis=0)
        return cls(g.num_nodes, edges, weights), actual_p

    def save(self, fp):
        if self.weights is not None:
            data = pd.DataFrame({"source": self.edges[:, 0], "target": self.edges[:, 1], "weight": self.weights})
        else:
            data = pd.DataFrame({"source": self.edges[:, 0], "target": self.edges[:, 1]})
        if "b" in fp.mode:
            fp.write(f"%{self.num_nodes}\n".encode("utf-8"))
        else:
            fp.write(f"%{self.num_nodes}\n")
        data.to_csv(fp, sep="\t", index=False, header=False)


class AlignedGraphs:
    g1_num_nodes: int
    g2_num_nodes: int
    # g2_to_merged: pd.Series
    # merged_to_g2: pd.Series
    g1_to_g2: pd.Series
    g2_to_g1: pd.Series

    def __init__(self, g1_num_nodes, g2_num_nodes, g2_to_g1):
        assert g1_num_nodes >= g2_num_nodes
        assert len(g2_to_g1) == g2_num_nodes
        # assert len(perm_backward) == g2_num_nodes

        self.g1_num_nodes = g1_num_nodes
        self.g2_num_nodes = g2_num_nodes

        self.g2_to_g1 = g2_to_g1
        self.g1_to_g2 = pd.Series(g2_to_g1.index, index=g2_to_g1)
        # self.g2_to_merged = pd.Series(np.arange(g2_num_nodes, dtype=np.int64) + self.g1_num_nodes)
        # self.merged_to_g2 = pd.Series(range(g2_num_nodes), index=self.g2_to_merged)

    @staticmethod
    def g2_to_merged(g1_num_nodes: int, g2_num_nodes: int):
        return pd.Series(np.arange(g2_num_nodes, dtype=np.int64) + g1_num_nodes)

    @staticmethod
    def merged_to_g2(g1_num_nodes: int, g2_num_nodes: int):
        return pd.Series(np.arange(g2_num_nodes, dtype=np.int64),
                         index=np.arange(g2_num_nodes, dtype=np.int64) + g1_num_nodes)

    @classmethod
    def load_from_file(cls, path: str):
        with open(path, "r") as fp:
            first_line = fp.readline()
        g1_num_nodes, g2_num_nodes = first_line.strip("%\n ").split("::")
        g1_num_nodes = int(g1_num_nodes)
        g2_num_nodes = int(g2_num_nodes)
        g2_to_g1 = pd.read_csv(path, index_col=0, names=["g2_to_g1"], header=None, comment="%")["g2_to_g1"]
        return cls(g1_num_nodes, g2_num_nodes, g2_to_g1)

    def save2file(self, fp):
        fp.write("%" + str(self.g1_num_nodes) + "::" + str(self.g2_num_nodes) + "\n")
        pd.Series(self.g2_to_g1).to_csv(fp, index=True, header=False)

    # def merge_edges(self, g1_edges: np.ndarray, g2_edges: np.ndarray):
    #     assert np.max(g1_edges) < self.g1_num_nodes
    #     assert np.max(g2_edges) < self.g2_num_nodes
    #     new_sources = self.g2_to_merged[g2_edges[:, 0]]
    #     new_targets = self.g2_to_merged[g2_edges[:, 1]]
    #     g2_edges = np.stack((new_sources, new_targets), axis=1)
    #     return np.concatenate((g1_edges, g2_edges), axis=0)

    # def save_merged_edges(self, fp, g1g1_edges: np.ndarray, g2_edges: np.ndarray):
    #     edges = pd.DataFrame(self.merge_edges(g1_edges, g2_edges))
    #     fp.write("%" + str(self.g1_num_nodes + self.g2_num_nodes) + "\n")
    #     edges.to_csv(fp, sep="\t", index=False, header=False)


def get_noise_edges(current_edges: np.ndarray, num_add: int, num_nodes: int, max_attepts: int) -> np.ndarray:
    current_edges = current_edges[current_edges[:, 0] != current_edges[:, 1], :]  # Remove self-loops if exists
    rng = default_rng()

    forbidden_to_add = {(s, t) for s, t in np.sort(current_edges, axis=1)}
    edges_to_add = set()
    num_attempts = 0
    while len(edges_to_add) < num_add and num_attempts < max_attepts:
        new_edges = np.unique(
            np.sort(
                np.stack((rng.integers(low=0, high=num_nodes, size=num_add),
                          rng.integers(low=0, high=num_nodes, size=num_add)),
                         axis=1),
                axis=1),
            axis=0)
        new_edges = new_edges[new_edges[:, 0] != new_edges[:, 1], :]  # Remove self-loops if exists
        edges_to_add = edges_to_add.union({(s, t) for s, t in new_edges} - forbidden_to_add)
        num_attempts += 1
    index = np.random.permutation(len(edges_to_add))[:num_add]
    edges = np.asarray(list(edges_to_add))[index]
    return edges


def create_permuted(g: SimpleGraph):
    rng = default_rng()
    g2_to_g1 = pd.Series(rng.permutation(g.num_nodes))
    g1_to_g2 = pd.Series(g2_to_g1.index, index=g2_to_g1)
    new_sources = g1_to_g2[g.edges[:, 0]]
    new_targets = g1_to_g2[g.edges[:, 1]]
    g2_edges = np.stack((new_sources, new_targets), axis=1)
    new_g = SimpleGraph(num_nodes=g.num_nodes, edges=g2_edges, weights=g.weights)
    return new_g, AlignedGraphs(g.num_nodes, g.num_nodes, g2_to_g1)


def split_embeddings(embeddings: np.ndarray, g2_to_merged: pd.Series):
    g1_num_nodes = embeddings.shape[0] - len(g2_to_merged)
    g1_embeddings = embeddings[:g1_num_nodes, :]
    g2_merged = g2_to_merged[np.arange(len(g2_to_merged), dtype=np.int64)]
    g2_embeddings = embeddings[g2_merged, :]
    return g1_embeddings, g2_embeddings


def calc_topk_similarties(g1_embeddings, g2_embeddings, alpha=50):
    kd_tree = KDTree(g1_embeddings, metric="euclidean")

    dist, ind = kd_tree.query(g2_embeddings, k=alpha)
    similarity = np.exp(-dist)
    return similarity, ind


def get_top_sim(embeddings: np.ndarray, g2_to_merged: pd.Series, alpha=50):
    num_g2_nodes = len(g2_to_merged)
    g1_embeddings, g2_embeddings = split_embeddings(embeddings, g2_to_merged)
    similarity, ind = calc_topk_similarties(g1_embeddings, g2_embeddings, alpha=alpha)
    assert similarity.shape[0] == num_g2_nodes and ind.shape[0] == num_g2_nodes

    res = dict()
    for i in range(num_g2_nodes):
        res[i] = [(s, node) for s, node in zip(similarity[i, :], ind[i, :])]

    return res


def calc_topk_acc_score(y_true: Mapping, top_sim: dict):
    topk_vals = np.asarray([1, 5, 10, 25, 50])
    scores = np.zeros(len(topk_vals), dtype=np.float64)

    for g2_node, y in y_true.items():
        vals = sorted(top_sim[g2_node], key=lambda x: -x[0])
        the_k = np.inf
        for k_, (sim, val) in enumerate(vals):
            if y == val:
                the_k = k_
        scores[the_k < topk_vals] += 1.
    scores = scores / len(y_true)
    return dict(zip(topk_vals, scores))


def eval_topk_sim(embeddings: np.ndarray, alignment: AlignedGraphs):
    top_sim = get_top_sim(embeddings, alignment.g2_to_merged(alignment.g1_num_nodes, alignment.g2_num_nodes))
    res = calc_topk_acc_score(alignment.g2_to_g1, top_sim)
    return res
