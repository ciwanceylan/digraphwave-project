import random
import pytest
import numpy as np
import scipy.sparse as sp
import networkx as nx


def compose_graphs_disjoint(*graphs: nx.Graph) -> nx.Graph:
    num_nodes = 0
    reindexed_graphs = []
    for graph in graphs:
        _num_graph_nodes = graph.number_of_nodes()
        reindexed_graphs.append(nx.relabel_nodes(nx.convert_node_labels_to_integers(graph),
                                                 {i: i + num_nodes for i in range(_num_graph_nodes)}))
        num_nodes += _num_graph_nodes

    return nx.compose_all(reindexed_graphs)


def a_graph(num_nodes, create_using: type, weighted: bool = True) -> nx.Graph:
    clique = nx.complete_graph(num_nodes, create_using=create_using)
    cycle = nx.cycle_graph(num_nodes, create_using=create_using)
    tree = nx.random_tree(num_nodes)
    tree = nx.DiGraph([(u, v) for (u, v) in tree.edges() if u < v])
    singles = create_using()
    singles.add_node(0)
    singles.add_node(1)
    singles.add_edge(1, 1)
    graph = compose_graphs_disjoint(clique, cycle, tree, singles)
    if weighted:
        for (u, v, w) in graph.edges(data=True):
            w['weight'] = 10 * random.random()
    return graph


def graph2adj(graph: nx.Graph) -> sp.spmatrix:
    return nx.adjacency_matrix(graph, weight='weight').T


@pytest.fixture(scope="class")
def dir_graph() -> nx.Graph:
    return a_graph(100, nx.DiGraph, weighted=True)


@pytest.fixture(scope="class")
def undir_graph() -> nx.Graph:
    return a_graph(100, nx.Graph, weighted=True)


@pytest.fixture(scope="class")
def dir_adj_mat(dir_graph: nx.DiGraph) -> sp.spmatrix:
    """
    A directed adjacency  matrix consisting of four weakly connected components: a clique, a tree, a cycle and an
    isolated node
    :return:
    """
    return graph2adj(dir_graph)


@pytest.fixture(scope="class")
def undir_adj_mat(undir_graph: nx.Graph) -> sp.spmatrix:
    """
    A directed adjacency  matrix consisting of four weakly connected components: a clique, a tree, a cycle and an
    isolated node
    :return:
    """
    return graph2adj(undir_graph)
