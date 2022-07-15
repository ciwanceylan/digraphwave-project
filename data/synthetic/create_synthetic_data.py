#!/usr/bin/env python
# coding: utf-8


import graph_tool.all as gt
import pandas as pd


def L5(reverse):
    g = gt.complete_graph(5, directed=True)
    g.set_directed(True)
    for i in range(5):
        g.add_edge(4 + i, 4 + i + 1)

    if reverse:
        g.set_reversed(True)

    edges = [(int(s), int(t)) for s, t in g.edges()]
    pd.DataFrame(edges).to_csv(f"L5/L5_{int(reverse)}.edgelist", sep=" ", index=False, header=False)
    labels = [
        0, 0, 0, 0, 1, 2, 3, 4, 5, 6
    ]
    pd.DataFrame(labels).to_csv(f"L5/L5_{int(reverse)}_nodes.txt", sep=" ", index=True, header=False)
    return g


def C8(reverse):
    g = gt.circular_graph(8, directed=False)
    g.set_directed(True)
    if reverse:
        g.set_reversed(True)
    g.add_edge(0, 5)
    g.add_edge(6, 3)
    edges = [(int(s), int(t)) for s, t in g.edges()]
    pd.DataFrame(edges).to_csv(f"C8/C8_{int(reverse)}.edgelist", sep=" ", index=False, header=False)
    labels = list(range(g.num_vertices()))
    pd.DataFrame(labels).to_csv(f"C8/C8_{int(reverse)}_nodes.txt", sep=" ", index=True, header=False)
    return g


def B5(reverse):
    g1 = gt.complete_graph(5, directed=True)
    g1.set_directed(True)
    n = g1.num_vertices()
    for i in range(4):
        g1.add_edge(4 - 1 + i, 4 + i)
    for i in range(2):
        g1.add_edge(8 + i, 8 + i - 1)

    g2 = gt.complete_graph(5, directed=True)
    g = gt.graph_union(g1, g2)

    g.add_edge(10, 9)
    if reverse:
        g.set_reversed(True)

    edges = [(int(s), int(t)) for s, t in g.edges()]
    labels = [
        0, 0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 0, 0
    ]
    assert len(labels) == g.num_vertices()
    pd.DataFrame(edges).to_csv(f"B5/B5_{int(reverse)}.edgelist", sep=" ", index=False, header=False)
    labels = [
        0, 0, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 0, 0
    ]

    pd.DataFrame(labels).to_csv(f"B5/B5_{int(reverse)}_nodes.txt", sep=" ", index=True, header=False)
    return g


def DB55(reverse):
    g1 = gt.complete_graph(5, directed=True)
    g2 = gt.complete_graph(5, directed=True)
    g = gt.graph_union(g1, g2)
    g.add_edge(4, 5)
    if reverse:
        g.add_edge(5, 4)

    edges = [(int(s), int(t)) for s, t in g.edges()]
    if reverse:
        labels = [0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    else:
        labels = [0, 0, 0, 0, 1, 2, 3, 3, 3, 3]
    assert len(labels) == g.num_vertices()
    pd.DataFrame(edges).to_csv(f"DB55/DB55_{int(reverse)}.edgelist", sep=" ", index=False, header=False)

    pd.DataFrame(labels).to_csv(f"DB55/DB55_{int(reverse)}_nodes.txt", sep=" ", index=True, header=False)
    return g


def S5(reverse):
    g = gt.Graph(directed=True)
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(0, 3)
    g.add_edge(0, 4)
    g.add_edge(0, 5)
    if reverse:
        g.set_reversed(True)

    labels = [0, 1, 1, 1, 1, 1]
    assert len(labels) == g.num_vertices()
    edges = [(int(s), int(t)) for s, t in g.edges()]
    pd.DataFrame(edges).to_csv(f"S5/S5_{int(reverse)}.edgelist", sep=" ", index=False, header=False)

    pd.DataFrame(labels).to_csv(f"S5/S5_{int(reverse)}_nodes.txt", sep=" ", index=True, header=False)
    return g


def DS5(reverse):
    g = gt.Graph(directed=True)
    for i in range(1, 6):
        g.add_edge(0, i)
        g.add_edge(i, 0)
    g.add_edge(0, 6)
    if reverse:
        g.add_edge(6, 0)
    for i in range(7, 7 + 5):
        g.add_edge(6, i)
        g.add_edge(i, 6)

    if reverse:
        labels = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    else:
        labels = [1, 0, 0, 0, 0, 0, 2, 3, 3, 3, 3, 3]
    assert len(labels) == g.num_vertices()
    edges = [(int(s), int(t)) for s, t in g.edges()]
    pd.DataFrame(edges).to_csv(f"DS5/DS5_{int(reverse)}.edgelist", sep=" ", index=False, header=False)

    pd.DataFrame(labels).to_csv(f"DS5/DS5_{int(reverse)}_nodes.txt", sep=" ", index=True, header=False)
    return g


def D5(reverse):
    g = gt.Graph(directed=True)
    for i in range(1, 6):
        g.add_edge(0, i)
        g.add_edge(i, i + 5)
        g.add_edge(i + 5, 11)

    if reverse:
        for i in range(1, 6):
            g.add_edge(i, 0)
            g.add_edge(i + 5, i)
            g.add_edge(11, i + 5)

    if reverse:
        labels = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    else:
        labels = [0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3]
    assert len(labels) == g.num_vertices()
    edges = [(int(s), int(t)) for s, t in g.edges()]
    pd.DataFrame(edges).to_csv(f"D5/D5_{int(reverse)}.edgelist", sep=" ", index=False, header=False)

    pd.DataFrame(labels).to_csv(f"D5/D5_{int(reverse)}_nodes.txt", sep=" ", index=True, header=False)
    return g


def PB5(reverse):
    g1 = gt.circular_graph(5, directed=True)
    g2 = gt.circular_graph(5, directed=True)
    g = gt.graph_union(g1, g2)
    g.set_directed(True)
    g.add_edge(4, 5)
    g.add_edge(5, 4)
    if reverse:
        g.add_edge(0, 2)
        g.add_edge(3, 1)
        g.add_edge(9, 7)
        g.add_edge(6, 8)
    else:
        g.add_edge(2, 0)
        g.add_edge(1, 3)
        g.add_edge(7, 9)
        g.add_edge(8, 6)

    labels = [0, 1, 1, 0, 2, 2, 0, 1, 1, 0]
    assert len(labels) == g.num_vertices()
    edges = [(int(s), int(t)) for s, t in g.edges()]
    pd.DataFrame(edges).to_csv(f"PB5/PB5_{int(reverse)}.edgelist", sep=" ", index=False, header=False)

    pd.DataFrame(labels).to_csv(f"PB5/PB5_{int(reverse)}_nodes.txt", sep=" ", index=True, header=False)
    return g


def W5(reverse):
    g1 = gt.circular_graph(5, directed=True)
    g2 = gt.circular_graph(5, directed=True)
    for i in range(5):
        g1.add_edge(5, i)
        g2.add_edge(5, i)

    if reverse:
        g2.set_reversed(True)

    g = gt.graph_union(g1, g2)
    g.add_edge(0, 7)
    if reverse:
        g.add_edge(2, 10)
    else:
        g.add_edge(10, 2)

    if reverse:
        labels = [0, 1, 0, 2, 2, 3, 4, 5, 6, 6, 5, 7]
    else:
        labels = [0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 0, 5]

    assert len(labels) == g.num_vertices()
    edges = [(int(s), int(t)) for s, t in g.edges()]
    pd.DataFrame(edges).to_csv(f"W5/W5_{int(reverse)}.edgelist", sep=" ", index=False, header=False)

    pd.DataFrame(labels).to_csv(f"W5/W5_{int(reverse)}_nodes.txt", sep=" ", index=True, header=False)
    return g


def U5(reverse):
    if reverse:
        edges = [
            [1, 0],
            [1, 2],
            [1, 3],
            [1, 4],
            [1, 5],
            [1, 6],
            [2, 1],
            [2, 3],
            [3, 1],
            [3, 4],
            [4, 1],
            [4, 5],
            [5, 1],
            [5, 6],
            [6, 1],
        ]

    else:
        edges = [
            (0, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (2, 1),
            (2, 3),
            (3, 1),
            (3, 4),
            (4, 1),
            (4, 5),
            (5, 1),
            (5, 6),
            (6, 1),
        ]
    labels = list(range(7))
    pd.DataFrame(edges).to_csv(f"U5/U5_{int(reverse)}.edgelist", sep=" ", index=False, header=False)

    pd.DataFrame(labels).to_csv(f"U5/U5_{int(reverse)}_nodes.txt", sep=" ", index=True, header=False)
    return None


def H5(reverse):
    if reverse:
        edges = [
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (1, 4),
            (2, 1),
            (2, 3),
            (2, 4),
            (3, 1),
            (3, 4),
            (4, 2),
            (4, 3),
        ]

    else:
        edges = [
            (1, 0),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
            (2, 4),
            (3, 1),
            (3, 2),
            (3, 4),
            (4, 1),
            (4, 2),
            (4, 3),

        ]

    labels = [0, 1, 1, 2, 2]
    pd.DataFrame(edges).to_csv(f"H5/H5_{int(reverse)}.edgelist", sep=" ", index=False, header=False)

    pd.DataFrame(labels).to_csv(f"H5/H5_{int(reverse)}_nodes.txt", sep=" ", index=True, header=False)
    return None


if __name__ == "__main__":
    B5(False)
    B5(True)

    C8(False)
    C8(True)

    D5(False)
    D5(True)

    DB55(False)
    DB55(True)

    DS5(False)
    DS5(True)

    H5(False)
    H5(True)

    L5(False)
    L5(True)

    PB5(False)
    PB5(True)

    S5(False)
    S5(True)

    U5(False)
    U5(True)

    W5(False)
    W5(True)
