import numpy as np
import scipy.sparse as sp


def edges2adj(sources, targets, directed: bool = True, use_csc: bool = True):
    n = max(sources.max(), targets.max()) + 1
    adj = sp.coo_matrix((np.ones(len(sources), dtype=np.float64), (targets, sources)), shape=[n, n])
    adj = adj.tocsc() if use_csc else adj.tocsr()
    if not directed:
        adj = adj.maximum(adj.T)
    return adj


def _diamond_edges(n: int):
    sources = np.concatenate((
        np.zeros(n - 2, dtype=np.int64),
        np.arange(1, n - 1, dtype=np.int64)
    ))
    targets = np.concatenate((
        np.arange(1, n - 1, dtype=np.int64),
        np.full(shape=n - 2, fill_value=n - 1, dtype=np.int64),
    ))
    return sources, targets


def diamond_adj(n: int, directed: bool = True, use_csc: bool = True):
    sources, targets = _diamond_edges(n)
    return edges2adj(sources, targets, directed=directed, use_csc=use_csc)


def _cycle_edges(n: int):
    sources = np.arange(0, n, dtype=np.int64)
    targets = np.arange(1, n + 1, dtype=np.int64) % n
    return sources, targets


def cycle_adj(n: int, directed: bool = True, use_csc: bool = True):
    sources, targets = _cycle_edges(n)
    return edges2adj(sources, targets, directed=directed, use_csc=use_csc)


def _cycle_with_random_chords_edges(n: int, num_chords: int):
    sources, targets = _cycle_edges(n)
    chord_sources = np.random.randint(low=0, high=n, size=3 * num_chords, dtype=np.int64)
    chord_targets = np.random.randint(low=0, high=n, size=3 * num_chords, dtype=np.int64)
    chords = np.stack((chord_sources, chord_targets), axis=1)
    chords.sort(axis=1)
    chords = np.unique(chords, axis=0)
    chords = chords[chords[:, 0] != chords[:, 1]]
    sources = np.concatenate((sources, chords[:num_chords, 0]))
    targets = np.concatenate((targets, chords[:num_chords, 1]))
    return sources, targets


def _cycle_with_diameter_chords_edges(n: int, num_chords: int):
    sources, targets = _cycle_edges(n)
    chord_sources = np.random.randint(low=0, high=n, size=3 * num_chords, dtype=np.int64)
    chord_sources = np.unique(chord_sources)
    chord_targets = (chord_sources + (n // 2)) % n
    sources = np.concatenate((sources, chord_sources[:num_chords]))
    targets = np.concatenate((targets, chord_targets[:num_chords]))
    return sources, targets


def cycle_with_chords_adj(n: int, num_chords: int, diameter_chords: bool = True,
                          directed: bool = True, use_csc: bool = True):
    if diameter_chords:
        sources, targets = _cycle_with_diameter_chords_edges(n, num_chords)
    else:
        sources, targets = _cycle_with_random_chords_edges(n, num_chords)
    return edges2adj(sources, targets, directed=directed, use_csc=use_csc)


def _ss_num_nodes(d, beta, ell):
    return 1 + d * np.sum(np.power(beta, np.arange(0, ell, 1)))


def _ss_vl2i(v, l, beta, d):
    if l == 0:
        return 0
    if v > d * beta ** (l - 1):
        raise ValueError
    powers = np.power(beta, np.arange(0, l - 1, 1))
    return int(v + d * np.sum(powers))


def _ss_i2l(i, beta, d):
    beta = int(beta)
    if beta < 1:
        raise ValueError
    elif beta == 1:
        return int(np.ceil(i / d))
    if beta > 1:
        x = 1 + ((beta - 1) * i) / d
        xx = (np.log(x) / np.log(beta))
        return int(np.ceil(xx))


def _ss_il2v(i, l, beta, d):
    if i == 0:
        return 0
    beta = int(beta)
    if beta < 1:
        raise ValueError
    elif beta == 1:
        return int(i - d * (l - 1))
    else:
        return int(i - (d * (beta ** (l - 1) - 1) / (beta - 1)))


def _source_star_edges(source_degree: int, beta: int, ell: int):
    assert source_degree > 0
    assert beta > 0
    assert ell > 0
    sources = []
    targets = []
    d = source_degree
    for l in range(1, ell + 1):
        num_nodes_in_prev_layer = int(d * (beta ** (l - 2))) if l > 1 else 1
        num_nodes_in_layer = int(d * (beta ** (l - 1)))
        ratio = num_nodes_in_layer // num_nodes_in_prev_layer

        nodes_in_prev_layer = np.fromiter(
            (_ss_vl2i(v, l - 1, beta, d) for v in range(1, num_nodes_in_prev_layer + 1)),
            dtype=np.int64
        )
        s = np.repeat(nodes_in_prev_layer, ratio)
        t = np.fromiter(
            (_ss_vl2i(v, l, beta, d) for v in range(1, num_nodes_in_layer + 1)),
            dtype=np.int64
        )

        sources.append(s)
        targets.append(t)
    sources = np.concatenate(sources)
    targets = np.concatenate(targets)
    return sources, targets


def source_star_adj(source_degree: int, beta: int, ell: int, directed: bool = True, use_csc: bool = True):
    sources, targets = _source_star_edges(source_degree, beta, ell)
    return edges2adj(sources, targets, directed=directed, use_csc=use_csc)
