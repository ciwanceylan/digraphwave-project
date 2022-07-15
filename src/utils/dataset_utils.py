from typing import Sequence, Dict
import requests
import tarfile
import os
import dataclasses as dc
import logging
import time

import graph_tool.all as gt
import pandas as pd
import numpy as np
import scipy.sparse as sp


@dc.dataclass(frozen=True)
class DatasetMetadata:
    name: str
    url: str
    data_subpath: str
    compression: str
    filetype: str


DATASETS_METADATA: Dict[str, DatasetMetadata] = {
    'enron': DatasetMetadata('enron',
                             "http://konect.cc/files/download.tsv.enron.tar.bz2",
                             os.path.join('download', 'enron', 'out.enron'),
                             'bz2',
                             'konect-tsv'),
    'hep-ph': DatasetMetadata('hep-ph',
                              "http://konect.cc/files/download.tsv.cit-HepPh.tar.bz2",
                              os.path.join('download', 'cit-HepPh', 'out.cit-HepPh'),
                              'bz2',
                              'konect-tsv'),
    'hep-th': DatasetMetadata('hep-th',
                              "http://konect.cc/files/download.tsv.cit-HepTh.tar.bz2",
                              os.path.join('download', 'cit-HepTh', 'out.cit-HepTh'),
                              'bz2',
                              'konect-tsv'),
    'cora': DatasetMetadata('cora',
                            "http://konect.cc/files/download.tsv.subelj_cora.tar.bz2",
                            os.path.join('download', 'subelj_cora', 'out.subelj_cora_cora'),
                            'bz2',
                            'konect-tsv'),
    # 'cora-ml': DatasetMetadata('cora-ml',
    #                            "https://github.com/danielzuegner/netgan/raw/master/data/cora_ml.npz",
    #                            '',
    #                            '',
    #                            'netgan-npz-adj'
    #                            ),
    'polblogs': DatasetMetadata('polblogs',
                                "http://konect.cc/files/download.tsv.dimacs10-polblogs.tar.bz2",
                                os.path.join('download', 'dimacs10-polblogs', 'out.dimacs10-polblogs'),
                                'bz2',
                                'konect-tsv'
                                ),
    # 'citeseer': DatasetMetadata('citeseer',
    #                             "https://github.com/abojchevski/graph2gauss/raw/master/data/citeseer.npz",
    #                             '',
    #                             '',
    #                             'g2g-npz-adj'
    #                             ),
    # 'dblp': DatasetMetadata('dblp',
    #                         "https://github.com/abojchevski/graph2gauss/raw/master/data/dblp.npz",
    #                         '',
    #                         '',
    #                         'g2g-npz-adj'
    #                         ),
    # 'pubmed': DatasetMetadata('pubmed',
    #                           "https://github.com/abojchevski/graph2gauss/raw/master/data/pubmed.npz",
    #                           '',
    #                           '',
    #                           'g2g-npz-adj'
    #                           ),
}


# def netgan_load_npz(file_name):
#     """Load a SparseGraph from a Numpy binary file.
#     Taken from NetGAN github https://github.com/danielzuegner/netgan/blob/master/netgan/utils.py
#     Parameters
#     ----------
#     file_name : str
#         Name of the file to load.
#     Returns
#     -------
#     sparse_graph : SparseGraph
#         Graph in sparse matrix format.
#     """
#     if not file_name.endswith('.npz'):
#         file_name += '.npz'
#     with np.load(file_name, allow_pickle=True) as loader:
#         loader = dict(loader)['arr_0'].item()
#         adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
#                                     loader['adj_indptr']), shape=loader['adj_shape'])
#
#         if 'attr_data' in loader:
#             attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
#                                          loader['attr_indptr']), shape=loader['attr_shape'])
#         else:
#             attr_matrix = None
#
#         labels = loader.get('labels')
#
#     return adj_matrix, attr_matrix, labels


# def graph2gauss_load_npz(file_name):
#     """Load a SparseGraph from a Numpy binary file.
#     Taken from NetGAN github https://github.com/danielzuegner/netgan/blob/master/netgan/utils.py
#     Parameters
#     ----------
#     file_name : str
#         Name of the file to load.
#     Returns
#     -------
#     sparse_graph : SparseGraph
#         Graph in sparse matrix format.
#     """
#     if not file_name.endswith('.npz'):
#         file_name += '.npz'
#     with np.load(file_name, allow_pickle=False) as loader:
#         adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
#                                     loader['adj_indptr']), shape=loader['adj_shape'])
#
#         if 'attr_data' in loader:
#             attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
#                                          loader['attr_indptr']), shape=loader['attr_shape'])
#         else:
#             attr_matrix = None
#
#         labels = loader.get('labels')
#
#     return adj_matrix, attr_matrix, labels


def get_save_target(save_dir, data_metadata: DatasetMetadata):
    filename = data_metadata.url.split("/")[-1]
    filepath = os.path.join(save_dir, filename)
    if data_metadata is not None:
        extraction_dir_target = os.path.join(save_dir, data_metadata.name)
    else:
        extraction_dir_target = os.path.join(
            save_dir, os.path.split(filename)[-1].split(".tar")[0]
        )

    if not data_metadata.compression:
        data_path = filepath
    else:
        data_path = os.path.join(extraction_dir_target, data_metadata.data_subpath)
    return filepath, extraction_dir_target, data_path


def extract_bz2(filename, path="."):
    extraction_dir_target = os.path.join(
        path, os.path.split(filename)[-1].split(".")[0]
    )
    with tarfile.open(filename, "r:bz2") as tar:
        tar.extractall(extraction_dir_target)


def extract_gz(filename, path="."):
    extraction_dir_target = os.path.join(
        path, os.path.split(filename)[-1].split(".")[0]
    )
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(extraction_dir_target)


def download_from_url(url: str, save_path: str):
    r = requests.get(url, allow_redirects=True)
    with open(save_path, "wb") as fp:
        fp.write(r.content)


def download_and_extract(save_dir: str, graph_metadata: DatasetMetadata):
    tar_save_path, extraction_dir_target, data_path = get_save_target(
        save_dir, graph_metadata
    )
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(tar_save_path):
        logging.info("Downloading data...")
        download_from_url(graph_metadata.url, tar_save_path)
        logging.info("download finished!")

    if not (
            os.path.exists(extraction_dir_target)
            and len(os.listdir(extraction_dir_target)) > 0
    ):
        if graph_metadata.compression:
            if graph_metadata.compression == "bz2":
                extract_bz2(tar_save_path, path=extraction_dir_target)
            elif graph_metadata.compression == "gz":
                extract_gz(tar_save_path, path=extraction_dir_target)
            else:
                raise ValueError(f"Can not untar. Unknown file ending of {tar_save_path}")

    return data_path


def read_to_edges(path: str, filetype: str):
    if filetype == 'konect-tsv' or filetype == 'konect-csv':
        sep = ',' if filetype == 'csv' else '\s+'
        edges_df = pd.read_csv(path, sep=sep, comment='%', index_col=False, header=None).loc[:, [0, 1]]
        edges_df = edges_df.groupby([0, 1], as_index=False).agg('sum')

        edges = edges_df.loc[:, [0, 1]].to_numpy() - 1  # Connect index from 1
    # elif filetype == 'netgan-npz-adj':
    #     adj, _, _ = netgan_load_npz(path)
    #     adj = adj.tocoo()
    #     edges = np.stack((adj.row, adj.col), axis=1)
    # elif filetype == 'g2g-npz-adj':
    #     adj, _, _ = graph2gauss_load_npz(path)
    #     adj = adj.tocoo()
    #     edges = np.stack((adj.row, adj.col), axis=1)
    else:
        raise NotImplementedError(f"Reading edges not implemented for filetype {filetype}.")

    return edges


def read_to_edges_and_weights(path: str, filetype: str):
    if filetype == 'konect-tsv' or filetype == 'konect-csv':
        sep = ',' if filetype == 'csv' else '\s+'
        edges_df = pd.read_csv(path, sep=sep, comment='%', index_col=False, header=None).loc[:, [0, 1, 2]]
        edges_df = edges_df.groupby([0, 1], as_index=False).agg('sum')

        edges = edges_df.loc[:, [0, 1]].to_numpy() - 1  # Connect index from 1
        weights = edges_df.loc[:, 2].to_numpy()
    # elif filetype == 'netgan-npz-adj':
    #     adj, _, _ = netgan_load_npz(path)
    #     adj = adj.tocoo()
    #     edges = np.stack((adj.row, adj.col), axis=1)
    # elif filetype == 'g2g-npz-adj':
    #     adj, _, _ = graph2gauss_load_npz(path)
    #     adj = adj.tocoo()
    #     edges = np.stack((adj.row, adj.col), axis=1)
    else:
        raise NotImplementedError(f"Reading edges not implemented for filetype {filetype}.")

    return edges, weights


def edges_to_gt(edges: np.ndarray, weights: np.ndarray, remove_isolated_nodes: bool = False, extract_lcc: bool = False):
    # gt.load_graph_from_csv(path, directed=False, )    # gt.load_graph_from_csv(path, directed=False, )
    graph = gt.Graph(directed=True)

    eweights = graph.new_ep('double')
    graph.add_edge_list(np.column_stack((edges, weights)), eprops=[eweights])
    if extract_lcc:
        graph = gt.extract_largest_component(graph, directed=False, prune=True)
    elif remove_isolated_nodes:
        not_isolated_nodes = graph.new_vp('bool')
        not_isolated_nodes.a = graph.degree_property_map('total').a > 0
        graph = gt.Graph(gt.GraphView(graph, vfilt=not_isolated_nodes), prune=True)
    return graph


def load_gt_graph(graph_name: str, use_weights: bool, data_root: str = "./data/real_data",
                  remove_isolated_nodes: bool = False, extract_lcc: bool = False):
    if graph_name not in DATASETS_METADATA:
        raise ValueError(f"Not metadata available for {graph_name}.")

    max_download_extract_attempts = 10
    for attempt in range(max_download_extract_attempts):
        try:
            data_path = download_and_extract(save_dir=data_root, graph_metadata=DATASETS_METADATA[graph_name])
            break
        except tarfile.ReadError:
            logging.info(f"Could not unzip, attempt {attempt}")
            time.sleep(10)
    else:
        data_path = download_and_extract(save_dir=data_root, graph_metadata=DATASETS_METADATA[graph_name])

    if use_weights:
        edges, weights = read_to_edges_and_weights(data_path, DATASETS_METADATA[graph_name].filetype)
    else:
        edges = read_to_edges(data_path, DATASETS_METADATA[graph_name].filetype)
        weights = np.ones(edges.shape[0], dtype=np.float64)

    graph = edges_to_gt(edges, weights, remove_isolated_nodes=remove_isolated_nodes, extract_lcc=extract_lcc)
    return graph


if __name__ == "__main__":
    graph = load_gt_graph('enron', use_weights=True)

    print(graph)
