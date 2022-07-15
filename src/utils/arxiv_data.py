import os
import pandas as pd
import numpy as np

import src.utils.enron_data as enrondata


def reindex_data(path_to_ca_astroph: str):
    folder = "./data/arxiv"
    os.makedirs(folder, exist_ok=True)

    arxivdata = pd.read_csv(path_to_ca_astroph, sep="\s+", comment="#", names=["source", "target"],
                            header=None)
    unique_vals = np.unique(np.concatenate((arxivdata["source"], arxivdata["target"])))
    map_to_index = pd.Series(np.arange(len(unique_vals), dtype=np.int64), index=unique_vals)
    arxivdata["reindex_source"] = arxivdata["source"].map(map_to_index)
    arxivdata["reindex_target"] = arxivdata["target"].map(map_to_index)
    enrondata.save_edges(os.path.join(folder, "arxiv_edges.tsv"),
                         arxivdata.loc[:, ["reindex_source", "reindex_target"]],
                         num_nodes=len(map_to_index))


if __name__ == "__main__":
    path_to_ca_astroph = "./data/arxiv/CA-AstroPh.txt"
    reindex_data(path_to_ca_astroph)
