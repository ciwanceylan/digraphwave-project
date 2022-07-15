import graph_tool as gt
import numpy as np
import pandas as pd

import src.utils.enron_data as enron_utils
import src.utils.analysis_utils as analysis_utils

ODD_TRIPLETT = [11352, 8531, 16669]


def make_vis_data(embeddings: np.ndarray, graph: gt.Graph):
    vis_df = analysis_utils.make_vis_data(embeddings, graph)

    enron_roles = pd.read_csv("./data/parsed_enron/enron_roles.csv", index_col=False)
    enron_roles = enron_roles.astype({"role": "category"})
    role_map = enron_roles.loc[:, ["node_index", "role"]]
    role_map = role_map.set_index("node_index")
    vis_df = vis_df.join(role_map)
    vis_df["role"] = vis_df["role"].cat.add_categories("Unknown").fillna("Unknown")

    enron_emails = pd.read_csv("./data/parsed_enron/enron_index2email.csv", header=None, index_col=0)
    enron_emails = enron_emails.rename(columns={1: "email"})
    vis_df = vis_df.join(enron_utils.get_email_domains(enron_emails))
    return vis_df
