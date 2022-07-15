from typing import Iterable, Set
import copy
import os
import pandas as pd
import numpy as np
import re

from src.utils.visaulisation_utils import make_pretty_name


def get_layout(row_columns, other_cols):
    layout = r"{"
    for col in row_columns + other_cols:
        if len(col) < 14:
            layout += "c"
        else:
            layout += r"p{15mm}"
    layout += "}"
    return layout


def create_table_start(row_columns, other_cols):
    # layout = "{" + f"{len(row_columns) * 'r' + len(other_cols) * 'c'}" + "}"
    layout = get_layout(row_columns, other_cols)
    out = "\\begin{{tabular}}{layout}".format(layout=layout)
    out = "\n".join([out, "\\toprule", " & ".join(row_columns + other_cols) + "\\\\", "\\midrule\n"])
    return out


def create_table_end():
    return "\\bottomrule\n\\end{tabular}"


def get_best_threshold(grouped, highorlow, num_sigma=1):
    best_val_threshold = {}
    for col in grouped.columns.levels[0]:
        if highorlow[col] == 'high':
            index = grouped.loc[:, (col, "mean")].argmax()
            thres = grouped.iloc[index][(col, "mean")] - num_sigma * grouped.iloc[index][(col, "std")]
        elif highorlow[col] == 'low':
            index = grouped.loc[:, (col, "mean")].argmin()
            thres = grouped.iloc[index][(col, "mean")] + num_sigma * grouped.iloc[index][(col, "std")]
        else:
            raise ValueError()

        best_val_threshold[col] = (thres, highorlow[col] == "high")
    return best_val_threshold


def tbl_elm(value, std, is_best, num_decimal=2):
    element = f"{np.round(value, decimals=num_decimal):.{num_decimal}f} \\pm {np.round(std, decimals=num_decimal):.{num_decimal}f}"
    element = "$\\mathbf{}$".format("{" + element + "}") if is_best else "${}$".format(element)
    return element


def tbl_elm_no_std(value, is_best, num_decimal=2):
    element = f"{np.round(value, decimals=num_decimal):.{num_decimal}f}"
    element = "$\\mathbf{}$".format("{" + element + "}") if is_best else "${}$".format(element)
    return element


def create_tbl_row(grouped, row_index, columns, thres):
    elements = []
    for col in columns:
        val = grouped.loc[row_index, (col, "mean")]
        std = grouped.loc[row_index, (col, "std")]
        if col.lower() in {'time (s)', 'duration'}:
            elements.append(tbl_elm_no_std(val, val >= thres[col][0] if thres[col][1] else val <= thres[col][0]))
        else:
            elements.append(tbl_elm(val, std, val >= thres[col][0] if thres[col][1] else val <= thres[col][0]))
    if isinstance(row_index, str) or isinstance(row_index, int):
        row_index = [str(row_index)]
    elif isinstance(row_index, Iterable):
        row_index = list(str(r) for r in row_index)
    out = " & ".join(row_index + elements)
    out += "\\\\\n"
    return out


def make_table(df, row_columns, other_columns, highorlow_, include_param: bool = True):
    highorlow = copy.deepcopy(highorlow_)
    if isinstance(row_columns, dict):
        df = df.rename(columns=row_columns)
        row_columns = list(row_columns.values())
    if isinstance(other_columns, dict):
        df = df.rename(columns=other_columns)
        for old_col, new_col in other_columns.items():
            highorlow[new_col] = highorlow_[old_col]
        other_columns = list(other_columns.values())

    grouped = df.loc[:, row_columns + other_columns].groupby(row_columns).agg(["mean", "std"]).sort_values(
        (other_columns[0], "mean"), ascending=False)
    grouped.index = grouped.index.set_levels(
        grouped.index.levels[2].map(lambda x: make_pretty_name(x, include_param, False)), level=2)
    thres = get_best_threshold(grouped, highorlow)
    out = create_table_start(row_columns, other_columns)

    # def order_key(x):
    #
    #     if x.lower().startswith("our"):
    #         return "zzzzzz"
    #     else:
    #         return x

    # for row in sorted(grouped.index, key=order_key):
    for row in grouped.index:
        out += create_tbl_row(grouped, row, other_columns, thres)

    out += create_table_end()
    return grouped, out


def load_all_dataframes(kemb, label_type: str, selected_algs: Set[str]):
    dfs = []
    for weighted in ["weighted", "unweighted"]:
        for directed in ["directed", "undirected"]:
            df = pd.read_json(
                f"./results/enron/node_classification_{kemb}_{weighted}_{directed}_simplified_updated.json")
            df["weights"] = "Yes" if weighted == "weighted" else "No"
            df["directed"] = "Yes" if directed == "directed" else "No"
            dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)
    selected = data["label_type"] == label_type
    if selected_algs is not None:
        selected = selected & np.logical_or.reduce([data["name"].str.startswith(s) for s in selected_algs])
    data = data.loc[selected, :]
    return data


def make_enron_table():
    kemb = 128
    table_dir = "./tables/enron"
    os.makedirs(table_dir, exist_ok=True)

    selected_algs = {"role": {
        "digraphwave_128_2",
        "graphwave_128_2",
        # "refex_128_-6",
        "refexnew_128_-6",
        "ember128",
        "degreefeatures"
    },
        "email_type": {
            "digraphwave_128_3",
            "graphwave_128_3",
            # "refex_128_-6",
            "refexnew_128_-6",
            "ember128",
            "degreefeatures"
        },
        "role_all": {
            # "digraphwave_128_2",
            "digraphwave_128_3",
            # "graphwave_128_2",
            "graphwave_128_3",
            # "refex_128_-6",
            "refexnew_128_-6",
            "ember128",
            "degreefeatures"}
    }

    for label_type in ["role", "email_type", "role_all"]:
        # table_filename = f"node_classification_{label_type}_{kemb}_{weighted}_{directed}_simplified.tex"
        table_filename = f"node_classification_{label_type}_{kemb}_simplified.tex"
        data = load_all_dataframes(kemb, label_type, selected_algs[label_type])
        _, code = make_table(data,
                             {"directed": "Directed",
                              "weights": "Weights",
                              "name": "Embedding algorithm"},
                             {
                                 "macro_f1": "F1 (macro)",
                                 # "duration": "Time (s)"
                             }
                             ,
                             {
                                 "duration": "low",
                                 "macro_f1": "high"
                             },
                             include_param=False
                             )
        with open(os.path.join(table_dir, table_filename), "w") as fp:
            fp.write(code)


if __name__ == "__main__":
    make_enron_table()
