import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler
import tikzplotlib

import src.utils.enron_data as enrondata
import src.utils.synthetic_dataset as synthdata

default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

markers = ['v', '^', 's', '*', 'P', 'X', 'D', 'd', 'o']
linestyles = ['-', '--', '-.', ':']
colors_cb_bw = {
    "seq": ["#fef0d9", "#fdcc8a", "#fc8d59", "#d7301f"],
    "div": ["#e66101", "#fdb863", "#b2abd2", "#5e3c99"],
    "qual": ["#1b9e77", "#d95f02", "#7570b3", "#000000"]
}

colors_cb = {
    "seq_warm": ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026"],
    "seq_cold": ["#fff7fb", "#ece7f2", "#d0d1e6", "#a6bddb", "#74a9cf", "#3690c0", "#0570b0", "#045a8d", "#023858"],
    "div": ["#a50026", "#d73027", "#f46d43", "#fdae61", "#fee090", "#e0f3f8", "#abd9e9", "#74add1", "#4575b4",
            "#313695"]
}

colors_cb_github = {'qual': ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c',
                             '#dede00']}

alg_map = {"digraphwave": "Our (dir)", "graphwave": "Our (undir)", "ember128": "EMBER",
           "graphwave2018": "GraphWave2018", "furutani_maggraphwave": "Furutani et al.", "refex": "ReFeX",
           "refexnew": "ReFeX",
           "degreefeatures": r"In/out-degrees"}


def setup_matplotlib(fontsize=32, fontsize_legend=28):
    rc_extra = {
        "font.size": fontsize,
        'legend.fontsize': fontsize_legend,
        'figure.figsize': (12, 9),
        'legend.frameon': True,
        'legend.edgecolor': '1',
        'legend.facecolor': 'inherit',
        'legend.framealpha': 0.6,
        # 'text.latex.preview': True,
        'text.usetex': True,
        'svg.fonttype': 'none',
        'text.latex.preamble': r'\usepackage{libertine}',
        'font.family': 'Linux Libertine',
        'mathtext.fontset': 'custom',
        'mathtext.rm': 'libertine',
        'mathtext.it': 'libertine:italic',
        'mathtext.bf': 'libertine:bold',
        'axes.prop_cycle': cycler('color', colors_cb_github['qual']),
        'patch.facecolor': '#0072B2',
        'figure.autolayout': True,
        'lines.linewidth': 3,
    }

    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update(rc_extra)


def save_all_formats(main_dir: str, name: str):
    save_dir_png = os.path.join(main_dir, "png")
    save_dir_tex = os.path.join(main_dir, "tex")
    save_dir_pdf = os.path.join(main_dir, "pdf")
    os.makedirs(save_dir_png, exist_ok=True)
    os.makedirs(save_dir_tex, exist_ok=True)
    os.makedirs(save_dir_pdf, exist_ok=True)

    plt.savefig(os.path.join(save_dir_png, f"{name}.png"))
    plt.savefig(os.path.join(save_dir_pdf, f"{name}.pdf"))
    tikzplotlib.save(os.path.join(save_dir_tex, f"{name}.tex"))


def get_device(s):
    return re.sub(r"cuda\[\d\]", "gpu", s)


def make_pretty_name(name: str, include_param: bool, include_device: bool = False):
    selected_param = {"a", "at", "t", "notransform"}
    name = name.replace("no_transform", "notransform")
    name = alg_map.get(name, name)
    split1 = name.split("::")
    undir = False
    if len(split1) > 1:
        undir = True
    name = split1[0]

    out = name.split("_")
    device = ""
    other = ""
    undir = "" if not undir else "|undir."
    if "cpu" in out[-1] or "cuda" in out[-1]:
        param = out[1:-1]
        if include_device:
            device = "[" + get_device(out[-1]).upper() + "]"
    else:
        param = out[1:]

    param = [p.upper() for p in param if p.lower() in selected_param or p.isnumeric()]

    if include_param and len(param) > 0:
        other = f"[{'/'.join(param)}]"

    alg = alg_map.get(out[0], out[0])
    return "".join([alg, other, device, undir]).strip()


def make_pretty_name_ablation(name: str, include_param: bool = True, include_device: bool = False):
    selected_param = {"a", "at", "t"}
    name = name.replace("no_transform", "notransform")
    name = alg_map.get(name, name)
    split1 = name.split("::")
    undir = False
    if len(split1) > 1:
        undir = True
    name = split1[0]
    # Specific for ablation
    name = name.replace("Our", "")
    out = name.split("_")
    device = ""
    other = ""
    # undir = "" if not undir else "|undir."
    if "cpu" in out[-1] or "cuda" in out[-1]:
        param = out[1:-1]
        if include_device:
            device = "[" + get_device(out[-1]).upper() + "]"
    else:
        param = out[1:]
    param = [p.upper() for p in param if p.lower() in selected_param]
    if undir:
        res = "No enhanc. + \n undir. graph"
    elif param:
        res = f"{'-'.join(param)} enhanc."
    else:
        res = "No enhanc."
    return res


def get_synthlabels_and_gridpoints():
    synthlabels = synthdata.get_unique_labels()
    synth_labels_df = pd.Series(synthlabels).str.split("_", expand=True)
    synth_labels_df["order"] = synth_labels_df.index
    grid_points = synth_labels_df.groupby([1, 0]).agg({"order": "max"})
    return synthlabels, grid_points


def synthetic_data_cm(cm):
    fig, ax = plt.subplots(figsize=(20, 16))

    synthlabels, grid_points = get_synthlabels_and_gridpoints()

    # cm = grouped.loc[(5, "digraphwave_128_3_at_no_transform_cpu"), "cm"]
    cm_norm = cm / np.sum(cm, axis=1, keepdims=True)
    g = sns.heatmap(cm_norm, vmin=0, vmax=1, xticklabels=synthlabels[::6], linewidths=0.2, linecolor=(0.2, 0.2, 0.2))

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ticklocs = grid_points.loc['1', "order"] + 1 - 0.5 * np.diff(
        np.concatenate(([0], grid_points.loc['1', "order"] + 1)))
    ax.set_xticks(ticklocs, grid_points.loc['0', "order"].index)
    g.set_xticklabels(labels=g.get_xticklabels(), ha="center", rotation=0, fontsize=28)

    ax.set_yticks(ticklocs, grid_points.loc['0', "order"].index)
    g.set_yticklabels(labels=g.get_yticklabels(), va="center", rotation=90, fontsize=28)

    for val in grid_points.loc['1', "order"]:
        ax.plot(ax.get_xlim(), [val + 1, val + 1], 'w-')
        ax.plot([val + 1, val + 1], ax.get_ylim(), 'w-')

    for val in grid_points.loc['0', "order"]:
        ax.plot(ax.get_xlim(), [val + 1, val + 1], 'w:')
        ax.plot([val + 1, val + 1], ax.get_ylim(), 'w:')

    return fig, ax


def synthetic_data_cm_diff(cm1, cm2):
    cm1_norm = cm1 / np.sum(cm1, axis=1, keepdims=True)
    cm2_norm = cm2 / np.sum(cm2, axis=1, keepdims=True)
    cm_diff = cm2_norm - cm1_norm

    fig, ax = plt.subplots(figsize=(20, 16))
    synthlabels, grid_points = get_synthlabels_and_gridpoints()
    # cmap = sns.color_palette("vlag", as_cmap=True)
    cmap = sns.color_palette("icefire", as_cmap=True)
    linecolor = 'w'
    g = sns.heatmap(cm_diff, cmap=cmap, vmin=-1, vmax=1, xticklabels=synthlabels[::6], linewidths=0.2, linecolor=(0.2, 0.2, 0.2))

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ticklocs = grid_points.loc['1', "order"] + 1 - 0.5 * np.diff(
        np.concatenate(([0], grid_points.loc['1', "order"] + 1)))
    ax.set_xticks(ticklocs, grid_points.loc['0', "order"].index)
    g.set_xticklabels(labels=g.get_xticklabels(), ha="center", rotation=0, fontsize=28)

    ax.set_yticks(ticklocs, grid_points.loc['0', "order"].index)
    g.set_yticklabels(labels=g.get_yticklabels(), va="center", rotation=90, fontsize=28)

    for val in grid_points.loc['1', "order"]:
        ax.plot(ax.get_xlim(), [val + 1, val + 1], f'{linecolor}-')
        ax.plot([val + 1, val + 1], ax.get_ylim(), f'{linecolor}-')

    for val in grid_points.loc['0', "order"]:
        ax.plot(ax.get_xlim(), [val + 1, val + 1], f'{linecolor}:')
        ax.plot([val + 1, val + 1], ax.get_ylim(), f'{linecolor}:')

    return fig, ax


def enron_data_cm(cm, label_type: str, simplified: bool):
    if simplified:
        _, enron_labels = enrondata.load_simplified_labels("data/enron/parsed_enron", label_type)
    else:
        _, enron_labels = enrondata.load_labels("data/enron/parsed_enron", label_type)
    ticklabels = enron_labels

    if label_type == "role":
        fig, ax = plt.subplots(figsize=(15, 12))
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
    cm_norm = cm / np.sum(cm, axis=1, keepdims=True)
    g = sns.heatmap(cm_norm, vmin=0, vmax=1, annot=True, xticklabels=ticklabels, yticklabels=ticklabels)
    g.set_xticklabels(labels=g.get_xticklabels(), ha="center", rotation=0, fontsize=28)
    g.set_yticklabels(labels=g.get_yticklabels(), va="center", fontsize=28)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    return fig, ax


def get_avg_confusion_matrices_synthetic(data: pd.DataFrame):
    data["cm"] = data["cm"].apply(np.asarray)
    out = data.groupby(["num_connecting_edges", "name"]).agg({"cm": "mean"})
    return out


def get_avg_confusion_matrices_enron(data: pd.DataFrame):
    data["cm"] = data["cm"].apply(np.asarray)
    out = data.groupby(["label_type", "name"]).agg({"cm": "mean"})
    return out


def make_legend_pretty(g, include_param: bool, include_device: bool):
    g.get_legend().set_title(None)
    for legend_text in g.get_legend().texts:
        legend_text.set_text(make_pretty_name(legend_text.get_text(), include_param, include_device))


def make_legend_pretty_ablation(g, include_param: bool, include_device: bool):
    g.get_legend().set_title(None)
    for legend_text in g.get_legend().texts:
        legend_text.set_text(make_pretty_name_ablation(legend_text.get_text(), include_param, include_device))


def make_legend_pretty_new(legend_obj, include_param: bool, include_device: bool):
    legend_obj.set_title(None)
    for legend_text in legend_obj.texts:
        legend_text.set_text(make_pretty_name(legend_text.get_text(), include_param, include_device))


def plot_synthetic_data_results(data: pd.DataFrame, mode, save=False):
    figs = []
    axes = []
    include_param = mode == "ablation"
    include_device = False

    if mode == "alg_comp":
        selected_algs = {"digraphwave_128_3_at_no_transform_cpu",
                         "ember128",
                         "furutani_maggraphwave",
                         "graphwave2018",
                         # "refex_128_-6",
                         # "refexnew_128_-3",
                         "refexnew_128_-6",
                         "degreefeatures"}
        ignored_algs = {"graphwave2018::undir"}
        make_legend_pretty_fun = make_legend_pretty
    elif mode == "ablation":
        selected_algs = {"digraphwave_128_3",
                         "graphwave_128_3_a_no_transform_cpu::undirected"}
        ignored_algs = {}
        make_legend_pretty_fun = make_legend_pretty_ablation
    else:
        raise ValueError(f"unknown mode {mode}")
    alg_order = {
        "digraphwave_128_3_at_no_transform_cpu": "0",
        "refexnew_128_-6": "1",
        "ember128": "2",
        "furutani_maggraphwave": "3",
        "graphwave2018": "4",
        "degreefeatures": "5"
    }

    save_dir = f"./figures/synthetic/{mode}"

    selected = np.logical_or.reduce([data["name"].str.startswith(s) for s in selected_algs])
    ignored = np.logical_or.reduce([data["name"].str.startswith(s) for s in ignored_algs])
    selected = selected & ~ignored
    data = data.loc[selected, :]
    if mode == "alg_comp":
        data = data.sort_values(by="name", ignore_index=True, key=lambda x: x.replace(alg_order))

    fig, ax = plt.subplots()
    g = sns.lineplot(data=data, x="num_connecting_edges", y="macro_f1", hue="name", style="name",
                     markers=True, markersize=14, legend=True)
    ax.set_xlabel("Number added noise edges")
    ax.set_ylabel("F1 (macro)")
    ax.set_xlim([-.1, 5.1])
    ax.set_ylim([-.05, 1.1])
    # ax.axis([-.1, 5.1, -.05, 1.05])
    make_legend_pretty_fun(g, include_param, include_device)
    figs.append(fig)
    axes.append(ax)
    if save:
        save_all_formats(save_dir, "f1")

    fig, ax = plt.subplots()
    g = sns.lineplot(data=data, x="num_connecting_edges", y="adjusted_mi", hue="name", style="name",
                     markers=True, markersize=14, legend=True)

    ax.set_xlabel("Number added noise edges")
    ax.set_ylabel("Adjusted mutual information")
    ax.set_xlim([-.1, 5.1])
    ax.set_ylim([0.3, 1])
    # ax.axis([-.1, 5.1, -.05, 1.05])
    make_legend_pretty_fun(g, include_param, include_device)
    figs.append(fig)
    axes.append(ax)
    if save:
        save_all_formats(save_dir, "ami")

    fig, ax = plt.subplots()
    g = sns.lineplot(data=data, x="num_connecting_edges", y="adjusted_rand_score", hue="name", style="name",
                     markers=True, markersize=14, legend=True)

    ax.set_xlabel("Number added noise edges")
    ax.set_ylabel("Adjusted rand score")
    ax.set_xlim([-.1, 5.1])
    ax.set_ylim([-.05, 1])
    # ax.axis([-.1, 5.1, -.05, 1.05])
    make_legend_pretty_fun(g, include_param, include_device)
    figs.append(fig)
    axes.append(ax)
    if save:
        save_all_formats(save_dir, "ars")

    fig, ax = plt.subplots()
    g = sns.lineplot(data=data, x="num_connecting_edges", y="fmi", hue="name", style="name",
                     markers=True, markersize=14, legend=True)

    ax.set_xlabel("Number added noise edges")
    ax.set_ylabel("Fowlkes-Mallows score")
    ax.set_xlim([-.1, 5.1])
    ax.set_ylim([-.05, 1])
    # ax.axis([-.1, 5.1, -.05, 1.05])
    make_legend_pretty_fun(g, include_param, include_device)
    figs.append(fig)
    axes.append(ax)
    if save:
        save_all_formats(save_dir, "fmi")

    fig, ax = plt.subplots()
    g = sns.lineplot(data=data, x="num_connecting_edges", y="davies_bouldin_score", hue="name", style="name",
                     markers=True, markersize=14, legend=True)
    ax.set_xlabel("Number added noise edges")
    ax.set_ylabel("David-Bouldin score")
    ax.set_xlim([-.1, 5.1])
    ax.set_ylim([-.05, 10])
    make_legend_pretty_fun(g, include_param, include_device)
    figs.append(fig)
    axes.append(ax)
    if save:
        save_all_formats(save_dir, "dbs")

    return figs, axes


def load_alignment_results_data(alignment_type: str, directed: bool, weighted: bool):
    if alignment_type not in {'arxiv', 'enron_all_internal', 'enron_internal'}:
        raise ValueError(f"Unknown alignment type {alignment_type}")
    weighted = "weighted" if weighted else "unweighted"
    directed = "directed" if directed else "undirected"
    path = f"./results/alignment/alignment_{alignment_type}_{weighted}_{directed}.json"
    df = pd.read_json(path)
    return df


def load_both_dir_and_weight(alignment_type: str):
    dfs = []
    for weighted, weighted_str in zip([True, False], ["weighted", "unweighted"]):
        for directed, directed_str in zip([True, False], ["directed", "undirected"]):
            data = load_alignment_results_data(alignment_type, directed, weighted=weighted)
            data['directed'] = directed
            data['weighted'] = weighted
            # if not directed:
            #     data['name'] = data['name'] + "::undir"
            dfs.append(data)

    return pd.concat(dfs, ignore_index=True)


def plot_alignment_results_seperate(alignment_type: str, directed: bool, weighted: bool,
                                    save: bool = False, include_param=True, kvals=(1, 10)):
    figs = []
    axes = []
    # include_param = True
    include_device = False

    selected_algs = {"digraphwave_128_2_at_no_transform_cuda[3]",
                     "graphwave_128_2_a_no_transform_cuda[3]",
                     "ember128",
                     "furutani_maggraphwave",
                     "graphwave2018",
                     # "refex_128_-6",
                     # "refexnew_128_-3",
                     "refexnew_128_-6",
                     "degreefeatures"}

    alg_order = {
        "digraphwave_128_2_at_no_transform_cuda[3]": "0",
        "graphwave_128_2_a_no_transform_cuda[3]": "0",
        "refexnew_128_-6": "1",
        "ember128": "2",
        "furutani_maggraphwave": "3",
        "graphwave2018": "4",
        "degreefeatures": "5"
    }

    ignored_algs = {"digraphwave_128_2_at_cuda[3]", "graphwave_128_2_a_cuda[3]"}

    save_dir = f"./figures/alignment/{alignment_type}/"
    data = load_alignment_results_data(alignment_type, directed=directed, weighted=weighted)

    selected = np.logical_or.reduce([data["name"].str.startswith(s) for s in selected_algs])
    ignored = np.logical_or.reduce([data["name"].str.startswith(s) for s in ignored_algs])
    selected = selected & ~ignored
    data = data.loc[selected, :]
    data = data.sort_values(by="name", ignore_index=True, key=lambda x: x.replace(alg_order))

    for k in kvals:
        kcol = f"k@{k}"

        fig, ax = plt.subplots()
        g = sns.lineplot(data=data, x="noise_p", y=kcol, hue="name", style="name",
                         markers=True, markersize=14, legend=True)
        ax.set_xlabel("Noise level")
        ax.set_ylabel(f"Top-{k} accuracy")
        # ax.set_xlim([-.1, 5.1])
        ax.set_ylim([-.05, 1.])
        make_legend_pretty(g, include_param, include_device)
        figs.append(fig)
        axes.append(ax)
        if save:
            name = f"{'weighted' if weighted else 'unweighted'}_{'directed' if directed else 'undirected'}_{kcol}"
            save_all_formats(save_dir, name)


def plot_alignment_results_joint(alignment_type: str, save: bool = False, kvals=(1, 10)):
    figs = []
    axes = []
    include_param = True
    include_device = False

    selected_algs = {"digraphwave_128_2_at_no_transform_cuda[3]",
                     "graphwave_128_2_a_no_transform_cuda[3]",
                     "ember128",
                     "furutani_maggraphwave",
                     "graphwave2018",
                     # "refex_128_-6",
                     # "refexnew_128_-3",
                     "refexnew_128_-6",
                     "degreefeatures"}

    alg_order = {
        "digraphwave_128_2_at_no_transform_cuda[3]": "0",
        "graphwave_128_2_a_no_transform_cuda[3]": "0",
        "refexnew_128_-6": "1",
        "ember128": "2",
        "furutani_maggraphwave": "3",
        "graphwave2018": "4",
        "degreefeatures": "5"
    }
    ignored_algs = {"digraphwave_128_2_at_cuda[3]", "graphwave_128_2_a_cuda[3]"}

    save_dir = f"./figures/alignment/{alignment_type}/"
    if alignment_type == 'arxiv':
        raise ValueError("Joint plot not applicable to arxiv")
    else:
        data = load_both_dir_and_weight(alignment_type)
    # data = load_alignment_results_data(alignment_type, directed=directed, weighted=weighted)

    selected = np.logical_or.reduce([data["name"].str.startswith(s) for s in selected_algs])
    ignored = np.logical_or.reduce([data["name"].str.startswith(s) for s in ignored_algs])
    selected = selected & ~ignored
    data = data.loc[selected, :]
    data = data.sort_values(by="name", ignore_index=True, key=lambda x: x.replace(alg_order))

    for k in kvals:
        kcol = f"k@{k}"
        facet_kws = {
            "subplot_kws": {
                "xlabel": "Noise level",
                "ylabel": f"Top-{k} accuracy",
                "ylim": [-.05, 1.]
            }
        }

        # fig, ax = plt.subplots()
        sns.set_style("ticks", {'axes.grid': True})
        g = sns.relplot(data=data, x="noise_p", y=kcol, hue="name", style="name", row='weighted', col="directed",
                        kind="line", markers=True, markersize=14, legend=True, facet_kws=facet_kws)
        # plt.grid()
        # g.set_xlabel("Noise level")
        # g.set_ylabel(f"Top-{k} accuracy")
        # ax.set_xlim([-.1, 5.1])
        # g.set_ylim([-.05, 1.])
        make_legend_pretty_new(g.legend, include_param, include_device)
        # figs.append(fig)
        # axes.append(ax)
        if save:
            name = f"joint_{kcol}"
            save_all_formats(save_dir, name)
