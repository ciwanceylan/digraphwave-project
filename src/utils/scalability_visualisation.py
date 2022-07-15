import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import seaborn as sns
from cycler import cycler
import tikzplotlib
import os

import src.utils.visaulisation_utils as visutils


def load_alg_results():
    # df1 = pd.read_json("./cluster_results/results/scalability/alg_compare_165305929072.json")
    # df2 = pd.read_json("./cluster_results/results/scalability/alg_compare_165329882833.json")
    # df3 = pd.read_json("./cluster_results/results/scalability/alg_compare_165338135433.json")
    # alg_comp_df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
    alg_comp_df = pd.read_json("./results/scalability/alg_comp.json")
    return alg_comp_df


def agg_over_reps(df: pd.DataFrame):
    df = df.loc[df["outcome"] == "completed", :] \
        .groupby(["alg", "num_nodes", "m"]) \
        .agg({"num_edges": ["mean", "std"], "duration": ["mean", "std"]})
    df.columns = ['_'.join(col).rstrip('_') for col in df.columns.values]
    return df


def post_process_alg_data(df: pd.DataFrame):
    order_rename_map = {
        "graphwave_a_cpu": "0graphwave_a_cpu",
        "ember": "2ember",
        "furutani_maggraphwave": "3furutani_maggraphwave",
        # "refex_-5": "3refex_-5",
        "refexnew_-6": "1refexnew_-6",
        "graphwave2018": "4graphwave2018",
        "graphwave_a_cuda[1]": "5graphwave_a_cuda[1]",
    }

    algs_rename_map = {
        "0graphwave_a_cpu": "Our [CPU]",
        "2ember": "EMBER",
        "3furutani_maggraphwave": "Furutani et al.",
        # "3refex_-5": "ReFeX",
        "1refexnew_-6": "ReFeX",
        "4graphwave2018": "GraphWave2018",
        "5graphwave_a_cuda[1]": "Our [GPU]",
    }

    the_data = df.loc[
               (df["alg"].isin(order_rename_map)) &
               (df["outcome"] == "completed"),
               :].copy()
    the_data['alg'] = the_data['alg'].replace(order_rename_map)
    the_data = the_data.sort_values("alg")
    the_data['alg'] = the_data['alg'].replace(algs_rename_map)
    return the_data


def plot_alg_compare_multi_m(alg_comp_df=None, save: bool = True):
    fig, ax = plt.subplots()
    if alg_comp_df is None:
        alg_comp_df = load_alg_results()
    the_data = post_process_alg_data(alg_comp_df)

    plt_res = sns.relplot(
        data=the_data,
        x="num_nodes",
        y="duration",
        hue='alg',
        style='alg',
        markers=True,  # ['o', 's', 'd', '^', 'x', '*'],
        legend=True,
        markersize=14,
        col="m",
        kind='line'
    )
    plt_res.set(xlabel="Number of nodes", ylabel="Time (s)")
    plt_res.set(xscale='log', yscale='log')
    plt_res.get_legend().set_title(None)

    if save:
        visutils.save_all_formats("figures/scalability", f"alg_compare_multi_m")
 
    return fig, ax, plt_res


def plot_alg_compare_single_m(mvalue=1, alg_comp_df=None, save: bool = True):
    fig, ax = plt.subplots()
    if alg_comp_df is None:
        alg_comp_df = load_alg_results()
    the_data = post_process_alg_data(alg_comp_df)
    the_data = the_data.loc[the_data["m"] == mvalue, :]

    plt_res = sns.lineplot(
        data=the_data,
        x="num_nodes",
        y="duration",
        hue='alg',
        style='alg',
        markers=True,  # ['o', 's', 'd', '^', 'x', '*'],
        legend=True,
        markersize=14,
    )
    plt_res.set(xlabel="Number of nodes", ylabel="Time (s)")
    plt_res.set(xscale='log', yscale='log')
    sns.move_legend(plt_res, "upper left")
    plt_res.get_legend().set_title(None)
    plt.grid(True, which="both")
    plt.grid(True, which="minor", color='0.95')
    plt_res.set_ylim([4e-2, 10e4])

    add_horizonal(plt_res, 60, "1min")
    add_horizonal(plt_res, 3600, "1h")

    # plt_res.get_legend().set_location("southwest")
    if save:
        visutils.save_all_formats("figures/scalability", f"alg_compare_m_{mvalue}")

    return fig, ax, plt_res


def add_horizonal(ax, val, label):
    ax.axhline(y=val, color="k", linestyle=":")
    trans = transforms.blended_transform_factory(
        ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0.007, val-10, label, fontsize=24, color="k", transform=trans, ha="right", va="center")


if __name__ == "__main__":
    visutils.setup_matplotlib()
    g = plot_alg_compare_single_m(1, save=True)
    g = plot_alg_compare_single_m(5, save=True)
    g = plot_alg_compare_single_m(10, save=True)
