import pandas as pd

import src.utils.visaulisation_utils as visutils


def load_alg_comp_results():
    df = pd.read_json("./results/synthetic/alg_compare.json")
    return df


def load_ablation_results():
    df = pd.read_json("./results/synthetic/digw_ablation.json")
    return df


if __name__ == "__main__":
    visutils.setup_matplotlib(32, 28)
    data = load_alg_comp_results()
    figs, axes = visutils.plot_synthetic_data_results(data, "alg_comp", save=True)

    data = load_ablation_results()
    figs, axes = visutils.plot_synthetic_data_results(data, "ablation", save=True)
