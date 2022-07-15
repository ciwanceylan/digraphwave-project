import src.utils.visaulisation_utils as visutils

if __name__ == "__main__":
    visutils.setup_matplotlib(36, 32)

    visutils.plot_alignment_results_seperate("arxiv", directed=False, weighted=False, save=True, include_param=False)

    for directed in [True, False]:
        for weighted in [True, False]:
            visutils.plot_alignment_results_seperate("enron_all_internal", directed=directed,
                                                     weighted=weighted, save=True, include_param=False)
