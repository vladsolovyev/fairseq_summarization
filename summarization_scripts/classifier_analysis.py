from pathlib import Path

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn

languages = ["en_XX", "es_XX", "ru_RU", "tr_TR"]


def train_classifiers():
    Path("figures").mkdir(parents=True, exist_ok=True)
    for dir_name, output_file, title in zip(["base_model_with_adv/monolingual_multi",
                                             "base_model_with_adv/monolingual_with_classifier_kldivloss",
                                             "base_model_with_adv/monolingual_with_classifier_nll",
                                             "residual_drop_at_7/monolingual_with_classifier_kldivloss"],
                                            ["baseline.png", "kldivloss.png", "nll.png", "kldivloss_residual.png"],
                                            ["Baseline", "Adversarial loss (KL-div)", "Adversarial loss",
                                             "Adversarial loss (KL-div) + residual"]):
        directory = "wiki_results/2023-10-12/{}".format(dir_name)
        metrics_file = "{}/metrics.csv".format(directory)
        df = pd.read_csv(metrics_file, index_col=0)
        seaborn.set(font_scale=1.4)
        fig = seaborn.heatmap(df, cmap="YlGnBu", annot=True, fmt=".3f", vmin=0, vmax=1)
        plt.title(title)
        fig.get_figure().savefig("figures/{}".format(output_file))
        plt.close(fig.get_figure())


if __name__ == "__main__":
    train_classifiers()
