import gc
import shutil
from datetime import datetime

import pandas as pd
import torch
from GPUtil import showUtilization

from summarization_scripts.generate_summaries import generate_and_evaluate_summaries
from summarization_scripts.train_summarization import train_summarization_model

languages = ["en_XX", "es_XX", "ru_RU"]
metrics = dict()

output_dir = datetime.today().strftime('%Y-%m-%d')


def save_metrics():
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.to_csv("{}/metrics.csv".format(output_dir))


def free_memory():
    showUtilization()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    showUtilization()


def main():
    # every language separately
    for language in languages:
        checkpoint_dir = "{}/xlsum/{}".format(output_dir, language)
        train_summarization_model(data_dir="xlsum",
                                  lang_pairs="{}-{}".format(language, language),
                                  save_dir=checkpoint_dir)
        free_memory()
        metrics[language] = generate_and_evaluate_summaries(language=language,
                                                            lang_pairs="{}-{}".format(language, language),
                                                            checkpoint_dir=checkpoint_dir)
        if language != "en_XX":
            shutil.rmtree(checkpoint_dir)
        save_metrics()
        free_memory()

    # zero shot. Evaluate spanish and russian datasets using english model
    for language in languages[1:3]:
        metrics["en_XX_zero_{}".format(language)] = \
            generate_and_evaluate_summaries(language=language,
                                            lang_pairs="{}-{}".format(language, language),
                                            checkpoint_dir="{}/xlsum/en_XX".format(output_dir))
        save_metrics()
        free_memory()

    # few shot experiments. Tune english model using few data from spanish and russian datasets
    for language in languages[1:3]:
        for data_size in [10, 100, 1000, 10000]:
            checkpoint_dir = "{}/xlsum_{}/{}".format(output_dir, data_size, language)
            train_summarization_model(data_dir="xlsum_{}".format(data_size),
                                      lang_pairs="{}-{}".format(language, language),
                                      checkpoint="{}/xlsum/en_XX/checkpoint_last.pt".format(output_dir),
                                      save_dir=checkpoint_dir)
            free_memory()
            metrics["en_XX_tuned_{}_{}".format(language, data_size)] = \
                generate_and_evaluate_summaries(language=language,
                                                lang_pairs="{}-{}".format(language, language),
                                                checkpoint_dir=checkpoint_dir)
            shutil.rmtree(checkpoint_dir)
            save_metrics()
            free_memory()

    # tune english model using complete data from spanish and russian datasets
    for language in languages[1:3]:
        checkpoint_dir = "{}/xlsum/{}".format(output_dir, language)
        train_summarization_model(data_dir="xlsum",
                                  lang_pairs="{}-{}".format(language, language),
                                  checkpoint="{}/xlsum/en_XX/checkpoint_last.pt".format(output_dir),
                                  save_dir=checkpoint_dir)
        free_memory()
        metrics["en_XX_tuned_{}".format(language)] = \
            generate_and_evaluate_summaries(language=language,
                                            lang_pairs="{}-{}".format(language, language),
                                            checkpoint_dir=checkpoint_dir)
        shutil.rmtree(checkpoint_dir)
        save_metrics()
        free_memory()

    # all three languages together
    checkpoint_dir = "{}/multilingual".format(output_dir)
    train_summarization_model(data_dir="xlsum",
                              lang_pairs=",".join(["{}-{}".format(language, language) for language in languages]),
                              save_dir=checkpoint_dir)
    free_memory()
    for language in languages:
        metrics["multilingual_{}".format(language)] = \
            generate_and_evaluate_summaries(language=language,
                                            lang_pairs="{}-{}".format(language, language),
                                            checkpoint_dir=checkpoint_dir)
        save_metrics()
        free_memory()

    shutil.rmtree(checkpoint_dir)
    save_metrics()


if __name__ == "__main__":
    main()
