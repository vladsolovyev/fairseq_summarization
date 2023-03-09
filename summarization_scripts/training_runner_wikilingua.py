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
    # two crosslingual cases separately
    for language in languages[1:]:
        checkpoint_dir = "{}/wikilingua/{}-en_XX".format(output_dir, language)
        train_summarization_model(data_dir="wikilingua_cross_{}-en_XX".format(language),
                                  lang_pairs="{}-en_XX".format(language),
                                  save_dir=checkpoint_dir)
        free_memory()
        metrics["{}-en_XX".format(language)] = \
            generate_and_evaluate_summaries(directory="wikilingua_cross_{}-en_XX".format(language),
                                            source_language=language,
                                            target_language="en_XX",
                                            lang_pairs="{}-en_XX".format(language),
                                            checkpoint_dir=checkpoint_dir)

    # all three languages together, but monolingual data
    checkpoint_dir = "{}/multilingual".format(output_dir)
    train_summarization_model(data_dir="wikilingua_mono",
                              lang_pairs=",".join(["{}-{}".format(language, language) for language in languages]),
                              save_dir=checkpoint_dir)
    free_memory()
    for language in languages[1:]:
        metrics["zero_{}-en_XX".format(language)] = \
            generate_and_evaluate_summaries(directory="wikilingua_cross_{}-en_XX".format(language),
                                            source_language=language,
                                            target_language="en_XX",
                                            lang_pairs="{}-en_XX".format(language),
                                            checkpoint_dir=checkpoint_dir)
        save_metrics()
        free_memory()

    # few shot experiments.
    # Tune multilingual model using few data from spanish-english and russian-english datasets
    for language in languages[1:3]:
        for data_size in [10, 100, 1000, 10000]:
            checkpoint_dir = "{}/wikilingua_{}/{}-en_XX".format(output_dir, data_size, language)
            train_summarization_model(data_dir="wikilingua_cross_{}-en_XX_{}".format(language, data_size),
                                      lang_pairs="{}-en_XX".format(language),
                                      checkpoint="{}/multilingual/checkpoint_last.pt".format(output_dir),
                                      save_dir=checkpoint_dir)
            free_memory()
            metrics["tuned_{}-en_XX_{}".format(language, data_size)] = \
                generate_and_evaluate_summaries(directory="wikilingua_cross_{}-en_XX".format(language),
                                                source_language=language,
                                                target_language="en_XX",
                                                lang_pairs="{}-en_XX".format(language),
                                                checkpoint_dir=checkpoint_dir)
            shutil.rmtree(checkpoint_dir)
            save_metrics()
            free_memory()

    # tune multilingual model using complete data from spanish-english and russian-english datasets
    for language in languages[1:3]:
        checkpoint_dir = "{}/wikilingua_all/{}-en_XX".format(output_dir, data_size, language)
        train_summarization_model(data_dir="wikilingua_cross_{}-en_XX".format(language),
                                  lang_pairs="{}-en_XX".format(language),
                                  checkpoint="{}/multilingual/checkpoint_last.pt".format(output_dir),
                                  save_dir=checkpoint_dir)
        free_memory()
        metrics["tuned_{}-en_XX".format(language)] = \
            generate_and_evaluate_summaries(directory="wikilingua_cross_{}-en_XX".format(language),
                                            source_language=language,
                                            target_language="en_XX",
                                            lang_pairs="{}-en_XX".format(language),
                                            checkpoint_dir=checkpoint_dir)
        shutil.rmtree(checkpoint_dir)
        save_metrics()
        free_memory()


if __name__ == "__main__":
    main()
