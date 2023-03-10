import shutil
from datetime import datetime

from summarization_scripts.generate_summaries import generate_and_evaluate_summaries
from summarization_scripts.train_summarization import train_summarization_model
from summarization_scripts.utils import free_memory, save_metrics

languages = ["en_XX", "es_XX", "ru_RU"]
lenpen = 0.6


def main():
    metrics = dict()
    output_dir = datetime.today().strftime('%Y-%m-%d')
    # every language separately
    for language in languages:
        checkpoint_dir = "{}/xlsum/{}".format(output_dir, language)
        train_summarization_model(data_dir="xlsum",
                                  lang_pairs="{}-{}".format(language, language),
                                  save_dir=checkpoint_dir)
        free_memory()
        metrics[language] = generate_and_evaluate_summaries(directory="xlsum",
                                                            source_language=language,
                                                            target_language=language,
                                                            lang_pairs="{}-{}".format(language, language),
                                                            checkpoint_dir=checkpoint_dir,
                                                            lenpen=lenpen)
        if language != "en_XX":
            shutil.rmtree(checkpoint_dir)
        save_metrics(metrics, output_dir)
        free_memory()

    # zero shot. Evaluate spanish and russian datasets using english model
    for language in languages[1:]:
        metrics["{}_zero".format(language)] = \
            generate_and_evaluate_summaries(directory="xlsum",
                                            source_language=language,
                                            target_language=language,
                                            lang_pairs="{}-{}".format(language, language),
                                            checkpoint_dir="{}/xlsum/en_XX".format(output_dir),
                                            lenpen=lenpen)
        save_metrics(metrics, output_dir)
        free_memory()

    # few shot experiments. Tune english model using few data from spanish and russian datasets
    for language in languages[1:]:
        for data_size in [10, 100, 1000, 10000]:
            checkpoint_dir = "{}/xlsum_{}/{}".format(output_dir, data_size, language)
            train_summarization_model(data_dir="xlsum_{}".format(data_size),
                                      lang_pairs="{}-{}".format(language, language),
                                      checkpoint="{}/xlsum/en_XX/checkpoint_last.pt".format(output_dir),
                                      save_dir=checkpoint_dir)
            free_memory()
            metrics["{}_tuned_{}".format(language, data_size)] = \
                generate_and_evaluate_summaries(directory="xlsum",
                                                source_language=language,
                                                target_language=language,
                                                lang_pairs="{}-{}".format(language, language),
                                                checkpoint_dir=checkpoint_dir,
                                                lenpen=lenpen)
            shutil.rmtree(checkpoint_dir)
            save_metrics(metrics, output_dir)
            free_memory()

    # tune english model using complete data from spanish and russian datasets
    for language in languages[1:]:
        checkpoint_dir = "{}/xlsum_all/{}".format(output_dir, language)
        train_summarization_model(data_dir="xlsum",
                                  lang_pairs="{}-{}".format(language, language),
                                  checkpoint="{}/xlsum/en_XX/checkpoint_last.pt".format(output_dir),
                                  save_dir=checkpoint_dir)
        free_memory()
        metrics["{}_tuned_all".format(language)] = \
            generate_and_evaluate_summaries(directory="xlsum",
                                            source_language=language,
                                            target_language=language,
                                            lang_pairs="{}-{}".format(language, language),
                                            checkpoint_dir=checkpoint_dir,
                                            lenpen=lenpen)
        shutil.rmtree(checkpoint_dir)
        save_metrics(metrics, output_dir)
        free_memory()

    # all three languages together
    checkpoint_dir = "{}/multilingual".format(output_dir)
    train_summarization_model(data_dir="xlsum",
                              lang_pairs=",".join(["{}-{}".format(language, language) for language in languages]),
                              save_dir=checkpoint_dir)
    free_memory()
    for language in languages:
        metrics["{}_multilingual".format(language)] = \
            generate_and_evaluate_summaries(directory="xlsum",
                                            source_language=language,
                                            target_language=language,
                                            lang_pairs="{}-{}".format(language, language),
                                            checkpoint_dir=checkpoint_dir,
                                            lenpen=lenpen)
        save_metrics(metrics, output_dir)
        free_memory()

    shutil.rmtree(checkpoint_dir)
    shutil.rmtree("{}/xlsum/en_XX".format(output_dir))


if __name__ == "__main__":
    main()
