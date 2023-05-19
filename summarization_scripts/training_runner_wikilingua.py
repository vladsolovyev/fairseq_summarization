import shutil

from summarization_scripts.generate_summaries import generate_and_evaluate_summaries
from summarization_scripts.train_summarization import train_summarization_model
from summarization_scripts.utils import free_memory, save_metrics

languages = ["de_DE", "en_XX", "es_XX", "ru_RU"]
lenpen = "1.0"
min_len = "10"


def run_wikilingua_experiments(encoder_drop_residual=None, prefix="", freeze_encoder_layers="0"):
    metrics = dict()
    output_dir = "wiki_results/{}".format(prefix)

    # two crosslingual cases (spanish-english and russian-english) together as baseline
    checkpoint_dir = "{}/baseline".format(output_dir)
    train_summarization_model(data_dir="wikilingua",
                              lang_pairs=",".join(["{}-en_XX".format(language) for language in languages[2:]]),
                              save_dir=checkpoint_dir,
                              encoder_drop_residual=encoder_drop_residual,
                              freeze_encoder_layers=freeze_encoder_layers)
    free_memory()

    for language in languages[2:]:
        metrics["{}-baseline".format(language)] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language,
                                            target_language="en_XX",
                                            lang_pairs="{}-en_XX".format(language),
                                            checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            min_len=min_len)
        shutil.rmtree(checkpoint_dir)
        save_metrics(metrics, output_dir)
        free_memory()

    # english, spanish, russian together, but monolingual data
    checkpoint_dir = "{}/monolingual".format(output_dir)
    train_summarization_model(data_dir="wikilingua",
                              lang_pairs=",".join(["{}-{}".format(language) for language in languages[1:]]),
                              save_dir=checkpoint_dir,
                              encoder_drop_residual=encoder_drop_residual,
                              freeze_encoder_layers=freeze_encoder_layers)
    free_memory()
    for language in languages[2:]:
        metrics["{}_monolingual".format(language)] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language,
                                            target_language="en_XX",
                                            lang_pairs="{}-en_XX".format(language),
                                            checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            min_len=min_len)
        save_metrics(metrics, output_dir)
        free_memory()

    # few shot experiments.
    # Tune monolingual model using few data from spanish-english and russian-english datasets
    for language in languages[2:]:
        for data_size, validate_interval in zip([10, 100, 1000, 10000],
                                                ["5", "3", "1", "1"]):
            checkpoint_dir = "{}/wikilingua_{}/{}-en_XX".format(output_dir, data_size, language)
            train_summarization_model(data_dir="wikilingua_{}".format(data_size),
                                      lang_pairs="{}-en_XX".format(language),
                                      checkpoint="{}/monolingual/checkpoint_best.pt".format(output_dir),
                                      save_dir=checkpoint_dir,
                                      max_update="20000",
                                      validate_interval=validate_interval,
                                      validate_interval_updates="0",
                                      encoder_drop_residual=encoder_drop_residual,
                                      num_workers="1")
            free_memory()
            metrics["{}-monolingual_{}".format(language, data_size)] = \
                generate_and_evaluate_summaries(directory="wikilingua",
                                                source_language=language,
                                                target_language="en_XX",
                                                lang_pairs="{}-en_XX".format(language),
                                                checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                                lenpen=lenpen,
                                                min_len=min_len)
            shutil.rmtree(checkpoint_dir)
            save_metrics(metrics, output_dir)
            free_memory()

    # tune monolingual model using complete data from spanish-english and russian-english datasets
    for language in languages[2:]:
        checkpoint_dir = "{}/wikilingua_all/{}-en_XX".format(output_dir, language)
        train_summarization_model(data_dir="wikilingua",
                                  lang_pairs="{}-en_XX".format(language),
                                  checkpoint="{}/monolingual/checkpoint_best.pt".format(output_dir),
                                  save_dir=checkpoint_dir,
                                  encoder_drop_residual=encoder_drop_residual)
        free_memory()
        metrics["{}-monolingual_all".format(language)] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language,
                                            target_language="en_XX",
                                            lang_pairs="{}-en_XX".format(language),
                                            checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            min_len=min_len)
        shutil.rmtree(checkpoint_dir)
        save_metrics(metrics, output_dir)
        free_memory()

    # 1-to-1 case with english data
    checkpoint_dir = "{}/1-to-1".format(output_dir)
    train_summarization_model(data_dir="wikilingua",
                              lang_pairs="en_XX-en_XX",
                              save_dir=checkpoint_dir,
                              encoder_drop_residual=encoder_drop_residual,
                              freeze_encoder_layers=freeze_encoder_layers)
    free_memory()
    for language in languages[2:]:
        metrics["{}_1-to-1".format(language)] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language,
                                            target_language="en_XX",
                                            lang_pairs="{}-en_XX".format(language),
                                            checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            min_len=min_len)
        save_metrics(metrics, output_dir)
        free_memory()

    # few shot experiments.
    # Tune 1-to-1 model using few data from spanish-english and russian-english datasets
    for language in languages[2:]:
        for data_size, validate_interval in zip([10, 100, 1000, 10000],
                                                ["5", "3", "1", "1"]):
            checkpoint_dir = "{}/wikilingua_{}/{}-en_XX".format(output_dir, data_size, language)
            train_summarization_model(data_dir="wikilingua_{}".format(data_size),
                                      lang_pairs="{}-en_XX".format(language),
                                      checkpoint="{}/1-to-1/checkpoint_best.pt".format(output_dir),
                                      save_dir=checkpoint_dir,
                                      max_update="20000",
                                      validate_interval=validate_interval,
                                      validate_interval_updates="0",
                                      encoder_drop_residual=encoder_drop_residual,
                                      num_workers="1")
            free_memory()
            metrics["{}-1-to-1_tuned_{}".format(language, data_size)] = \
                generate_and_evaluate_summaries(directory="wikilingua",
                                                source_language=language,
                                                target_language="en_XX",
                                                lang_pairs="{}-en_XX".format(language),
                                                checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                                lenpen=lenpen,
                                                min_len=min_len)
            shutil.rmtree(checkpoint_dir)
            save_metrics(metrics, output_dir)
            free_memory()

    # tune 1-to-1 model using complete data from spanish-english and russian-english datasets
    for language in languages[2:]:
        checkpoint_dir = "{}/wikilingua_all/{}-en_XX".format(output_dir, language)
        train_summarization_model(data_dir="wikilingua",
                                  lang_pairs="{}-en_XX".format(language),
                                  checkpoint="{}/1-to-1/checkpoint_best.pt".format(output_dir),
                                  save_dir=checkpoint_dir,
                                  encoder_drop_residual=encoder_drop_residual)
        free_memory()
        metrics["{}-1-to-1_tuned_all".format(language)] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language,
                                            target_language="en_XX",
                                            lang_pairs="{}-en_XX".format(language),
                                            checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            min_len=min_len)
        shutil.rmtree(checkpoint_dir)
        save_metrics(metrics, output_dir)
        free_memory()

    # 1-to-n case with crosslingual data from german to english/spanish/russian
    checkpoint_dir = "{}/1-to-n".format(output_dir)
    train_summarization_model(data_dir="wikilingua",
                              lang_pairs=",".join(["{}-de_DE".format(language) for language in languages[1:]]),
                              save_dir=checkpoint_dir,
                              encoder_drop_residual=encoder_drop_residual,
                              freeze_encoder_layers=freeze_encoder_layers)
    free_memory()
    for language in languages[2:]:
        metrics["{}_1-to-n".format(language)] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language,
                                            target_language="en_XX",
                                            lang_pairs="{}-en_XX".format(language),
                                            checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            min_len=min_len)
        save_metrics(metrics, output_dir)
        free_memory()

    # few shot experiments.
    # Tune 1-to-n model using few data from spanish-english and russian-english datasets
    for language in languages[2:]:
        for data_size, validate_interval in zip([10, 100, 1000, 10000],
                                                ["5", "3", "1", "1"]):
            checkpoint_dir = "{}/wikilingua_{}/{}-en_XX".format(output_dir, data_size, language)
            train_summarization_model(data_dir="wikilingua_{}".format(data_size),
                                      lang_pairs="{}-en_XX".format(language),
                                      checkpoint="{}/1-to-n/checkpoint_best.pt".format(output_dir),
                                      save_dir=checkpoint_dir,
                                      max_update="20000",
                                      validate_interval=validate_interval,
                                      validate_interval_updates="0",
                                      encoder_drop_residual=encoder_drop_residual,
                                      num_workers="1")
            free_memory()
            metrics["{}-1-to-n_tuned_{}".format(language, data_size)] = \
                generate_and_evaluate_summaries(directory="wikilingua",
                                                source_language=language,
                                                target_language="en_XX",
                                                lang_pairs="{}-en_XX".format(language),
                                                checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                                lenpen=lenpen,
                                                min_len=min_len)
            shutil.rmtree(checkpoint_dir)
            save_metrics(metrics, output_dir)
            free_memory()

    # tune 1-to-n model using complete data from spanish-english and russian-english datasets
    for language in languages[2:]:
        checkpoint_dir = "{}/wikilingua_all/{}-en_XX".format(output_dir, language)
        train_summarization_model(data_dir="wikilingua",
                                  lang_pairs="{}-en_XX".format(language),
                                  checkpoint="{}/1-to-n/checkpoint_best.pt".format(output_dir),
                                  save_dir=checkpoint_dir,
                                  encoder_drop_residual=encoder_drop_residual)
        free_memory()
        metrics["{}-1-to-n_tuned_all".format(language)] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language,
                                            target_language="en_XX",
                                            lang_pairs="{}-en_XX".format(language),
                                            checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            min_len=min_len)
        shutil.rmtree(checkpoint_dir)
        save_metrics(metrics, output_dir)
        free_memory()

    # n-to-1-to-n case with crosslingual data from german to english/spanish/russian
    checkpoint_dir = "{}/n-to-1-to-n".format(output_dir)
    train_summarization_model(data_dir="wikilingua",
                              lang_pairs=",".join(["{}-de_DE".format(language) for language in languages[1:]]),
                              save_dir=checkpoint_dir,
                              encoder_drop_residual=encoder_drop_residual,
                              freeze_encoder_layers=freeze_encoder_layers)
    free_memory()
    for language in languages[2:]:
        metrics["{}_n-to-1-to-n".format(language)] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language,
                                            target_language="en_XX",
                                            lang_pairs="{}-en_XX".format(language),
                                            checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            min_len=min_len)
        save_metrics(metrics, output_dir)
        free_memory()

    # few shot experiments.
    # Tune n-to-1-to-n model using few data from spanish-english and russian-english datasets
    for language in languages[2:]:
        for data_size, validate_interval in zip([10, 100, 1000, 10000],
                                                ["5", "3", "1", "1"]):
            checkpoint_dir = "{}/wikilingua_{}/{}-en_XX".format(output_dir, data_size, language)
            train_summarization_model(data_dir="wikilingua_{}".format(data_size),
                                      lang_pairs="{}-en_XX".format(language),
                                      checkpoint="{}/n-to-1-to-n/checkpoint_best.pt".format(output_dir),
                                      save_dir=checkpoint_dir,
                                      max_update="20000",
                                      validate_interval=validate_interval,
                                      validate_interval_updates="0",
                                      encoder_drop_residual=encoder_drop_residual,
                                      num_workers="1")
            free_memory()
            metrics["{}-n-to-1-to-n_tuned_{}".format(language, data_size)] = \
                generate_and_evaluate_summaries(directory="wikilingua",
                                                source_language=language,
                                                target_language="en_XX",
                                                lang_pairs="{}-en_XX".format(language),
                                                checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                                lenpen=lenpen,
                                                min_len=min_len)
            shutil.rmtree(checkpoint_dir)
            save_metrics(metrics, output_dir)
            free_memory()

    # tune n-to-1-to-n model using complete data from spanish-english and russian-english datasets
    for language in languages[2:]:
        checkpoint_dir = "{}/wikilingua_all/{}-en_XX".format(output_dir, language)
        train_summarization_model(data_dir="wikilingua",
                                  lang_pairs="{}-en_XX".format(language),
                                  checkpoint="{}/n-to-1-to-n/checkpoint_best.pt".format(output_dir),
                                  save_dir=checkpoint_dir,
                                  encoder_drop_residual=encoder_drop_residual)
        free_memory()
        metrics["{}-n-to-1-to-n_tuned_all".format(language)] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language,
                                            target_language="en_XX",
                                            lang_pairs="{}-en_XX".format(language),
                                            checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            min_len=min_len)
        shutil.rmtree(checkpoint_dir)
        save_metrics(metrics, output_dir)
        free_memory()


if __name__ == "__main__":
    run_wikilingua_experiments()
