import shutil

from summarization_scripts.generate_summaries import generate_and_evaluate_summaries
from summarization_scripts.train_summarization import train_summarization_model
from summarization_scripts.utils import free_memory, save_metrics

languages = ["en_XX", "es_XX", "ru_RU", "gu_IN"]
lenpen = "0.6"
rouge_scorer = "multilingual"


def run_xlsum_experiments(encoder_drop_residual=None, prefix="", freeze_encoder_layers="0"):
    metrics = dict()
    output_dir = "xlsum_results/{}".format(prefix)
    # every language separately
    for language in languages:
        checkpoint_dir = "{}/xlsum/{}".format(output_dir, language)
        train_summarization_model(data_dir="xlsum",
                                  lang_pairs="{}-{}".format(language, language),
                                  save_dir=checkpoint_dir,
                                  encoder_drop_residual=encoder_drop_residual,
                                  freeze_encoder_layers=freeze_encoder_layers)
        free_memory()
        metrics[language] = generate_and_evaluate_summaries(directory="xlsum",
                                                            source_language=language,
                                                            target_language=language,
                                                            lang_pairs="{}-{}".format(language, language),
                                                            checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                                            lenpen=lenpen,
                                                            rouge_scorer=rouge_scorer)
        if language != "en_XX":
            shutil.rmtree(checkpoint_dir)
        save_metrics(metrics, output_dir)
        free_memory()

    # zero shot. Evaluate spanish, russian and gujarati datasets using english model
    for language in languages[1:]:
        metrics["{}_zero".format(language)] = \
            generate_and_evaluate_summaries(directory="xlsum",
                                            source_language=language,
                                            target_language=language,
                                            lang_pairs="{}-{}".format(language, language),
                                            checkpoint="{}/xlsum/en_XX/checkpoint_best.pt".format(output_dir),
                                            lenpen=lenpen,
                                            rouge_scorer=rouge_scorer)
        save_metrics(metrics, output_dir)
        free_memory()

    # input is translated from spanish, russian and gujarati into english. Create summaries using english model.
    # Translate summaries in english back into spanish, russian and gujarati
    # and evaluate using original data in these languages.
    for translation_language in ["es", "ru", "gu"]:
        metrics["{}_translated".format(translation_language)] = \
            generate_and_evaluate_summaries(directory="xlsum_{}_en".format(translation_language),
                                            source_language="en_XX",
                                            target_language="en_XX",
                                            lang_pairs="en_XX-en_XX",
                                            checkpoint="{}/xlsum/en_XX/checkpoint_best.pt".format(output_dir),
                                            lenpen=lenpen,
                                            translate_to_lang=translation_language,
                                            rouge_scorer=rouge_scorer)
        save_metrics(metrics, output_dir)
        free_memory()

    # few shot experiments. Tune english model using few data from spanish, russian and gujarati datasets
    for language in languages[1:]:
        for data_size, validate_interval in zip([10, 100, 1000, 10000],
                                                ["5", "2", "1", "1"]):
            if language == "gu_IN" and data_size == 10000:
                break
            checkpoint_dir = "{}/xlsum_{}/{}".format(output_dir, data_size, language)
            train_summarization_model(data_dir="xlsum_{}".format(data_size),
                                      lang_pairs="{}-{}".format(language, language),
                                      checkpoint="{}/xlsum/en_XX/checkpoint_best.pt".format(output_dir),
                                      save_dir=checkpoint_dir,
                                      max_update="20000",
                                      validate_interval=validate_interval,
                                      validate_interval_updates="0",
                                      encoder_drop_residual=encoder_drop_residual,
                                      num_workers="1")
            free_memory()
            metrics["{}_tuned_{}".format(language, data_size)] = \
                generate_and_evaluate_summaries(directory="xlsum",
                                                source_language=language,
                                                target_language=language,
                                                lang_pairs="{}-{}".format(language, language),
                                                checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                                lenpen=lenpen,
                                                rouge_scorer=rouge_scorer)
            shutil.rmtree(checkpoint_dir)
            save_metrics(metrics, output_dir)
            free_memory()

    # tune english model using complete data from spanish, russian and gujarati datasets
    for language in languages[1:]:
        checkpoint_dir = "{}/xlsum_all/{}".format(output_dir, language)
        train_summarization_model(data_dir="xlsum",
                                  lang_pairs="{}-{}".format(language, language),
                                  checkpoint="{}/xlsum/en_XX/checkpoint_best.pt".format(output_dir),
                                  save_dir=checkpoint_dir,
                                  encoder_drop_residual=encoder_drop_residual)
        free_memory()
        metrics["{}_tuned_all".format(language)] = \
            generate_and_evaluate_summaries(directory="xlsum",
                                            source_language=language,
                                            target_language=language,
                                            lang_pairs="{}-{}".format(language, language),
                                            checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            rouge_scorer=rouge_scorer)
        shutil.rmtree(checkpoint_dir)
        save_metrics(metrics, output_dir)
        free_memory()

    # train using english, spanish and russian together and evaluate with all 4 languages
    checkpoint_dir = "{}/multilingual".format(output_dir)
    train_summarization_model(data_dir="xlsum",
                              lang_pairs=",".join(["{}-{}".format(language, language) for language in languages[:-1]]),
                              save_dir=checkpoint_dir,
                              encoder_drop_residual=encoder_drop_residual,
                              freeze_encoder_layers=freeze_encoder_layers)
    free_memory()
    for language in languages:
        metrics["{}_multilingual".format(language)] = \
            generate_and_evaluate_summaries(directory="xlsum",
                                            source_language=language,
                                            target_language=language,
                                            lang_pairs="{}-{}".format(language, language),
                                            checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            rouge_scorer=rouge_scorer)
        save_metrics(metrics, output_dir)
        free_memory()

    # tune multilingual model and evaluate it using gujarati dataset
    checkpoint_dir = "{}/multilingual_tuned_gujarati".format(output_dir)
    train_summarization_model(data_dir="xlsum",
                              lang_pairs="gu_IN-gu_IN",
                              checkpoint="{}/multilingual/checkpoint_best.pt".format(output_dir),
                              save_dir=checkpoint_dir,
                              encoder_drop_residual=encoder_drop_residual)
    free_memory()
    metrics["gu_IN_multilingual_tuned_gujarati"] = \
        generate_and_evaluate_summaries(directory="xlsum",
                                        source_language="gu_IN",
                                        target_language="gu_IN",
                                        lang_pairs="gu_IN-gu_IN",
                                        checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                        lenpen=lenpen,
                                        rouge_scorer=rouge_scorer)
    save_metrics(metrics, output_dir)
    free_memory()

    shutil.rmtree(checkpoint_dir)
    shutil.rmtree("{}/xlsum/en_XX".format(output_dir))
    shutil.rmtree("{}/multilingual".format(output_dir))


if __name__ == "__main__":
    run_xlsum_experiments()
