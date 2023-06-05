import shutil
from pathlib import Path

from summarization_scripts.generate_summaries import generate_and_evaluate_summaries
from summarization_scripts.train_summarization import train_summarization_model
from summarization_scripts.utils import free_memory, save_metrics

languages = ["en_XX", "es_XX", "ru_RU", "gu_IN"]
lenpen = "0.6"
rouge_scorer = "multilingual"


def calculate_xlsum_baseline(output_dir=""):
    shutil.copyfile("baselines/xlsum_benchmark.csv", "{}/metrics.csv".format(output_dir))
    metrics = dict()

    # train baseline model and evaluate using english, spanish, russian and gujarati
    checkpoint_dir = "{}/baseline".format(output_dir)
    train_summarization_model(data_dir="xlsum",
                              lang_pairs=",".join(["{}-{}".format(language, language) for language in languages]),
                              save_dir=checkpoint_dir)
    free_memory()
    for language in languages:
        metrics["{}_baseline".format(language)] = \
            generate_and_evaluate_summaries(directory="xlsum",
                                            source_language=language,
                                            target_language=language,
                                            lang_pairs="{}-{}".format(language, language),
                                            checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            rouge_scorer=rouge_scorer)
        save_metrics(metrics, output_dir)
        free_memory()
    shutil.rmtree(checkpoint_dir)


def run_xlsum_experiments(encoder_drop_residual=None, experiments_folder="", prefix="", freeze_encoder_layers="0"):
    output_dir = "{}/{}".format(experiments_folder, prefix)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile("{}/metrics.csv".format(experiments_folder),
                    "{}/metrics.csv".format(output_dir))
    metrics = dict()

    # train a model and evaluate it using only english data
    checkpoint_dir = "{}/xlsum/en_XX".format(output_dir)
    train_summarization_model(data_dir="xlsum",
                              lang_pairs="en_XX-en_XX",
                              save_dir=checkpoint_dir,
                              encoder_drop_residual=encoder_drop_residual,
                              freeze_encoder_layers=freeze_encoder_layers)
    free_memory()
    metrics["en_XX"] = generate_and_evaluate_summaries(directory="xlsum",
                                                       source_language="en_XX",
                                                       target_language="en_XX",
                                                       lang_pairs="en_XX-en_XX",
                                                       checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                                       lenpen=lenpen,
                                                       rouge_scorer=rouge_scorer)
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
    # Do it only once at the beginning and use these results as a reference in other runs.
    if freeze_encoder_layers == "0" and not encoder_drop_residual:
        translation_metrics = dict()
        for translation_language in ["es", "ru", "gu"]:
            result = generate_and_evaluate_summaries(directory="xlsum_{}_en".format(translation_language),
                                                     source_language="en_XX",
                                                     target_language="en_XX",
                                                     lang_pairs="en_XX-en_XX",
                                                     checkpoint="{}/xlsum/en_XX/checkpoint_best.pt".format(output_dir),
                                                     lenpen=lenpen,
                                                     translate_to_lang=translation_language,
                                                     rouge_scorer=rouge_scorer)
            metrics["{}_translated".format(translation_language)] = result
            translation_metrics["{}_translated".format(translation_language)] = result
            save_metrics(metrics, output_dir)
            save_metrics(translation_metrics, experiments_folder)
            free_memory()

    # few shot experiments. Tune english model using few data from spanish, russian and gujarati datasets
    for language in languages[1:]:
        for data_size, max_epoch in zip([10, 100, 1000, 10000], ["12", "6", "4", "2"]):
            if language == "gu_IN" and data_size == 10000:
                continue
            checkpoint_dir = "{}/xlsum_{}/{}".format(output_dir, data_size, language)
            train_summarization_model(data_dir="xlsum_{}".format(data_size),
                                      lang_pairs="{}-{}".format(language, language),
                                      checkpoint="{}/xlsum/en_XX/checkpoint_best.pt".format(output_dir),
                                      save_dir=checkpoint_dir,
                                      encoder_drop_residual=encoder_drop_residual,
                                      num_workers="1",
                                      validate=False,
                                      max_epoch=max_epoch)
            free_memory()
            metrics["{}_tuned_{}".format(language, data_size)] = \
                generate_and_evaluate_summaries(directory="xlsum",
                                                source_language=language,
                                                target_language=language,
                                                lang_pairs="{}-{}".format(language, language),
                                                checkpoint="{}/checkpoint_last.pt".format(checkpoint_dir),
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
    shutil.rmtree("{}/xlsum/en_XX".format(output_dir))

    # train using english, spanish and russian together and evaluate with all 4 languages
    checkpoint_dir = "{}/multiEnEsRu".format(output_dir)
    train_summarization_model(data_dir="xlsum",
                              lang_pairs=",".join(["{}-{}".format(language, language) for language in languages[:-1]]),
                              save_dir=checkpoint_dir,
                              encoder_drop_residual=encoder_drop_residual,
                              freeze_encoder_layers=freeze_encoder_layers)
    free_memory()
    for language in languages:
        metrics["{}_multiEnEsRu".format(language)] = \
            generate_and_evaluate_summaries(directory="xlsum",
                                            source_language=language,
                                            target_language=language,
                                            lang_pairs="{}-{}".format(language, language),
                                            checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            rouge_scorer=rouge_scorer)
        save_metrics(metrics, output_dir)
        free_memory()
    # few shot experiments. Tune multilingual model using few data from gujarati dataset
    for data_size, max_epoch in zip([10, 100, 1000], ["12", "6", "4"]):
        checkpoint_dir = "{}/multiEnEsRu_gujarati_{}".format(output_dir, data_size)
        train_summarization_model(data_dir="xlsum_{}".format(data_size),
                                  lang_pairs="gu_IN-gu_IN",
                                  checkpoint="{}/multiEnEsRu/checkpoint_best.pt".format(output_dir),
                                  save_dir=checkpoint_dir,
                                  encoder_drop_residual=encoder_drop_residual,
                                  num_workers="1",
                                  validate=False,
                                  max_epoch=max_epoch)
        metrics["gu_IN_multiEnEsRu_{}".format(data_size)] = \
            generate_and_evaluate_summaries(directory="xlsum",
                                            source_language="gu_IN",
                                            target_language="gu_IN",
                                            lang_pairs="gu_IN-gu_IN",
                                            checkpoint="{}/checkpoint_last.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            rouge_scorer=rouge_scorer)
        save_metrics(metrics, output_dir)
        shutil.rmtree(checkpoint_dir)
        free_memory()

    # tune multilingual model and evaluate it using gujarati dataset
    checkpoint_dir = "{}/multiEnEsRu_gujarati".format(output_dir)
    train_summarization_model(data_dir="xlsum",
                              lang_pairs="gu_IN-gu_IN",
                              checkpoint="{}/multiEnEsRu/checkpoint_best.pt".format(output_dir),
                              save_dir=checkpoint_dir,
                              encoder_drop_residual=encoder_drop_residual)
    free_memory()
    metrics["gu_IN_multiEnEsRu_all"] = \
        generate_and_evaluate_summaries(directory="xlsum",
                                        source_language="gu_IN",
                                        target_language="gu_IN",
                                        lang_pairs="gu_IN-gu_IN",
                                        checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                        lenpen=lenpen,
                                        rouge_scorer=rouge_scorer)
    save_metrics(metrics, output_dir)
    shutil.rmtree(checkpoint_dir)
    free_memory()

    # Tune multilingual model using adversarial loss
    checkpoint_dir = "{}/multiEnEsRu_with_classifier".format(output_dir)
    train_summarization_model(data_dir="xlsum",
                              lang_pairs=",".join(["{}-{}".format(language, language) for language in languages[:-1]]),
                              checkpoint="{}/multiEnEsRu/checkpoint_best.pt".format(output_dir),
                              save_dir=checkpoint_dir,
                              use_adversarial_loss=True,
                              max_update="60000",
                              validate=False,
                              encoder_drop_residual=encoder_drop_residual,
                              freeze_encoder_layers=freeze_encoder_layers,
                              append_src_tok=False,
                              sampling_temperature="30")
    free_memory()
    for language in languages:
        metrics["{}_multiEnEsRu_adv".format(language)] = \
            generate_and_evaluate_summaries(directory="xlsum",
                                            source_language=language,
                                            target_language=language,
                                            lang_pairs="{}-{}".format(language, language),
                                            checkpoint="{}/checkpoint_last.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            rouge_scorer=rouge_scorer,
                                            append_src_tok=False)
        save_metrics(metrics, output_dir)
        free_memory()
    # few shot experiments. Tune multilingual model using few data from gujarati dataset
    for data_size, max_epoch in zip([10, 100, 1000], ["12", "6", "4"]):
        checkpoint_dir = "{}/multiEnEsRu_gujarati_{}".format(output_dir, data_size)
        train_summarization_model(data_dir="xlsum_{}".format(data_size),
                                  lang_pairs="gu_IN-gu_IN",
                                  checkpoint="{}/multiEnEsRu_with_classifier/checkpoint_last.pt".format(output_dir),
                                  save_dir=checkpoint_dir,
                                  encoder_drop_residual=encoder_drop_residual,
                                  num_workers="1",
                                  validate=False,
                                  max_epoch=max_epoch,
                                  append_src_tok=False)
        metrics["gu_IN_multiEnEsRu_adv_{}".format(data_size)] = \
            generate_and_evaluate_summaries(directory="xlsum",
                                            source_language="gu_IN",
                                            target_language="gu_IN",
                                            lang_pairs="gu_IN-gu_IN",
                                            checkpoint="{}/checkpoint_last.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            rouge_scorer=rouge_scorer,
                                            append_src_tok=False)
        save_metrics(metrics, output_dir)
        shutil.rmtree(checkpoint_dir)
        free_memory()

    # tune multilingual model and evaluate it using gujarati dataset
    checkpoint_dir = "{}/multiEnEsRu_gujarati".format(output_dir)
    train_summarization_model(data_dir="xlsum",
                              lang_pairs="gu_IN-gu_IN",
                              checkpoint="{}/multiEnEsRu_with_classifier/checkpoint_last.pt".format(output_dir),
                              save_dir=checkpoint_dir,
                              encoder_drop_residual=encoder_drop_residual,
                              append_src_tok=False)
    free_memory()
    metrics["gu_IN_multiEnEsRu_adv_all"] = \
        generate_and_evaluate_summaries(directory="xlsum",
                                        source_language="gu_IN",
                                        target_language="gu_IN",
                                        lang_pairs="gu_IN-gu_IN",
                                        checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                        lenpen=lenpen,
                                        rouge_scorer=rouge_scorer,
                                        append_src_tok=False)
    save_metrics(metrics, output_dir)
    shutil.rmtree(checkpoint_dir)
    free_memory()

    shutil.rmtree("{}/multiEnEsRu".format(output_dir))
    shutil.rmtree("{}/multiEnEsRu_with_classifier".format(output_dir))


if __name__ == "__main__":
    run_xlsum_experiments()
