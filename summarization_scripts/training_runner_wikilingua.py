import shutil
from pathlib import Path

from summarization_scripts.generate_summaries import generate_and_evaluate_summaries
from summarization_scripts.train_summarization import train_summarization_model
from summarization_scripts.utils import free_memory, save_metrics

languages = ["en_XX", "es_XX", "ru_RU", "tr_TR"]
language_pairs_evaluation = [("es_XX", "en_XX"), ("ru_RU", "en_XX"), ("tr_TR", "en_XX"),
                             ("es_XX", "ru_RU"), ("en_XX", "tr_TR"), ("tr_TR", "tr_TR")]
lenpen = "1.0"
min_len = "10"


def calculate_wikilingua_supervised(output_dir=""):
    shutil.copyfile("benchmark/wiki_benchmark.csv", "{}/metrics.csv".format(output_dir))
    metrics = dict()

    # all supervised cases together as baseline
    checkpoint_dir = "{}/supervised".format(output_dir)
    train_summarization_model(data_dir="wikilingua",
                              lang_pairs=",".join(["{}-{}".format(language_pair[0], language_pair[1])
                                                   for language_pair in language_pairs_evaluation]),
                              save_dir=checkpoint_dir)
    free_memory()
    for language_pair in language_pairs_evaluation:
        metrics["{}_{}_supervised".format(language_pair[0], language_pair[1])] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language_pair[0],
                                            target_language=language_pair[1],
                                            lang_pairs="{}-{}".format(language_pair[0], language_pair[1]),
                                            checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            min_len=min_len)
        save_metrics(metrics, output_dir)
        free_memory()
    shutil.rmtree(checkpoint_dir)


def run_wikilingua_experiments(encoder_drop_residual=None, experiments_folder="", prefix="",
                               freeze_encoder_layers=False, use_encoder_output_adapter=False,
                               use_decoder_adapter=False, use_encoder_adapter="no",
                               adversarial_kldivloss=False, adversarial_nllloss=False,
                               masked_labels=False, label_smoothing="0.0",
                               freeze_decoder_layers=False, freeze_elements="everything",
                               add_translated_results=False, freeze_embeddings=True, few_shot=False,
                               english_model=True, tune_after_adversarial=False):
    if use_encoder_output_adapter or use_decoder_adapter or (use_encoder_adapter != "no"):
        language_pairs = language_pairs_evaluation[:4]
    else:
        language_pairs = language_pairs_evaluation
    output_dir = "{}/{}".format(experiments_folder, prefix)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile("{}/metrics.csv".format(experiments_folder),
                    "{}/metrics.csv".format(output_dir))
    metrics = dict()

    # train using only monolingual english data
    if english_model:
        monolingual_checkpoint_dir = "{}/monolingual_en".format(output_dir)
        train_summarization_model(data_dir="wikilingua",
                                  lang_pairs="en_XX-en_XX",
                                  save_dir=monolingual_checkpoint_dir,
                                  encoder_drop_residual=encoder_drop_residual,
                                  freeze_encoder_layers=freeze_encoder_layers,
                                  freeze_decoder_layers=freeze_decoder_layers,
                                  freeze_embeddings=freeze_embeddings,
                                  freeze_elements=freeze_elements,
                                  use_encoder_output_adapter=use_encoder_output_adapter,
                                  use_decoder_adapter=use_decoder_adapter,
                                  use_encoder_adapter=use_encoder_adapter,
                                  masked_labels=masked_labels,
                                  label_smoothing=label_smoothing)
        free_memory()
        # evaluate supervised cases
        for language_pair in language_pairs:
            metrics["{}_{}_mono_en".format(language_pair[0], language_pair[1])] = \
                generate_and_evaluate_summaries(directory="wikilingua",
                                                source_language=language_pair[0],
                                                target_language=language_pair[1],
                                                lang_pairs="{}-{}".format(language_pair[0], language_pair[1]),
                                                checkpoint="{}/checkpoint_best.pt".format(monolingual_checkpoint_dir),
                                                lenpen=lenpen,
                                                min_len=min_len,
                                                use_encoder_output_adapter=use_encoder_output_adapter,
                                                use_decoder_adapter=use_decoder_adapter,
                                                use_encoder_adapter=use_encoder_adapter)
            save_metrics(metrics, output_dir)
            free_memory()

        # input is translated from spanish, russian and gujarati into english. Create summaries using english model.
        # Translate summaries in english back into spanish, russian and gujarati
        # and evaluate using original data in these languages.
        # Do it only once at the beginning and use these results as a reference in other runs.
        if add_translated_results:
            translation_metrics = dict()
            for source_language in ["es", "ru", "tr"]:
                result = generate_and_evaluate_summaries(directory="wikilingua_{}_en".format(source_language),
                                                         source_language="en_XX",
                                                         target_language="en_XX",
                                                         lang_pairs="en_XX-en_XX",
                                                         checkpoint="{}/checkpoint_best.pt".format(
                                                             monolingual_checkpoint_dir),
                                                         lenpen=lenpen)
                metrics["{}_en_translated_nllb".format(source_language)] = result
                translation_metrics["{}_en_translated_nllb".format(source_language)] = result
                save_metrics(metrics, output_dir)
                save_metrics(translation_metrics, experiments_folder)
                free_memory()
            for source_language, translation_language in zip(["es", "en", "tr"], ["ru", "tr", "tr"]):
                result = generate_and_evaluate_summaries(
                    directory="wikilingua_{}_{}".format(source_language, translation_language),
                    source_language="en_XX",
                    target_language="en_XX",
                    lang_pairs="en_XX-en_XX",
                    checkpoint="{}/checkpoint_best.pt".format(monolingual_checkpoint_dir),
                    lenpen=lenpen,
                    translate_to_lang=translation_language)
                metrics["{}_{}_translated_nllb".format(source_language, translation_language)] = result
                translation_metrics["{}_{}_translated_nllb".format(source_language, translation_language)] = result
                save_metrics(metrics, output_dir)
                save_metrics(translation_metrics, experiments_folder)
                free_memory()

        # few shot experiments.
        # Tune monolingual model using few supervised data
        if few_shot:
            for language_pair in language_pairs:
                for data_size, max_epoch in zip([10, 100, 1000, 10000], ["12", "6", "4", "2"]):
                    if (language_pair[0] == "tr_TR" or language_pair[1] == "tr_TR") and data_size == 10000:
                        continue
                    checkpoint_dir = "{}/wikilingua_{}/{}-{}".format(output_dir, data_size, language_pair[0],
                                                                     language_pair[1])
                    train_summarization_model(data_dir="wikilingua_{}".format(data_size),
                                              lang_pairs="{}-{}".format(language_pair[0], language_pair[1]),
                                              checkpoint="{}/checkpoint_best.pt".format(monolingual_checkpoint_dir),
                                              save_dir=checkpoint_dir,
                                              encoder_drop_residual=encoder_drop_residual,
                                              freeze_embeddings=freeze_embeddings,
                                              num_workers="1",
                                              validate=False,
                                              max_epoch=max_epoch,
                                              use_encoder_output_adapter=use_encoder_output_adapter,
                                              use_decoder_adapter=use_decoder_adapter,
                                              use_encoder_adapter=use_encoder_adapter,
                                              masked_labels=masked_labels,
                                              label_smoothing=label_smoothing)
                    free_memory()
                    metrics["{}_{}_mono_en_{}".format(language_pair[0], language_pair[1], data_size)] = \
                        generate_and_evaluate_summaries(directory="wikilingua",
                                                        source_language=language_pair[0],
                                                        target_language=language_pair[1],
                                                        lang_pairs="{}-{}".format(language_pair[0], language_pair[1]),
                                                        checkpoint="{}/checkpoint_last.pt".format(checkpoint_dir),
                                                        lenpen=lenpen,
                                                        min_len=min_len,
                                                        use_encoder_output_adapter=use_encoder_output_adapter,
                                                        use_decoder_adapter=use_decoder_adapter,
                                                        use_encoder_adapter=use_encoder_adapter)
                    shutil.rmtree(checkpoint_dir)
                    save_metrics(metrics, output_dir)
                    free_memory()
        shutil.rmtree(monolingual_checkpoint_dir)

    # train using english, spanish, russian data together, but monolingual data
    monolingual_checkpoint_dir = "{}/monolingual_multi".format(output_dir)
    train_summarization_model(data_dir="wikilingua",
                              lang_pairs=",".join(
                                  ["{}-{}".format(language, language) for language in languages[:3]]),
                              save_dir=monolingual_checkpoint_dir,
                              sampling_temperature="10.0",
                              encoder_drop_residual=encoder_drop_residual,
                              freeze_encoder_layers=freeze_encoder_layers,
                              freeze_decoder_layers=freeze_decoder_layers,
                              freeze_embeddings=freeze_embeddings,
                              freeze_elements=freeze_elements,
                              use_encoder_output_adapter=use_encoder_output_adapter,
                              use_decoder_adapter=use_decoder_adapter,
                              use_encoder_adapter=use_encoder_adapter,
                              masked_labels=masked_labels,
                              label_smoothing=label_smoothing)
    free_memory()
    # evaluate supervised cases
    for language_pair in language_pairs:
        metrics["{}_{}_mono".format(language_pair[0], language_pair[1])] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language_pair[0],
                                            target_language=language_pair[1],
                                            lang_pairs="{}-{}".format(language_pair[0], language_pair[1]),
                                            checkpoint="{}/checkpoint_best.pt".format(monolingual_checkpoint_dir),
                                            lenpen=lenpen,
                                            min_len=min_len,
                                            use_encoder_output_adapter=use_encoder_output_adapter,
                                            use_decoder_adapter=use_decoder_adapter,
                                            use_encoder_adapter=use_encoder_adapter)
        save_metrics(metrics, output_dir)
        free_memory()

    # few shot experiments.
    # Tune monolingual model using few supervised data
    if few_shot:
        for language_pair in language_pairs:
            for data_size, max_epoch in zip([10, 100, 1000, 10000], ["12", "6", "4", "2"]):
                if (language_pair[0] == "tr_TR" or language_pair[1] == "tr_TR") and data_size == 10000:
                    continue
                checkpoint_dir = "{}/wikilingua_{}/{}-{}".format(output_dir, data_size, language_pair[0],
                                                                 language_pair[1])
                train_summarization_model(data_dir="wikilingua_{}".format(data_size),
                                          lang_pairs="{}-{}".format(language_pair[0], language_pair[1]),
                                          checkpoint="{}/checkpoint_best.pt".format(monolingual_checkpoint_dir),
                                          save_dir=checkpoint_dir,
                                          encoder_drop_residual=encoder_drop_residual,
                                          freeze_embeddings=freeze_embeddings,
                                          num_workers="1",
                                          validate=False,
                                          max_epoch=max_epoch,
                                          use_encoder_output_adapter=use_encoder_output_adapter,
                                          use_decoder_adapter=use_decoder_adapter,
                                          use_encoder_adapter=use_encoder_adapter,
                                          masked_labels=masked_labels,
                                          label_smoothing=label_smoothing)
                free_memory()
                metrics["{}_{}_mono_{}".format(language_pair[0], language_pair[1], data_size)] = \
                    generate_and_evaluate_summaries(directory="wikilingua",
                                                    source_language=language_pair[0],
                                                    target_language=language_pair[1],
                                                    lang_pairs="{}-{}".format(language_pair[0], language_pair[1]),
                                                    checkpoint="{}/checkpoint_last.pt".format(checkpoint_dir),
                                                    lenpen=lenpen,
                                                    min_len=min_len,
                                                    use_encoder_output_adapter=use_encoder_output_adapter,
                                                    use_decoder_adapter=use_decoder_adapter,
                                                    use_encoder_adapter=use_encoder_adapter)
                shutil.rmtree(checkpoint_dir)
                save_metrics(metrics, output_dir)
                free_memory()

    # Tune multilingual model using adversarial loss using nll
    if adversarial_nllloss:
        monolingual_adv_checkpoint_dir = "{}/monolingual_with_classifier_nll".format(output_dir)
        train_summarization_model(data_dir="wikilingua",
                                  lang_pairs=",".join(
                                      ["{}-{}".format(language, language) for language in languages[:3]]),
                                  checkpoint="{}/checkpoint_best.pt".format(monolingual_checkpoint_dir),
                                  save_dir=monolingual_adv_checkpoint_dir,
                                  sampling_temperature="10.0",
                                  encoder_drop_residual=encoder_drop_residual,
                                  freeze_decoder_layers=freeze_decoder_layers,
                                  freeze_embeddings=freeze_embeddings,
                                  freeze_elements=freeze_elements,
                                  use_adversarial_loss=True,
                                  max_update="100000",
                                  validate=False,
                                  use_encoder_output_adapter=use_encoder_output_adapter,
                                  use_decoder_adapter=use_decoder_adapter,
                                  use_encoder_adapter=use_encoder_adapter,
                                  masked_labels=masked_labels,
                                  label_smoothing=label_smoothing,
                                  append_src_tok=False,
                                  use_kldivloss=False)

        # evaluate supervised cases
        for language_pair in language_pairs:
            metrics["{}_{}_mono_adv_nll".format(language_pair[0], language_pair[1])] = \
                generate_and_evaluate_summaries(directory="wikilingua",
                                                source_language=language_pair[0],
                                                target_language=language_pair[1],
                                                lang_pairs="{}-{}".format(language_pair[0], language_pair[1]),
                                                checkpoint="{}/checkpoint_last.pt".format(
                                                    monolingual_adv_checkpoint_dir),
                                                lenpen=lenpen,
                                                min_len=min_len,
                                                use_encoder_output_adapter=use_encoder_output_adapter,
                                                use_decoder_adapter=use_decoder_adapter,
                                                use_encoder_adapter=use_encoder_adapter,
                                                append_src_tok=False)
            save_metrics(metrics, output_dir)
            free_memory()

        if tune_after_adversarial and not freeze_decoder_layers:
            tuned_adv_checkpoint_dir = "{}/monolingual_with_classifier_nll_tuned".format(output_dir)
            train_summarization_model(data_dir="wikilingua",
                                      lang_pairs=",".join(
                                          ["{}-{}".format(language, language) for language in languages[:3]]),
                                      checkpoint="{}/checkpoint_last.pt".format(monolingual_adv_checkpoint_dir),
                                      save_dir=tuned_adv_checkpoint_dir,
                                      sampling_temperature="10.0",
                                      encoder_drop_residual=encoder_drop_residual,
                                      freeze_embeddings=freeze_embeddings,
                                      freeze_encoder_layers=True,
                                      freeze_elements="everything",
                                      max_update="30000",
                                      use_encoder_output_adapter=use_encoder_output_adapter,
                                      use_decoder_adapter=use_decoder_adapter,
                                      use_encoder_adapter=use_encoder_adapter,
                                      masked_labels=masked_labels,
                                      label_smoothing=label_smoothing,
                                      append_src_tok=False)

            # evaluate supervised cases
            for language_pair in language_pairs:
                metrics["{}_{}_mono_adv_nll_tuned".format(language_pair[0], language_pair[1])] = \
                    generate_and_evaluate_summaries(directory="wikilingua",
                                                    source_language=language_pair[0],
                                                    target_language=language_pair[1],
                                                    lang_pairs="{}-{}".format(language_pair[0], language_pair[1]),
                                                    checkpoint="{}/checkpoint_best.pt".format(tuned_adv_checkpoint_dir),
                                                    lenpen=lenpen,
                                                    min_len=min_len,
                                                    use_encoder_output_adapter=use_encoder_output_adapter,
                                                    use_decoder_adapter=use_decoder_adapter,
                                                    use_encoder_adapter=use_encoder_adapter,
                                                    append_src_tok=False)
                save_metrics(metrics, output_dir)
                free_memory()
            shutil.rmtree(tuned_adv_checkpoint_dir)
        shutil.rmtree(monolingual_adv_checkpoint_dir)

    # Tune multilingual model using adversarial loss using kldivloss
    if adversarial_kldivloss:
        monolingual_adv_checkpoint_dir = "{}/monolingual_with_classifier_kldivloss".format(output_dir)
        train_summarization_model(data_dir="wikilingua",
                                  lang_pairs=",".join(
                                      ["{}-{}".format(language, language) for language in languages[:3]]),
                                  checkpoint="{}/checkpoint_best.pt".format(monolingual_checkpoint_dir),
                                  save_dir=monolingual_adv_checkpoint_dir,
                                  sampling_temperature="10.0",
                                  encoder_drop_residual=encoder_drop_residual,
                                  freeze_decoder_layers=freeze_decoder_layers,
                                  freeze_embeddings=freeze_embeddings,
                                  freeze_elements=freeze_elements,
                                  use_adversarial_loss=True,
                                  max_update="100000",
                                  validate=False,
                                  use_encoder_output_adapter=use_encoder_output_adapter,
                                  use_decoder_adapter=use_decoder_adapter,
                                  use_encoder_adapter=use_encoder_adapter,
                                  masked_labels=masked_labels,
                                  label_smoothing=label_smoothing,
                                  append_src_tok=False,
                                  use_kldivloss=True)

        # evaluate supervised cases
        for language_pair in language_pairs:
            metrics["{}_{}_mono_adv_kldivloss".format(language_pair[0], language_pair[1])] = \
                generate_and_evaluate_summaries(directory="wikilingua",
                                                source_language=language_pair[0],
                                                target_language=language_pair[1],
                                                lang_pairs="{}-{}".format(language_pair[0], language_pair[1]),
                                                checkpoint="{}/checkpoint_last.pt".format(
                                                    monolingual_adv_checkpoint_dir),
                                                lenpen=lenpen,
                                                min_len=min_len,
                                                use_encoder_output_adapter=use_encoder_output_adapter,
                                                use_decoder_adapter=use_decoder_adapter,
                                                use_encoder_adapter=use_encoder_adapter,
                                                append_src_tok=False)
            save_metrics(metrics, output_dir)
            free_memory()

        if tune_after_adversarial and not freeze_decoder_layers:
            tuned_adv_checkpoint_dir = "{}/monolingual_with_classifier_kldivloss_tuned".format(output_dir)
            train_summarization_model(data_dir="wikilingua",
                                      lang_pairs=",".join(
                                          ["{}-{}".format(language, language) for language in languages[:3]]),
                                      checkpoint="{}/checkpoint_last.pt".format(monolingual_adv_checkpoint_dir),
                                      save_dir=tuned_adv_checkpoint_dir,
                                      sampling_temperature="10.0",
                                      encoder_drop_residual=encoder_drop_residual,
                                      freeze_embeddings=freeze_embeddings,
                                      freeze_encoder_layers=True,
                                      freeze_elements="everything",
                                      max_update="30000",
                                      use_encoder_output_adapter=use_encoder_output_adapter,
                                      use_decoder_adapter=use_decoder_adapter,
                                      use_encoder_adapter=use_encoder_adapter,
                                      masked_labels=masked_labels,
                                      label_smoothing=label_smoothing,
                                      append_src_tok=False)

            # evaluate supervised cases
            for language_pair in language_pairs:
                metrics["{}_{}_mono_adv_kldivloss_tuned".format(language_pair[0], language_pair[1])] = \
                    generate_and_evaluate_summaries(directory="wikilingua",
                                                    source_language=language_pair[0],
                                                    target_language=language_pair[1],
                                                    lang_pairs="{}-{}".format(language_pair[0], language_pair[1]),
                                                    checkpoint="{}/checkpoint_best.pt".format(tuned_adv_checkpoint_dir),
                                                    lenpen=lenpen,
                                                    min_len=min_len,
                                                    use_encoder_output_adapter=use_encoder_output_adapter,
                                                    use_decoder_adapter=use_decoder_adapter,
                                                    use_encoder_adapter=use_encoder_adapter,
                                                    append_src_tok=False)
                save_metrics(metrics, output_dir)
                free_memory()
            shutil.rmtree(tuned_adv_checkpoint_dir)

        shutil.rmtree(monolingual_adv_checkpoint_dir)
    shutil.rmtree(monolingual_checkpoint_dir)


if __name__ == "__main__":
    run_wikilingua_experiments()
