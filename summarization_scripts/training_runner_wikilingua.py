import shutil
from pathlib import Path

from summarization_scripts.generate_summaries import generate_and_evaluate_summaries
from summarization_scripts.train_summarization import train_summarization_model
from summarization_scripts.utils import free_memory, save_metrics

languages = ["en_XX", "es_XX", "ru_RU", "tr_TR"]
lenpen = "1.0"
min_len = "10"


def calculate_wikilingua_baseline(output_dir=""):
    shutil.copyfile("baselines/wiki_benchmark.csv", "{}/metrics.csv".format(output_dir))
    metrics = dict()

    # three crosslingual cases (spanish-english, russian-english and turkish-english) together as baseline
    checkpoint_dir = "{}/baseline".format(output_dir)
    train_summarization_model(data_dir="wikilingua",
                              lang_pairs=",".join(["{}-en_XX".format(language) for language in languages[1:]]),
                              save_dir=checkpoint_dir,
                              label_smoothing="0.1")
    free_memory()
    for language in languages[1:]:
        metrics["{}_baseline".format(language)] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language,
                                            target_language="en_XX",
                                            lang_pairs="{}-en_XX".format(language),
                                            checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            min_len=min_len)
        save_metrics(metrics, output_dir)
        free_memory()
    shutil.rmtree(checkpoint_dir)


def run_wikilingua_experiments(encoder_drop_residual=None, experiments_folder="", prefix="", freeze_encoder_layers="0"):
    for use_encoder_output_adapter in [False, True]:
        prefix_lang_emb = \
            "{}/{}".format(prefix, "with_lang_adapter" if use_encoder_output_adapter else "no_lang_adapter")
        run_experiments(encoder_drop_residual=encoder_drop_residual,
                        experiments_folder=experiments_folder,
                        prefix=prefix_lang_emb,
                        freeze_encoder_layers=freeze_encoder_layers,
                        use_encoder_output_adapter=use_encoder_output_adapter)


def run_experiments(encoder_drop_residual=None, experiments_folder="", prefix="",
                    freeze_encoder_layers="0", use_encoder_output_adapter=False):
    output_dir = "{}/{}".format(experiments_folder, prefix)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile("{}/metrics.csv".format(experiments_folder),
                    "{}/metrics.csv".format(output_dir))
    metrics = dict()

    # english, spanish, russian together, but monolingual data
    monolingual_checkpoint_dir = "{}/monolingual".format(output_dir)
    train_summarization_model(data_dir="wikilingua",
                              lang_pairs=",".join(["{}-{}".format(language, language) for language in languages[:3]]),
                              save_dir=monolingual_checkpoint_dir,
                              encoder_drop_residual=encoder_drop_residual,
                              freeze_encoder_layers=freeze_encoder_layers,
                              use_encoder_output_adapter=use_encoder_output_adapter)
    free_memory()
    # evaluate crosslingual cases: from spanish, russian, turkish into english
    for language in languages[1:]:
        metrics["{}_mono".format(language)] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language,
                                            target_language="en_XX",
                                            lang_pairs="{}-en_XX".format(language),
                                            checkpoint="{}/checkpoint_best.pt".format(monolingual_checkpoint_dir),
                                            lenpen=lenpen,
                                            min_len=min_len,
                                            use_encoder_output_adapter=use_encoder_output_adapter)
        save_metrics(metrics, output_dir)
        free_memory()

    # few shot experiments.
    # Tune monolingual model using few data from spanish-english, russian-english and turkish-english datasets
    for language in languages[1:]:
        for data_size, max_epoch in zip([10, 100, 1000, 10000], ["12", "6", "4", "2"]):
            if language == "tr_TR" and data_size == 10000:
                continue
            checkpoint_dir = "{}/wikilingua_{}/{}-en_XX".format(output_dir, data_size, language)
            train_summarization_model(data_dir="wikilingua_{}".format(data_size),
                                      lang_pairs="{}-en_XX".format(language),
                                      checkpoint="{}/checkpoint_best.pt".format(monolingual_checkpoint_dir),
                                      save_dir=checkpoint_dir,
                                      encoder_drop_residual=encoder_drop_residual,
                                      num_workers="1",
                                      validate=False,
                                      max_epoch=max_epoch,
                                      use_encoder_output_adapter=use_encoder_output_adapter)
            free_memory()
            metrics["{}_mono_{}".format(language, data_size)] = \
                generate_and_evaluate_summaries(directory="wikilingua",
                                                source_language=language,
                                                target_language="en_XX",
                                                lang_pairs="{}-en_XX".format(language),
                                                checkpoint="{}/checkpoint_last.pt".format(checkpoint_dir),
                                                lenpen=lenpen,
                                                min_len=min_len,
                                                use_encoder_output_adapter=use_encoder_output_adapter)
            shutil.rmtree(checkpoint_dir)
            save_metrics(metrics, output_dir)
            free_memory()

    # tune monolingual model using complete data from spanish-english, russian-english and turkish-english datasets
    for language in languages[1:]:
        checkpoint_dir = "{}/wikilingua_all/{}-en_XX".format(output_dir, language)
        train_summarization_model(data_dir="wikilingua",
                                  lang_pairs="{}-en_XX".format(language),
                                  checkpoint="{}/checkpoint_best.pt".format(monolingual_checkpoint_dir),
                                  save_dir=checkpoint_dir,
                                  encoder_drop_residual=encoder_drop_residual,
                                  use_encoder_output_adapter=use_encoder_output_adapter)
        free_memory()
        metrics["{}_mono_All".format(language)] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language,
                                            target_language="en_XX",
                                            lang_pairs="{}-en_XX".format(language),
                                            checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            min_len=min_len,
                                            use_encoder_output_adapter=use_encoder_output_adapter)
        shutil.rmtree(checkpoint_dir)
        save_metrics(metrics, output_dir)
        free_memory()

    # Tune multilingual model using adversarial loss using nll
    monolingual_adv_checkpoint_dir = "{}/monolingual_with_classifier_nll".format(output_dir)
    train_summarization_model(data_dir="wikilingua",
                              lang_pairs=",".join(
                                  ["{}-{}".format(language, language) for language in languages[:3]]),
                              checkpoint="{}/checkpoint_best.pt".format(monolingual_checkpoint_dir),
                              save_dir=monolingual_adv_checkpoint_dir,
                              encoder_drop_residual=encoder_drop_residual,
                              freeze_encoder_layers=freeze_encoder_layers,
                              use_adversarial_loss=True,
                              max_update="30000",
                              validate=False,
                              use_encoder_output_adapter=use_encoder_output_adapter,
                              append_src_tok=False,
                              sampling_temperature="30",
                              use_kldivloss=False)

    # evaluate crosslingual cases: from spanish, russian, turkish into english
    for language in languages[1:]:
        metrics["{}_mono_adv_nll".format(language)] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language,
                                            target_language="en_XX",
                                            lang_pairs="{}-en_XX".format(language),
                                            checkpoint="{}/checkpoint_last.pt".format(monolingual_adv_checkpoint_dir),
                                            lenpen=lenpen,
                                            min_len=min_len,
                                            use_encoder_output_adapter=use_encoder_output_adapter,
                                            append_src_tok=False)
        save_metrics(metrics, output_dir)
        free_memory()
    shutil.rmtree(monolingual_adv_checkpoint_dir)

    # Tune multilingual model using adversarial loss using kldivloss
    monolingual_adv_checkpoint_dir = "{}/monolingual_with_classifier_kldivloss".format(output_dir)
    train_summarization_model(data_dir="wikilingua",
                              lang_pairs=",".join(
                                  ["{}-{}".format(language, language) for language in languages[:3]]),
                              checkpoint="{}/checkpoint_best.pt".format(monolingual_checkpoint_dir),
                              save_dir=monolingual_adv_checkpoint_dir,
                              encoder_drop_residual=encoder_drop_residual,
                              freeze_encoder_layers=freeze_encoder_layers,
                              use_adversarial_loss=True,
                              max_update="30000",
                              validate=False,
                              use_encoder_output_adapter=use_encoder_output_adapter,
                              append_src_tok=False,
                              sampling_temperature="30",
                              use_kldivloss=True)
    shutil.rmtree(monolingual_checkpoint_dir)
    # evaluate crosslingual cases: from spanish, russian, turkish into english
    for language in languages[1:]:
        metrics["{}_mono_adv_kldivloss".format(language)] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language,
                                            target_language="en_XX",
                                            lang_pairs="{}-en_XX".format(language),
                                            checkpoint="{}/checkpoint_last.pt".format(monolingual_adv_checkpoint_dir),
                                            lenpen=lenpen,
                                            min_len=min_len,
                                            use_encoder_output_adapter=use_encoder_output_adapter,
                                            append_src_tok=False)
        save_metrics(metrics, output_dir)
        free_memory()

    # few shot experiments.
    # Tune monolingual model using few data from spanish-english, russian-english and turkish-english datasets
    for language in languages[1:]:
        for data_size, max_epoch in zip([10, 100, 1000, 10000], ["12", "6", "4", "2"]):
            if language == "tr_TR" and data_size == 10000:
                continue
            checkpoint_dir = "{}/wikilingua_{}/{}-en_XX".format(output_dir, data_size, language)
            train_summarization_model(data_dir="wikilingua_{}".format(data_size),
                                      lang_pairs="{}-en_XX".format(language),
                                      checkpoint="{}/checkpoint_last.pt".format(monolingual_adv_checkpoint_dir),
                                      save_dir=checkpoint_dir,
                                      encoder_drop_residual=encoder_drop_residual,
                                      num_workers="1",
                                      validate=False,
                                      max_epoch=max_epoch,
                                      use_encoder_output_adapter=use_encoder_output_adapter,
                                      append_src_tok=False)
            free_memory()
            metrics["{}_mono_adv_{}".format(language, data_size)] = \
                generate_and_evaluate_summaries(directory="wikilingua",
                                                source_language=language,
                                                target_language="en_XX",
                                                lang_pairs="{}-en_XX".format(language),
                                                checkpoint="{}/checkpoint_last.pt".format(checkpoint_dir),
                                                lenpen=lenpen,
                                                min_len=min_len,
                                                use_encoder_output_adapter=use_encoder_output_adapter,
                                                append_src_tok=False)
            shutil.rmtree(checkpoint_dir)
            save_metrics(metrics, output_dir)
            free_memory()

    # tune monolingual model using complete data from spanish-english, russian-english and turkish-english datasets
    for language in languages[1:]:
        checkpoint_dir = "{}/wikilingua_all/{}-en_XX".format(output_dir, language)
        train_summarization_model(data_dir="wikilingua",
                                  lang_pairs="{}-en_XX".format(language),
                                  checkpoint="{}/checkpoint_last.pt".format(monolingual_adv_checkpoint_dir),
                                  save_dir=checkpoint_dir,
                                  encoder_drop_residual=encoder_drop_residual,
                                  use_encoder_output_adapter=use_encoder_output_adapter,
                                  append_src_tok=False)
        free_memory()
        metrics["{}_mono_adv_all".format(language)] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language,
                                            target_language="en_XX",
                                            lang_pairs="{}-en_XX".format(language),
                                            checkpoint="{}/checkpoint_best.pt".format(checkpoint_dir),
                                            lenpen=lenpen,
                                            min_len=min_len,
                                            use_encoder_output_adapter=use_encoder_output_adapter,
                                            append_src_tok=False)
        shutil.rmtree(checkpoint_dir)
        save_metrics(metrics, output_dir)
        free_memory()
    shutil.rmtree(monolingual_adv_checkpoint_dir)


if __name__ == "__main__":
    run_wikilingua_experiments()
