import shutil

from summarization_scripts.generate_summaries import generate_and_evaluate_summaries
from summarization_scripts.train_summarization import train_summarization_model
from summarization_scripts.utils import free_memory, save_metrics, create_metrics_dict

languages = ["en_XX", "es_XX", "ru_RU", "tr_TR"]
lenpen = "1.0"
min_len = "10"


def calculate_wikilingua_baseline(prefix=""):
    metrics = dict()
    output_dir = "wiki_results/{}".format(prefix)

    # three crosslingual cases (spanish-english, russian-english and turkish-english) together as baseline
    checkpoint_dir = "{}/baseline".format(output_dir)
    train_summarization_model(data_dir="wikilingua",
                              lang_pairs=",".join(["{}-en_XX".format(language) for language in languages[1:]]),
                              save_dir=checkpoint_dir)
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
        shutil.rmtree(checkpoint_dir)
        save_metrics(metrics, output_dir)
        free_memory()


def run_wikilingua_experiments(encoder_drop_residual=None, prefix="", freeze_encoder_layers="0",
                               use_adversarial_loss=False, baseline_dir=None):
    # check how it works
    metrics = create_metrics_dict(baseline_dir)
    output_dir = "wiki_results/{}".format(prefix)

    # english, spanish, russian together, but monolingual data
    monolingual_checkpoint_dir = "{}/monolingual".format(output_dir)
    train_summarization_model(data_dir="wikilingua",
                              lang_pairs=",".join(["{}-{}".format(language, language) for language in languages[:3]]),
                              save_dir=monolingual_checkpoint_dir,
                              encoder_drop_residual=encoder_drop_residual,
                              freeze_encoder_layers=freeze_encoder_layers)
    free_memory()
    for language in languages[1:]:
        metrics["{}_mono".format(language)] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language,
                                            target_language="en_XX",
                                            lang_pairs="{}-en_XX".format(language),
                                            checkpoint="{}/checkpoint_best.pt".format(monolingual_checkpoint_dir),
                                            lenpen=lenpen,
                                            min_len=min_len)
        save_metrics(metrics, output_dir)
        free_memory()

    # few shot experiments.
    # Tune monolingual model using few data from spanish-english, russian-english and turkish-english datasets
    for language in languages[1:]:
        for data_size, max_epoch in zip([10, 100, 1000, 10000], ["12", "6", "4", "2"]):
            checkpoint_dir = "{}/wikilingua_{}/{}-en_XX".format(output_dir, data_size, language)
            train_summarization_model(data_dir="wikilingua_{}".format(data_size),
                                      lang_pairs="{}-en_XX".format(language),
                                      checkpoint="{}/checkpoint_best.pt".format(monolingual_checkpoint_dir),
                                      save_dir=checkpoint_dir,
                                      encoder_drop_residual=encoder_drop_residual,
                                      num_workers="1",
                                      validate=False,
                                      max_epoch=max_epoch)
            free_memory()
            metrics["{}_mono_{}".format(language, data_size)] = \
                generate_and_evaluate_summaries(directory="wikilingua",
                                                source_language=language,
                                                target_language="en_XX",
                                                lang_pairs="{}-en_XX".format(language),
                                                checkpoint="{}/checkpoint_last.pt".format(checkpoint_dir),
                                                lenpen=lenpen,
                                                min_len=min_len)
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
                                  encoder_drop_residual=encoder_drop_residual)
        free_memory()
        metrics["{}_mono_all".format(language)] = \
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
    if use_adversarial_loss:
        monolingual_adv_checkpoint_dir = "{}/monolingual_with_classifier".format(output_dir)
        train_summarization_model(data_dir="wikilingua",
                                  lang_pairs=",".join(
                                      ["{}-{}".format(language, language) for language in languages[:3]]),
                                  checkpoint="{}/checkpoint_best.pt".format(monolingual_checkpoint_dir),
                                  save_dir=monolingual_adv_checkpoint_dir,
                                  encoder_drop_residual=encoder_drop_residual,
                                  freeze_encoder_layers=freeze_encoder_layers,
                                  use_adversarial_loss=True)
        for language in languages[1:]:
            metrics["{}_mono_adv".format(language)] = \
                generate_and_evaluate_summaries(directory="wikilingua",
                                                source_language=language,
                                                target_language="en_XX",
                                                lang_pairs="{}-en_XX".format(language),
                                                checkpoint="{}/checkpoint_best.pt".format(monolingual_adv_checkpoint_dir),
                                                lenpen=lenpen,
                                                min_len=min_len)
            save_metrics(metrics, output_dir)
            free_memory()

        # few shot experiments.
        # Tune monolingual model using few data from spanish-english, russian-english and turkish-english datasets
        for language in languages[1:]:
            for data_size, max_epoch in zip([10, 100, 1000, 10000], ["12", "6", "4", "2"]):
                checkpoint_dir = "{}/wikilingua_{}/{}-en_XX".format(output_dir, data_size, language)
                train_summarization_model(data_dir="wikilingua_{}".format(data_size),
                                          lang_pairs="{}-en_XX".format(language),
                                          checkpoint="{}/checkpoint_best.pt".format(monolingual_adv_checkpoint_dir),
                                          save_dir=checkpoint_dir,
                                          encoder_drop_residual=encoder_drop_residual,
                                          num_workers="1",
                                          validate=False,
                                          max_epoch=max_epoch)
                free_memory()
                metrics["{}_mono_adv_{}".format(language, data_size)] = \
                    generate_and_evaluate_summaries(directory="wikilingua",
                                                    source_language=language,
                                                    target_language="en_XX",
                                                    lang_pairs="{}-en_XX".format(language),
                                                    checkpoint="{}/checkpoint_last.pt".format(checkpoint_dir),
                                                    lenpen=lenpen,
                                                    min_len=min_len)
                shutil.rmtree(checkpoint_dir)
                save_metrics(metrics, output_dir)
                free_memory()

        # tune monolingual model using complete data from spanish-english, russian-english and turkish-english datasets
        for language in languages[1:]:
            checkpoint_dir = "{}/wikilingua_all/{}-en_XX".format(output_dir, language)
            train_summarization_model(data_dir="wikilingua",
                                      lang_pairs="{}-en_XX".format(language),
                                      checkpoint="{}/checkpoint_best.pt".format(monolingual_adv_checkpoint_dir),
                                      save_dir=checkpoint_dir,
                                      encoder_drop_residual=encoder_drop_residual)
            free_memory()
            metrics["{}_mono_adv_all".format(language)] = \
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
        shutil.rmtree(monolingual_adv_checkpoint_dir)
    shutil.rmtree(monolingual_checkpoint_dir)


if __name__ == "__main__":
    run_wikilingua_experiments()
