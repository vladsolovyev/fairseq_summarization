import shutil
from pathlib import Path

from summarization_scripts.generate_summaries import generate_and_evaluate_summaries
from summarization_scripts.train_summarization import train_summarization_model
from summarization_scripts.utils import free_memory, save_metrics

languages = ["en_XX", "es_XX", "ru_RU", "tr_TR"]
lenpen = "1.0"
min_len = "10"


def train_classifiers():
    for dir_name, checkpoint_name, encoder_drop_residual in zip(
            ["base_model_with_adv/monolingual_multi",
             "base_model_with_adv/monolingual_with_classifier_kldivloss",
             "base_model_with_adv/monolingual_with_classifier_nll",
             "residual_drop_at_7/monolingual_with_classifier_kldivloss",
             "residual_drop_at_7/monolingual_with_classifier_nll"],
            ["checkpoint_best.pt", "checkpoint_last.pt", "checkpoint_last.pt", "checkpoint_last.pt", "checkpoint_last.pt"],
            [None, None, None, "6", "6"]):
        directory = "wiki_results/2023-10-12/{}".format(dir_name)
        model = "{}/{}".format(directory, checkpoint_name)
        save_dir = "{}/classification".format(directory)
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        metrics = dict()
        train_summarization_model(data_dir="wikilingua",
                                  lang_pairs=",".join(["{}-{}".format(language, language) for language in languages]),
                                  checkpoint=model,
                                  save_dir=save_dir,
                                  encoder_drop_residual=encoder_drop_residual,
                                  use_classifier=True,
                                  sampling_temperature="3.0",
                                  num_language_to_classify="4",
                                  append_src_tok=False,
                                  max_update="120000",
                                  validate=False,
                                  freeze_encoder_layers=True,
                                  freeze_decoder_layers=True,
                                  freeze_elements="everything")
        free_memory()
        for language in languages:
            metrics[language] = \
                generate_and_evaluate_summaries(directory="wikilingua",
                                                source_language=language,
                                                target_language=language,
                                                lang_pairs="{}-{}".format(language, language),
                                                checkpoint="{}/checkpoint_last.pt".format(save_dir),
                                                scoring="classification",
                                                lenpen=lenpen,
                                                min_len=min_len)
            save_metrics(metrics, directory)
            free_memory()
        shutil.rmtree(save_dir)


if __name__ == "__main__":
    train_classifiers()
