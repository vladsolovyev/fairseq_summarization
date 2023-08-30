import shutil
from pathlib import Path

from summarization_scripts.generate_summaries import generate_and_evaluate_summaries
from summarization_scripts.train_summarization import train_summarization_model
from summarization_scripts.utils import free_memory, save_metrics

languages = ["en_XX", "es_XX", "ru_RU", "tr_TR"]
language_pairs = [("es_XX", "en_XX"), ("ru_RU", "en_XX"), ("tr_TR", "en_XX"),
                  ("es_XX", "ru_RU"), ("en_XX", "tr_TR"), ("tr_TR", "tr_TR")]
lenpen = "1.0"
min_len = "10"


def tune_nmt_models():
    for dir_name, use_decoder_adapter, use_encoder_output_adapter in zip(
            ["3_langs", "4_langs", "3_langs_frozen_decoder", "4_langs_frozen_decoder",
             "4_langs_decoder_adapter", "4_langs_encoder_output_adapter"],
            [False, False, False, False, True, False],
            [False, False, False, False, False, True]):
        for checkpoint_dir, checkpoint_name in zip(["", "adv_nll/", "adv_kldivloss/"],
                                                   ["checkpoint_best.pt", "checkpoint_last.pt", "checkpoint_last.pt"]):
            output_dir = "translated_tuned/{}/{}".format(dir_name, checkpoint_dir)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            shutil.copyfile("translated_tuned/metrics.csv", "{}/metrics.csv".format(output_dir))
            metrics = dict()
            model_dir = "{}/model_dir".format(output_dir)
            # train using only monolingual english data
            train_summarization_model(data_dir="wikilingua",
                                      lang_pairs="en_XX-en_XX",
                                      checkpoint="translated_pretrained/{}/{}{}".format(dir_name, checkpoint_dir, checkpoint_name),
                                      save_dir=model_dir,
                                      freeze_encoder_layers="12",
                                      freeze_decoder_layers=True,
                                      freeze_elements="attn_qk",
                                      use_encoder_output_adapter=use_encoder_output_adapter,
                                      use_decoder_adapter=use_decoder_adapter)
            free_memory()
            for language_pair in language_pairs:
                metrics["{}_{}_en".format(language_pair[0], language_pair[1])] = \
                    generate_and_evaluate_summaries(directory="wikilingua",
                                                    source_language=language_pair[0],
                                                    target_language=language_pair[1],
                                                    lang_pairs="{}-{}".format(language_pair[0], language_pair[1]),
                                                    checkpoint="{}/checkpoint_best.pt".format(model_dir),
                                                    lenpen=lenpen,
                                                    min_len=min_len,
                                                    use_encoder_output_adapter=use_encoder_output_adapter,
                                                    use_decoder_adapter=use_decoder_adapter)
                save_metrics(metrics, output_dir)
                free_memory()
            shutil.rmtree(model_dir)

            # train using english, spanish, russian data together, but monolingual data
            train_summarization_model(data_dir="wikilingua",
                                      lang_pairs=",".join(["{}-{}".format(language, language) for language in languages[:3]]),
                                      checkpoint="translated_pretrained/{}/{}{}".format(dir_name, checkpoint_dir, checkpoint_name),
                                      save_dir=model_dir,
                                      freeze_encoder_layers="12",
                                      freeze_decoder_layers=True,
                                      freeze_elements="attn_qk",
                                      use_encoder_output_adapter=use_encoder_output_adapter,
                                      use_decoder_adapter=use_decoder_adapter)
            free_memory()
            for language_pair in language_pairs:
                metrics["{}_{}_multi".format(language_pair[0], language_pair[1])] = \
                    generate_and_evaluate_summaries(directory="wikilingua",
                                                    source_language=language_pair[0],
                                                    target_language=language_pair[1],
                                                    lang_pairs="{}-{}".format(language_pair[0], language_pair[1]),
                                                    checkpoint="{}/checkpoint_best.pt".format(model_dir),
                                                    lenpen=lenpen,
                                                    min_len=min_len,
                                                    use_encoder_output_adapter=use_encoder_output_adapter,
                                                    use_decoder_adapter=use_decoder_adapter)
                save_metrics(metrics, output_dir)
                free_memory()
            shutil.rmtree(model_dir)


if __name__ == "__main__":
    tune_nmt_models()
