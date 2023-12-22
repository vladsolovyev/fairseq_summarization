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
    output_dir = "translated_tuned/4_langs"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile("translated_tuned/metrics.csv", "{}/metrics.csv".format(output_dir))
    metrics = dict()
    model_dir = "{}/model_dir".format(output_dir)
    # train using only monolingual english data
    train_summarization_model(data_dir="wikilingua",
                              lang_pairs="en_XX-en_XX",
                              checkpoint="translated_pretrained/4_langs/checkpoint_best.pt",
                              save_dir=model_dir,
                              freeze_encoder_layers=True,
                              freeze_decoder_layers=True,
                              freeze_elements="attn_qk",
                              freeze_adapters=True)
    free_memory()
    for language_pair in language_pairs:
        metrics["{}_{}_en".format(language_pair[0], language_pair[1])] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language_pair[0],
                                            target_language=language_pair[1],
                                            lang_pairs="{}-{}".format(language_pair[0], language_pair[1]),
                                            checkpoint="{}/checkpoint_best.pt".format(model_dir),
                                            lenpen=lenpen,
                                            min_len=min_len)
        save_metrics(metrics, output_dir)
        free_memory()
    shutil.rmtree(model_dir)

    # train using english, spanish, russian data together, but monolingual data
    train_summarization_model(data_dir="wikilingua",
                              lang_pairs=",".join(["{}-{}".format(language, language) for language in languages[:3]]),
                              checkpoint="translated_pretrained/4_langs/checkpoint_best.pt",
                              save_dir=model_dir,
                              sampling_temperature="10.0",
                              freeze_encoder_layers=True,
                              freeze_decoder_layers=True,
                              freeze_elements="attn_qk",
                              freeze_adapters=True)
    free_memory()
    for language_pair in language_pairs:
        metrics["{}_{}_multi_qk".format(language_pair[0], language_pair[1])] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language_pair[0],
                                            target_language=language_pair[1],
                                            lang_pairs="{}-{}".format(language_pair[0], language_pair[1]),
                                            checkpoint="{}/checkpoint_best.pt".format(model_dir),
                                            lenpen=lenpen,
                                            min_len=min_len)
        save_metrics(metrics, output_dir)
        free_memory()
    shutil.rmtree(model_dir)

    # train using english, spanish, russian data together, but monolingual data
    train_summarization_model(data_dir="wikilingua",
                              lang_pairs=",".join(["{}-{}".format(language, language) for language in languages[:3]]),
                              checkpoint="translated_pretrained/4_langs/checkpoint_best.pt",
                              save_dir=model_dir,
                              sampling_temperature="10.0",
                              freeze_encoder_layers=True,
                              freeze_decoder_layers=True,
                              freeze_elements="attn_and_layer_norm",
                              freeze_adapters=True)
    free_memory()
    for language_pair in language_pairs:
        metrics["{}_{}_multi_attn_and_layer_norm".format(language_pair[0], language_pair[1])] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language_pair[0],
                                            target_language=language_pair[1],
                                            lang_pairs="{}-{}".format(language_pair[0], language_pair[1]),
                                            checkpoint="{}/checkpoint_best.pt".format(model_dir),
                                            lenpen=lenpen,
                                            min_len=min_len)
        save_metrics(metrics, output_dir)
        free_memory()
    shutil.rmtree(model_dir)

    # train using english, spanish, russian data together, but monolingual data
    train_summarization_model(data_dir="wikilingua",
                              lang_pairs=",".join(["{}-{}".format(language, language) for language in languages[:3]]),
                              checkpoint="translated_pretrained/4_langs/checkpoint_best.pt",
                              save_dir=model_dir,
                              sampling_temperature="10.0",
                              freeze_decoder_layers=True,
                              freeze_elements="everything",
                              freeze_adapters=True)
    free_memory()
    for language_pair in language_pairs:
        metrics["{}_{}_multi_frozen_decoder".format(language_pair[0], language_pair[1])] = \
            generate_and_evaluate_summaries(directory="wikilingua",
                                            source_language=language_pair[0],
                                            target_language=language_pair[1],
                                            lang_pairs="{}-{}".format(language_pair[0], language_pair[1]),
                                            checkpoint="{}/checkpoint_best.pt".format(model_dir),
                                            lenpen=lenpen,
                                            min_len=min_len)
        save_metrics(metrics, output_dir)
        free_memory()
    shutil.rmtree(model_dir)


if __name__ == "__main__":
    tune_nmt_models()
