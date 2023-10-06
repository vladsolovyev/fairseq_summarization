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


def run_wikilingua_experiments_with_adapters(experiments_folder=""):
    adapters_experiments_folder = "{}/adapters".format(experiments_folder)

    # train using english, spanish, russian data together, but monolingual data
    no_adapters_checkpoint_dir = "{}/no_adapters".format(adapters_experiments_folder)
    train_summarization_model(data_dir="wikilingua",
                              lang_pairs=",".join(["{}-{}".format(language, language) for language in languages[:3]]),
                              save_dir=no_adapters_checkpoint_dir,
                              freeze_encoder_layers=True,
                              freeze_decoder_layers=True,
                              freeze_elements="attn_qk")
    free_memory()

    for prefix, use_encoder_adapter, use_decoder_adapter, use_encoder_output_adapter in zip(
            ["layers_tgt", "layers_src", "enc_output", "layers_tgt_enc_output"],
            ["tgt_lang_id", "src_lang_id", "no", "tgt_lang_id"],
            [True, True, False, True],
            [False, False, True, True]
    ):
        output_dir = "{}/{}".format(adapters_experiments_folder, prefix)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        shutil.copyfile("{}/metrics.csv".format(experiments_folder),
                        "{}/metrics.csv".format(output_dir))
        metrics = dict()
        train_summarization_model(data_dir="wikilingua",
                                  lang_pairs=",".join(["{}-{}".format(language, language) for language in languages]),
                                  checkpoint="{}/checkpoint_best.pt".format(no_adapters_checkpoint_dir),
                                  save_dir=output_dir,
                                  freeze_encoder_layers=True,
                                  freeze_decoder_layers=True,
                                  freeze_elements="everything",
                                  use_encoder_output_adapter=use_encoder_output_adapter,
                                  use_decoder_adapter=use_decoder_adapter,
                                  use_encoder_adapter=use_encoder_adapter)
        # evaluate supervised cases
        for language_pair in language_pairs:
            metrics["{}_{}".format(language_pair[0], language_pair[1])] = \
                generate_and_evaluate_summaries(directory="wikilingua",
                                                source_language=language_pair[0],
                                                target_language=language_pair[1],
                                                lang_pairs="{}-{}".format(language_pair[0], language_pair[1]),
                                                checkpoint="{}/checkpoint_best.pt".format(output_dir),
                                                lenpen=lenpen,
                                                min_len=min_len,
                                                use_encoder_output_adapter=use_encoder_output_adapter,
                                                use_decoder_adapter=use_decoder_adapter,
                                                use_encoder_adapter=use_encoder_adapter)
            save_metrics(metrics, output_dir)
            free_memory()

        # shutil.rmtree(output_dir)


if __name__ == "__main__":
    run_wikilingua_experiments_with_adapters()
