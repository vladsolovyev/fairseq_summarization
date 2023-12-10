from datetime import datetime
from pathlib import Path

from summarization_scripts.training_runner_wikilingua import run_wikilingua_experiments, calculate_wikilingua_supervised
from summarization_scripts.training_runner_xlsum import run_xlsum_experiments, calculate_xlsum_supervised


def main():
    date = datetime.today().strftime("%Y-%m-%d")
    start_xlsum_experiments(date)
    start_wikilingua_experiments(date)


def start_wikilingua_experiments(date):
    wikilingua_folder = "wiki_results"
    experiments_folder = "{}/{}".format(wikilingua_folder, date)
    Path(experiments_folder).mkdir(parents=True, exist_ok=True)
    calculate_wikilingua_supervised(output_dir=experiments_folder)

    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="base_model_with_adv",
                               adversarial_nllloss=True,
                               adversarial_kldivloss=True,
                               tune_after_adversarial=True,
                               add_translated_results=True,
                               few_shot=True)
    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="residual_drop_at_7",
                               encoder_drop_residual="6",
                               english_model=False,
                               adversarial_kldivloss=True,
                               tune_after_adversarial=True)
    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="encoder_output",
                               use_encoder_output_adapter=True,
                               english_model=False)
    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="decoder_adapter_tgt_lang_id",
                               use_decoder_adapter=True,
                               use_encoder_adapter="tgt_lang_id",
                               english_model=False)


def start_xlsum_experiments(date):
    xlsum_folder = "xlsum_results"
    experiments_folder = "{}/{}".format(xlsum_folder, date)
    Path(experiments_folder).mkdir(parents=True, exist_ok=True)
    calculate_xlsum_supervised(output_dir=experiments_folder)

    run_xlsum_experiments(experiments_folder=experiments_folder,
                          prefix="base_model",
                          add_translated_results=True,
                          few_shot=True)
    run_xlsum_experiments(experiments_folder=experiments_folder,
                          prefix="unfrozen_embeddings",
                          freeze_embeddings=False)
    run_xlsum_experiments(experiments_folder=experiments_folder,
                          prefix="frozen_decoder",
                          freeze_decoder_layers=True,
                          freeze_elements="everything")
    run_xlsum_experiments(experiments_folder=experiments_folder,
                          prefix="frozen_except_attn_qk",
                          freeze_encoder_layers=True,
                          freeze_decoder_layers=True,
                          freeze_elements="attn_qk")
    run_xlsum_experiments(experiments_folder=experiments_folder,
                          prefix="frozen_except_attn_and_layer_norm",
                          freeze_encoder_layers=True,
                          freeze_decoder_layers=True,
                          freeze_elements="attn_and_layer_norm")


if __name__ == "__main__":
    main()
