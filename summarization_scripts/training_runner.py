from datetime import datetime
from pathlib import Path

from summarization_scripts.training_runner_wikilingua import run_wikilingua_experiments, calculate_wikilingua_baseline
from summarization_scripts.training_runner_xlsum import run_xlsum_experiments, calculate_xlsum_baseline


def main():
    date = datetime.today().strftime("%Y-%m-%d")
    start_xlsum_experiments(date)
    start_wikilingua_experiments(date)


def start_wikilingua_experiments(date):
    wikilingua_folder = "wiki_results"
    experiments_folder = "{}/{}".format(wikilingua_folder, date)
    Path(experiments_folder).mkdir(parents=True, exist_ok=True)
    calculate_wikilingua_baseline(output_dir=experiments_folder)

    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="base_model_with_adv",
                               adversarial_nllloss=True,
                               adversarial_kldivloss=True)
    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="normal_label_smoothing",
                               label_smoothing="0.1")
    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="masked_label",
                               label_smoothing="0.1",
                               masked_labels=True)
    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="residual_drop_at_4",
                               encoder_drop_residual="3")
    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="frozen_layers_8",
                               freeze_encoder_layers="8")
    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="frozen_encoder_decoder",
                               freeze_encoder_layers="8",
                               freeze_decoder_layers=True)
    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="decoder_adapter",
                               use_decoder_adapter=True)
    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="encoder_output_adapter",
                               use_encoder_output_adapter=True)
    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="both_adapters",
                               use_decoder_adapter=True,
                               use_encoder_output_adapter=True)
    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="frozen_everything_except_attention_key_and_query",
                               freeze_encoder_layers="12",
                               freeze_decoder_layers=True)
    # here combine best parameters


def start_xlsum_experiments(date):
    xlsum_folder = "xlsum_results"
    experiments_folder = "{}/{}".format(xlsum_folder, date)
    Path(experiments_folder).mkdir(parents=True, exist_ok=True)
    calculate_xlsum_baseline(output_dir=experiments_folder)

    run_xlsum_experiments(experiments_folder=experiments_folder,
                          prefix="base_model_with_adv",
                          adversarial_nllloss=True,
                          adversarial_kldivloss=True,
                          add_translated_results=True)
    run_xlsum_experiments(experiments_folder=experiments_folder,
                          prefix="normal_label_smoothing",
                          label_smoothing="0.1")
    run_xlsum_experiments(experiments_folder=experiments_folder,
                          prefix="masked_label",
                          label_smoothing="0.1",
                          masked_labels=True)
    run_xlsum_experiments(experiments_folder=experiments_folder,
                          prefix="residual_drop_at_4",
                          encoder_drop_residual="3")
    run_xlsum_experiments(experiments_folder=experiments_folder,
                          prefix="frozen_layers_8",
                          freeze_encoder_layers="8")
    run_xlsum_experiments(experiments_folder=experiments_folder,
                          prefix="frozen_encoder_decoder",
                          freeze_encoder_layers="8",
                          freeze_decoder_layers=True)
    run_xlsum_experiments(experiments_folder=experiments_folder,
                          prefix="frozen_everything_except_attention_key_and_query",
                          freeze_encoder_layers="12",
                          freeze_decoder_layers=True)
    # here combine best parameters


if __name__ == "__main__":
    main()
