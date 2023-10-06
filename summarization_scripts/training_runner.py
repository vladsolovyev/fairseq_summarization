from datetime import datetime
from pathlib import Path

from summarization_scripts.training_runner_adapters import run_wikilingua_experiments_with_adapters
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
                               adversarial_kldivloss=True,
                               add_translated_results=True)
    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="unfrozen_embeddings",
                               freeze_embeddings=False)
    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="frozen_decoder",
                               freeze_decoder_layers=True,
                               freeze_elements="everything",
                               adversarial_kldivloss=True)
    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="frozen_except_attn_qk",
                               freeze_decoder_layers=True,
                               freeze_elements="attn_qk",
                               adversarial_kldivloss=True)
    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="frozen_except_attn_and_layer_norm",
                               freeze_decoder_layers=True,
                               freeze_elements="attn_and_layer_norm",
                               adversarial_kldivloss=True)
    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="frozen_except_attn_vqk",
                               freeze_decoder_layers=True,
                               freeze_elements="attn_vqk",
                               adversarial_kldivloss=True)
    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="residual_drop_at_4",
                               encoder_drop_residual="3",
                               freeze_decoder_layers=True,
                               freeze_elements="attn_vqk",
                               adversarial_kldivloss=True)
    # make experiments with label smoothing with the best model
    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="normal_label_smoothing",
                               label_smoothing="0.1",
                               adversarial_kldivloss=True)
    run_wikilingua_experiments(experiments_folder=experiments_folder,
                               prefix="masked_label",
                               label_smoothing="0.1",
                               masked_labels=True,
                               adversarial_kldivloss=True)
    run_wikilingua_experiments_with_adapters(experiments_folder=experiments_folder)


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
                          prefix="unfrozen_embeddings",
                          freeze_embeddings=False)
    run_xlsum_experiments(experiments_folder=experiments_folder,
                          prefix="frozen_decoder",
                          freeze_decoder_layers=True,
                          freeze_elements="everything",
                          adversarial_kldivloss=True)
    run_xlsum_experiments(experiments_folder=experiments_folder,
                          prefix="frozen_except_attn_qk",
                          freeze_encoder_layers=True,
                          freeze_decoder_layers=True,
                          freeze_elements="attn_qk",
                          adversarial_kldivloss=True)
    run_xlsum_experiments(experiments_folder=experiments_folder,
                          prefix="frozen_except_attn_and_layer_norm",
                          freeze_encoder_layers=True,
                          freeze_decoder_layers=True,
                          freeze_elements="attn_and_layer_norm",
                          adversarial_kldivloss=True)
    run_xlsum_experiments(experiments_folder=experiments_folder,
                          prefix="frozen_except_attn_vqk",
                          freeze_encoder_layers=True,
                          freeze_decoder_layers=True,
                          freeze_elements="attn_vqk",
                          adversarial_kldivloss=True)
    run_xlsum_experiments(experiments_folder=experiments_folder,
                          prefix="residual_drop_at_4",
                          encoder_drop_residual="3",
                          freeze_decoder_layers=True,
                          freeze_elements="attn_vqk",
                          adversarial_kldivloss=True)
    run_xlsum_experiments(experiments_folder=experiments_folder,
                          prefix="normal_label_smoothing",
                          freeze_encoder_layers=True,
                          freeze_decoder_layers=True,
                          freeze_elements="attn_vqk",
                          label_smoothing="0.1")
    run_xlsum_experiments(experiments_folder=experiments_folder,
                          prefix="masked_label",
                          freeze_encoder_layers=True,
                          freeze_decoder_layers=True,
                          freeze_elements="attn_vqk",
                          label_smoothing="0.1",
                          masked_labels=True)


if __name__ == "__main__":
    main()
