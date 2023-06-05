from datetime import datetime
from pathlib import Path

from summarization_scripts.training_runner_wikilingua import run_wikilingua_experiments, calculate_wikilingua_baseline
from summarization_scripts.training_runner_xlsum import run_xlsum_experiments, calculate_xlsum_baseline


def main():
    date = datetime.today().strftime("%Y-%m-%d")
    for run_experiments, calculate_baseline, folder in zip([run_wikilingua_experiments, run_xlsum_experiments],
                                                           [calculate_wikilingua_baseline, calculate_xlsum_baseline, calculate_wikilingua_baseline],
                                                           ["wiki_results", "xlsum_results"]):
        experiments_folder = "{}/{}".format(folder, date)
        Path(experiments_folder).mkdir(parents=True, exist_ok=True)
        calculate_baseline(output_dir=experiments_folder)

        prefix = "no_drop_not_frozen"
        run_experiments(experiments_folder=experiments_folder,
                        prefix=prefix)
        prefix = "with_drop_4"
        run_experiments(encoder_drop_residual="3",
                        experiments_folder=experiments_folder,
                        prefix=prefix)
        prefix = "frozen_layers_6"
        run_experiments(experiments_folder=experiments_folder,
                        prefix=prefix,
                        freeze_encoder_layers="6")


if __name__ == "__main__":
    main()
