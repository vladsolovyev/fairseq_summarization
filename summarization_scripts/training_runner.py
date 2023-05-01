from datetime import datetime

from summarization_scripts.training_runner_wikilingua import run_wikilingua_experiments
from summarization_scripts.training_runner_xlsum import run_xlsum_experiments


def main():
    date = datetime.today().strftime('%Y-%m-%d')
    for run_experiments in [run_xlsum_experiments, run_wikilingua_experiments]:
        for encoder_drop_residual in ["4", None]:
            drop_prefix = "drop_{}".format(encoder_drop_residual) if encoder_drop_residual else "no_drop"
            prefix = "{}/{}".format(date, drop_prefix)
            run_experiments(encoder_drop_residual=encoder_drop_residual,
                            prefix=prefix)
            if encoder_drop_residual is None:
                prefix_6 = "{}/frozen_layers_6".format(prefix)
                run_experiments(encoder_drop_residual=encoder_drop_residual,
                                prefix=prefix_6,
                                freeze_encoder_layers="6")
                prefix_12 = "{}/frozen_layers_12".format(prefix)
                run_experiments(encoder_drop_residual=encoder_drop_residual,
                                prefix=prefix_12,
                                freeze_encoder_layers="12")


if __name__ == "__main__":
    main()
