from datetime import datetime

from summarization_scripts.training_runner_wikilingua import run_wikilingua_experiments
from summarization_scripts.training_runner_xlsum import run_xlsum_experiments


def main():
    date = datetime.today().strftime('%Y-%m-%d')
    for run_experiments in [run_xlsum_experiments, run_wikilingua_experiments]:
        for encoder_drop_residual in [None, "5"]:
            for freeze_embeddings in [False, True]:
                drop_prefix = "drop_{}".format(encoder_drop_residual) if encoder_drop_residual else "no_drop"
                prefix = "{}/{}_freeze/{}".format(date,
                                                  "with" if freeze_embeddings else "without",
                                                  drop_prefix)
                run_experiments(freeze_embeddings=freeze_embeddings,
                                encoder_drop_residual=encoder_drop_residual,
                                prefix=prefix)
                if freeze_embeddings and encoder_drop_residual is None:
                    run_experiments(freeze_embeddings=freeze_embeddings,
                                    encoder_drop_residual=encoder_drop_residual,
                                    prefix=prefix,
                                    freeze_encoder_layers="6")


if __name__ == "__main__":
    main()
