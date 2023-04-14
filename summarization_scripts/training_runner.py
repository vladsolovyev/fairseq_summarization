from datetime import datetime

from summarization_scripts.training_runner_wikilingua import run_wikilingua_experiments
from summarization_scripts.training_runner_xlsum import run_xlsum_experiments


def main():
    date = datetime.today().strftime('%Y-%m-%d')
    for encoder_drop_residual in ["5", None]:
        for run_experiments in [run_xlsum_experiments, run_wikilingua_experiments]:
            for freeze_embeddings in [False, True]:
                drop_prefix = "drop_{}".format(encoder_drop_residual) if encoder_drop_residual else "no_drop"
                prefix = "{}/{}_freeze/{}".format(date,
                                                  "with" if freeze_embeddings else "without",
                                                  drop_prefix)
                run_experiments(freeze_embeddings=freeze_embeddings,
                                encoder_drop_residual=encoder_drop_residual,
                                prefix=prefix)


if __name__ == "__main__":
    main()
