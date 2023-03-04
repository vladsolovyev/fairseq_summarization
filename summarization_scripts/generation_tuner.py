import pandas as pd

from summarization_scripts.generate_summaries import generate_and_evaluate_summaries


def main():
    metrics = dict()
    for lenpen in ["0.5", "1.0", "1.5"]:
        for ngram in ["1", "2", "3"]:
            for min_len in ["30", "40", "50"]:
                print("lenpen: {}, ngram: {}, minlen: {}".format(lenpen, ngram, min_len))
                metrics["lenpen: {}, ngram: {}, minlen: {}".format(lenpen, ngram, min_len)] =\
                    generate_and_evaluate_summaries(lenpen=lenpen, ngram=ngram, min_len=min_len)
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.to_csv("tuning_results.csv")


if __name__ == "__main__":
    main()
