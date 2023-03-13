import pandas as pd

from summarization_scripts.generate_summaries import generate_and_evaluate_summaries


def main():
    metrics = dict()
    for lenpen in ["0.5", "1.0", "1.5"]:
        for min_len in ["0", "10", "20", "30"]:
            print("lenpen: {}, minlen: {}".format(lenpen, min_len))
            metrics["lenpen: {}, minlen: {}".format(lenpen, min_len)] =\
                generate_and_evaluate_summaries(directory="wikilingua_cross",
                                                source_language="es_XX",
                                                target_language="en_XX",
                                                lang_pairs="es_XX-en_XX",
                                                lenpen=lenpen,
                                                min_len=min_len)
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.to_csv("tuning_results.csv")


if __name__ == "__main__":
    main()
