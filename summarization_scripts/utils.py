import gc
import os
import sys

import pandas as pd
import torch
from GPUtil import showUtilization

from fairseq_cli import preprocess

DATA = "../summarization_datasets/"
DICT = "{}mbart.cc25.v2/dict.txt".format(DATA)


def preprocess_data(source_language, target_language, src_directory,
                    dst_directory, add_train_data=True, add_test_data=False, add_validation_data=False):
    sys.argv.extend(["--source-lang", "input_text.{}".format(source_language),
                     "--target-lang", "summary.{}".format(target_language),
                     "--destdir", dst_directory,
                     "--srcdict", DICT,
                     "--tgtdict", DICT,
                     "--workers", "20"])
    if add_train_data:
        sys.argv.extend(["--trainpref", "{}/{}/train".format(DATA, src_directory)])
    if add_test_data:
        sys.argv.extend(["--testpref", "{}/{}/test".format(DATA, src_directory)])
    if add_validation_data:
        sys.argv.extend(["--validpref", "{}/{}/validation".format(DATA, src_directory)])
    preprocess.cli_main()
    sys.argv = sys.argv[:1]


def save_metrics(metrics, output_dir):
    output_file = "{}/metrics.csv".format(output_dir)
    new_metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df = pd.read_csv(output_file, index_col=0)
    metrics_df = metrics_df.append(new_metrics_df)
    metrics_df = metrics_df[~metrics_df.index.duplicated(keep='last')].sort_index()
    metrics_df.to_csv(output_file, mode="w")


def free_memory():
    showUtilization()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    showUtilization()
