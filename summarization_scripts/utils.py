import gc
import sys

import pandas as pd
import torch
from GPUtil import showUtilization

from fairseq_cli import preprocess

DATA = "../summarization_datasets/"
DICT = "{}mbart.cc25.v2/dict.txt".format(DATA)


def preprocess_data(source_language, target_language, src_directory, dst_directory, add_test_and_validation_data=False):
    sys.argv.extend(["--source-lang", "input_text.{}".format(source_language),
                     "--target-lang", "summary.{}".format(target_language),
                     "--trainpref", "{}/{}/train".format(DATA, src_directory),
                     "--destdir", dst_directory,
                     "--srcdict", DICT,
                     "--tgtdict", DICT,
                     "--workers", "20"])
    if add_test_and_validation_data:
        sys.argv.extend(["--testpref", "{}/{}/test".format(DATA, src_directory)])
        sys.argv.extend(["--validpref", "{}/{}/validation".format(DATA, src_directory)])
    preprocess.cli_main()
    sys.argv = sys.argv[:1]


def save_metrics(metrics, output_dir):
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
    metrics_df.sort_index(inplace=True)
    metrics_df.to_csv("{}/metrics.csv".format(output_dir))


def free_memory():
    showUtilization()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    showUtilization()
