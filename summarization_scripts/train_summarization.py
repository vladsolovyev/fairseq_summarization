import os
import sys

import torch

from fairseq_cli import train

torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_summarization_model(data_dir,
                              save_dir="default",
                              lang_pairs="en_XX-en_XX",
                              checkpoint="../summarization_datasets/mbart.cc25.v2/model.pt",
                              freeze_embeddings=False,
                              encoder_drop_residual=None,
                              max_update="200000",
                              validate_interval_updates="5000",
                              validate_interval="1"):
    sys.argv.extend(
        [data_dir,
         "--encoder-normalize-before",
         "--decoder-normalize-before",
         "--arch", "mbart_large_residual_drop",
         "--layernorm-embedding",
         "--task", "translation_multi_simple_epoch",
         "--sampling-method", "temperature",
         "--sampling-temperature", "1.5",
         "--encoder-langtok", "src",
         "--decoder-langtok",
         "--langs", "ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,"
                    "kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN",
         "--lang-pairs", lang_pairs,
         "--criterion", "label_smoothed_cross_entropy",
         "--label-smoothing", "0.1",
         "--optimizer", "adam",
         "--adam-eps", "1e-08",
         "--adam-betas", "(0.9, 0.999)",
         "--lr-scheduler", "polynomial_decay",
         "--lr", "2e-05",
         "--power", "1",
         "--end-learning-rate", "5e-9",
         "--clip-norm", "0.1",
         "--total-num-update", "200000",
         "--weight-decay", "0.01",
         "--dropout", "0.1",
         "--attention-dropout", "0.1",
         "--max-tokens", "2800",
         "--save-dir", save_dir,
         "--seed", "222",
         "--log-format", "simple",
         "--restore-file", checkpoint,
         "--reset-optimizer",
         "--reset-meters",
         "--reset-dataloader",
         "--reset-lr-scheduler",
         "--max-update", max_update,
         "--keep-best-checkpoints", "1",
         "--no-last-checkpoints",
         "--patience", "2",
         "--truncate-source",
         "--lang-tok-style", "mbart",
         "--num-workers", "16",
         "--update-freq", "3",
         "--ddp-backend", "no_c10d",
         "--find-unused-parameters",
         "--no-epoch-checkpoints",
         "--validate-interval", validate_interval,
         "--validate-interval-updates", validate_interval_updates]
    )
    if freeze_embeddings:
        sys.argv.append("--freeze-embeddings")
    if encoder_drop_residual:
        sys.argv.extend(["--encoder-drop-residual", encoder_drop_residual])
    if torch.cuda.is_available():
        sys.argv.append("--fp16")
    train.cli_main()
    sys.argv = sys.argv[:1]


if __name__ == "__main__":
    train_summarization_model("xlsum_10")
