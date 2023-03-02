import sys

import torch

from fairseq_cli import train


def train_summarization_model(save_dir="default",
                              data_dir="xlsum",
                              lang_pairs="en_XX-en_XX",
                              checkpoint="../summarization_datasets/mbart.cc25.v2/model.pt",
                              freeze_embeddings=True):
    sys.argv.extend(
        [data_dir,
         "--encoder-normalize-before",
         "--decoder-normalize-before",
         "--arch", "mbart_large",
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
         "--total-num-update", "40000",
         "--dropout", "0.2",
         "--attention-dropout", "0.1",
         "--weight-decay", "0.01",
         "--max-tokens", "1024",
         "--no-epoch-checkpoints",
         "--save-dir", save_dir,
         "--seed", "222",
         "--log-format", "simple",
         "--restore-file", checkpoint,
         "--reset-optimizer",
         "--reset-meters",
         "--reset-dataloader",
         "--reset-lr-scheduler",
         "--max-epoch", "1",
         "--disable-validation",
         "--truncate-source",
         "--batch-size", "32",
         "--lang-tok-style", "mbart",
         "--num-workers", "8",
         "--update-freq", "2",
         "--ddp-backend", "no_c10d",
         "--find-unused-parameters"]
    )
    if freeze_embeddings:
        sys.argv.append("--freeze-embeddings")
    if torch.cuda.is_available():
        sys.argv.append("--fp16")
    train.cli_main()
    sys.argv = sys.argv[:1]


if __name__ == "__main__":
    train_summarization_model()
