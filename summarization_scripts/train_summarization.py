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
                              freeze_decoder_layers=False,
                              use_language_embeddings=False,
                              use_language_embeddings_encoder_output=False,
                              encoder_drop_residual=None,
                              max_update="120000",
                              freeze_encoder_layers="0",
                              num_workers="16",
                              use_adversarial_loss=False,
                              validate=True,
                              max_epoch=None):
    sys.argv.extend(
        [data_dir,
         "--encoder-normalize-before",
         "--decoder-normalize-before",
         "--layernorm-embedding",
         "--sampling-method", "temperature",
         "--sampling-temperature", "20",
         "--encoder-langtok", "src",
         "--decoder-langtok",
         "--langs", "ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,"
                    "kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN",
         "--lang-pairs", lang_pairs,
         "--enable-lang-ids",
         "--optimizer", "adam",
         "--adam-eps", "1e-08",
         "--adam-betas", "(0.9, 0.999)",
         "--lr-scheduler", "polynomial_decay",
         "--lr", "2e-05",
         "--power", "1",
         "--end-learning-rate", "5e-9",
         "--clip-norm", "0.1",
         "--total-num-update", max_update,
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
         "--truncate-source",
         "--lang-tok-style", "mbart",
         "--num-workers", num_workers,
         "--update-freq", "3",
         "--ddp-backend", "no_c10d",
         "--find-unused-parameters",
         "--no-epoch-checkpoints",
         "--freeze-embeddings",
         "--freeze-encoder-layers", freeze_encoder_layers]
    )
    if freeze_decoder_layers:
        sys.argv.append("--freeze-decoder-layers")
    if use_language_embeddings:
        sys.argv.append("--use-language-embeddings")
    if use_language_embeddings_encoder_output:
        sys.argv.append("--use-language-embeddings-encoder-output")
    if encoder_drop_residual:
        sys.argv.extend(["--encoder-drop-residual", encoder_drop_residual])
    if torch.cuda.is_available():
        sys.argv.append("--fp16")
    if use_adversarial_loss:
        sys.argv.extend(["--arch", "language_classification_transformer",
                         "--task", "translation_multi_simple_epoch_task_with_adversarial_loss",
                         "--criterion", "language_classification_cross_entropy",
                         "--num-language-to-classify", "3",
                         "--patience", "2",
                         "--language-classifier-one-vs-rest", "-1"])
    else:
        sys.argv.extend(["--arch", "mbart_large_residual_drop",
                         "--task", "translation_multi_simple_epoch",
                         "--patience", "1",
                         "--criterion", "cross_entropy"])
    if validate:
        sys.argv.extend(["--validate-interval-updates", "5000",
                         "--validate-after-updates", "30000",
                         "--no-last-checkpoints"])
    else:
        sys.argv.append("--disable-validation")
    if max_epoch:
        sys.argv.extend(["--max-epoch", max_epoch])

    train.cli_main()
    sys.argv = sys.argv[:1]


if __name__ == "__main__":
    train_summarization_model("xlsum_10")
