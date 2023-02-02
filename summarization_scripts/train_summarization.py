import sys

from fairseq_cli import train


def train_summarization_model(data_dir="xlsum", language="en_XX",
                              checkpoint="../summarization_datasets/mbart.cc25.v2/model.pt", save_dir=""):
    sys.argv.extend(
        [data_dir,
         "--encoder-normalize-before",
         "--decoder-normalize-before",
         "--arch", "mbart_large",
         "--layernorm-embedding",
         "--task", "translation_from_pretrained_bart",
         "--source-lang", language,
         "--target-lang", language,
         "--criterion", "label_smoothed_cross_entropy",
         "--label-smoothing", "0.2",
         "--optimizer", "adam",
         "--adam-eps", "1e-06",
         "--adam-betas", "(0.9, 0.98)",
         "--lr-scheduler", "polynomial_decay",
         "--lr", "3e-05",
         "--warmup-updates", "2500",
         "--total-num-update", "40000",
         "--dropout", "0.3",
         "--attention-dropout", "0.1",
         "--weight-decay", "0.0",
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
         "--langs", "ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,"
                    "kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN",
         "--max-epoch", "1",
         "--disable-validation",
         "--truncate-source"]
    )
    train.cli_main()


if __name__ == "__main__":
    train_summarization_model()
