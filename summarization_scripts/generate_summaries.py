import sys

import torch

from fairseq_cli import generate


def generate_and_evaluate_summaries(language="en_XX",
                                    lang_pairs="en_XX-en_XX",
                                    checkpoint_dir="default"):
    sys.argv.extend(
        ["xlsum",
         "--path", "{}/checkpoint_last.pt".format(checkpoint_dir),
         "--task", "translation_multi_simple_epoch",
         "--gen-subset", "test",
         "--encoder-langtok", "src",
         "--decoder-langtok",
         "--langs", "ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,"
                    "kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN",
         "--lang-pairs", lang_pairs,
         "--source-lang", language,
         "--target-lang", language,
         "--max-tokens", "1024",
         "--truncate-source",
         "--beam", "5",
         "--bpe", "sentencepiece",
         "--sentencepiece-model", "../summarization_datasets/mbart.cc25.v2/sentence.bpe.model",
         "--remove-bpe",
         "--scoring", "rougebert",
         "--batch-size", "64",
         "--num-workers", "16",
         "--lang-tok-style", "mbart",
         "--max-len-b", "84",
         "--min-len", "20",
         "--lenpen", "0.8",
         "--no-repeat-ngram-size", "2"]
    )
    if torch.cuda.is_available():
        sys.argv.append("--memory-efficient-fp16")
    results = generate.cli_main().scores
    print("Checkpoint: {}, language: {}, results: {}".format(checkpoint_dir, language, results))
    sys.argv = sys.argv[:1]
    return results


if __name__ == "__main__":
    generate_and_evaluate_summaries()
