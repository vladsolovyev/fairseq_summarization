import sys

from fairseq_cli import generate


def generate_and_evaluate_summaries(language, checkpoint_dir):
    sys.argv.extend(
        ["xlsum",
         "--path", "{}/checkpoint_last.pt".format(checkpoint_dir),
         "--task", "translation_from_pretrained_bart",
         "--source-lang", language,
         "--target-lang", language,
         "--gen-subset", "test",
         "--langs", "ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,"
                    "kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN",
         "--max-tokens", "1024",
         "--truncate-source",
         "--beam", "5",
         "--bpe", "sentencepiece",
         "--sentencepiece-model", "../summarization_datasets/mbart.cc25.v2/sentence.bpe.model",
         "--remove-bpe",
         "--scoring", "rougebert",
         "--required-batch-size-multiple", "1",
         "--batch-size", "4"]
    )
    results = generate.cli_main().scores
    print("Checkpoint: {}, language: {}, results: {}".format(checkpoint_dir, language, results))
    return results
