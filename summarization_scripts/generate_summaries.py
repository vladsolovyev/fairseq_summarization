import sys

from fairseq_cli import generate

if __name__ == "__main__":
    sys.argv.extend(
        ["xlsum_data",
         "--path", "checkpoints/checkpoint_best.pt",
         "--task", "translation_from_pretrained_bart",
         "--source-lang", "en_XX",
         "--target-lang", "en_XX",
         "--gen-subset", "test",
         "--langs", "ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,"
                    "kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN",
         "--max-tokens", "1024",
         "--truncate-source",
         "--beam", "5",
         "--bpe", "sentencepiece",
         "--sentencepiece-model", "../summarization_datasets/mbart.cc25.v2/sentence.bpe.model",
         "--remove-bpe"]
    )
    generate.cli_main()
