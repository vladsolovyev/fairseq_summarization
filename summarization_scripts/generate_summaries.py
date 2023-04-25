import sys

import torch

from fairseq_cli import generate


def generate_and_evaluate_summaries(directory,
                                    source_language="en_XX",
                                    target_language="en_XX",
                                    lang_pairs="en_XX-en_XX",
                                    checkpoint=None,
                                    lenpen="0.6",
                                    ngram="2",
                                    min_len="0",
                                    translate_to_lang="",
                                    rouge_scorer="huggingface",
                                    use_language_embeddings=False,):
    sys.argv.extend(
        [directory,
         "--path", checkpoint,
         "--task", "translation_multi_simple_epoch",
         "--gen-subset", "test",
         "--encoder-langtok", "src",
         "--decoder-langtok",
         "--langs", "ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,"
                    "kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN",
         "--lang-pairs", lang_pairs,
         "--source-lang", source_language,
         "--target-lang", target_language,
         "--max-tokens", "30000",
         "--truncate-source",
         "--beam", "5",
         "--bpe", "sentencepiece",
         "--sentencepiece-model", "../summarization_datasets/mbart.cc25.v2/sentence.bpe.model",
         "--remove-bpe",
         "--scoring", "rougebert",
         "--num-workers", "4",
         "--lang-tok-style", "mbart",
         "--max-len-b", "100",
         "--min-len", min_len,
         "--lenpen", lenpen,
         "--no-repeat-ngram-size", ngram,
         "--prefix-size", "1",
         "--translate-to-lang", translate_to_lang,
         "--rouge-scorer", rouge_scorer]
    )
    if torch.cuda.is_available():
        sys.argv.append("--fp16")
    if use_language_embeddings:
        sys.argv.append("--use-language-embeddings")
    results = generate.cli_main().scores
    print("Checkpoint: {}, languages: {}-{}, results: {}".format(
        checkpoint, source_language, target_language, results))
    sys.argv = sys.argv[:1]
    return results


if __name__ == "__main__":
    generate_and_evaluate_summaries()
