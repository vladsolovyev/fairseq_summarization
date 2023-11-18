import sys

import torch

from fairseq_cli import generate


def generate_and_evaluate_summaries(directory,
                                    source_language="en_XX",
                                    target_language="en_XX",
                                    lang_pairs="en_XX-en_XX",
                                    checkpoint=None,
                                    lenpen="0.6",
                                    ngram="3",
                                    min_len="0",
                                    translate_to_lang="",
                                    rouge_scorer="huggingface",
                                    append_src_tok=True,
                                    scoring="rougebert",
                                    use_encoder_output_adapter=False,
                                    use_decoder_adapter=False,
                                    use_encoder_adapter="no"):
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
         "--enable-lang-ids",
         "--source-lang", source_language,
         "--target-lang", target_language,
         "--max-tokens", "30000",
         "--truncate-source",
         "--beam", "5",
         "--bpe", "sentencepiece",
         "--sentencepiece-model", "../summarization_datasets/mbart.cc25.v2/sentence.bpe.model",
         "--remove-bpe",
         "--scoring", scoring,
         "--num-workers", "4",
         "--lang-tok-style", "mbart",
         "--max-len-b", "100",
         "--min-len", min_len,
         "--lenpen", lenpen,
         "--no-repeat-ngram-size", ngram,
         "--prefix-size", "1",
         "--use-encoder-adapter", use_encoder_adapter,
         "--unkpen", "1e6"]
    )
    if torch.cuda.is_available():
        sys.argv.append("--fp16")
    if append_src_tok:
        sys.argv.append("--append-src-tok")
    if use_encoder_output_adapter:
        sys.argv.append("--use-encoder-output-adapter")
    if use_decoder_adapter:
        sys.argv.append("--use-decoder-adapter")
    if scoring == "rougebert":
        sys.argv.extend(["--translate-to-lang", translate_to_lang,
                         "--rouge-scorer", rouge_scorer])

    results = generate.cli_main().scores
    print("Checkpoint: {}, languages: {}-{}, results: {}".format(
        checkpoint, source_language, target_language, results))
    sys.argv = sys.argv[:1]
    return results


if __name__ == "__main__":
    generate_and_evaluate_summaries()
