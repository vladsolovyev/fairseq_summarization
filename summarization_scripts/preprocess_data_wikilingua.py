import sys

from fairseq_cli import preprocess

DATA = "../summarization_datasets/"
DICT = "{}mbart.cc25.v2/dict.txt".format(DATA)


def preprocess_data(source_language, target_language, directory):
    sys.argv.extend(["--source-lang", "input_text.{}".format(source_language),
                     "--target-lang", "summary.{}".format(target_language),
                     "--trainpref", "{}/{}/train".format(DATA, directory),
                     "--destdir", directory,
                     "--srcdict", DICT,
                     "--tgtdict", DICT,
                     "--workers", "20"])
    if directory in ["wikilingua_cross_es_XX-en_XX", "wikilingua_cross_ru_RU-en_XX"]:
        sys.argv.extend(["--testpref", "{}/{}/test".format(DATA, directory)])
    preprocess.cli_main()
    sys.argv = sys.argv[:1]

def main():
    for directory in ["wikilingua_cross_es_XX-en_XX", "wikilingua_cross_es_XX-en_XX_10",
                      "wikilingua_cross_es_XX-en_XX_100", "wikilingua_cross_es_XX-en_XX_1000",
                      "wikilingua_cross_es_XX-en_XX_10000"]:
        preprocess_data("es_XX", "en_XX", directory)
    for directory in ["wikilingua_cross_ru_RU-en_XX", "wikilingua_cross_ru_RU-en_XX_10",
                      "wikilingua_cross_ru_RU-en_XX_100", "wikilingua_cross_ru_RU-en_XX_1000",
                      "wikilingua_cross_ru_RU-en_XX_10000"]:
        preprocess_data("ru_RU", "en_XX", directory)
    for language in ["en_XX", "es_XX", "ru_RU"]:
        preprocess_data(language, language, "wikilingua_mono")


if __name__ == '__main__':
    main()
