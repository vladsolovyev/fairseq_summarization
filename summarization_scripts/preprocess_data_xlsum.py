import sys

from fairseq_cli import preprocess

DATA = "../summarization_datasets/"
DICT = "{}mbart.cc25.v2/dict.txt".format(DATA)


def main():
    for directory in ["xlsum", "xlsum_10", "xlsum_100", "xlsum_1000", "xlsum_10000"]:
        for language in ["en_XX", "es_XX", "ru_RU"]:
            sys.argv.extend(["--source-lang", "input_text.{}".format(language),
                             "--target-lang", "summary.{}".format(language),
                             "--trainpref", "{}/{}/train".format(DATA, directory),
                             "--destdir", directory,
                             "--srcdict", DICT,
                             "--tgtdict", DICT,
                             "--workers", "20"])
            if directory == "xlsum":
                sys.argv.extend(["--testpref", "{}/{}/test".format(DATA, directory)])
            preprocess.cli_main()
            sys.argv = sys.argv[:1]


if __name__ == '__main__':
    main()
