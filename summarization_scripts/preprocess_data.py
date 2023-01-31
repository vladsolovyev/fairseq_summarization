import sys

from fairseq_cli import preprocess

DATA = "../summarization_datasets/"
DICT = "{}mbart.cc25.v2/dict.txt".format(DATA)


def main():
    for directory in ["xlsum", "xlsum_10", "xlsum_100", "xlsum_1000", "xlsum_10000"]:
        sys.argv.extend(
            ["--source-lang", "input_text.en_XX",
             "--target-lang", "summary.en_XX",
             "--trainpref", "{}/{}/train".format(DATA, directory),
             "--validpref", "{}/{}/valid".format(DATA, directory),
             "--testpref", "{}/{}/test".format(DATA, directory),
             "--destdir", directory,
             "--srcdict", DICT,
             "--tgtdict", DICT,
             "--workers", "20"]
        )
        preprocess.cli_main()


if __name__ == '__main__':
    main()
