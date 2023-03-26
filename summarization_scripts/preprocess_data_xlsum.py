from summarization_scripts.utils import preprocess_data


def main():
    for language in ["en_XX", "es_XX", "ru_RU", "my_MM"]:
        for directory in ["xlsum_10", "xlsum_100", "xlsum_1000"]:
            preprocess_data(language, language, directory, directory)
        preprocess_data(language, language, "xlsum", "xlsum", add_test_data=True)
    for language in ["en_XX", "es_XX", "ru_RU"]:
        preprocess_data(language, language, "xlsum_10000", "xlsum_10000")


if __name__ == '__main__':
    main()
