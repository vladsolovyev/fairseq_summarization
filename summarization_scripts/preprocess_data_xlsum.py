from summarization_scripts.utils import preprocess_data


def main():
    for language in ["en_XX", "es_XX", "ru_RU", "gu_IN"]:
        for directory in ["xlsum_10", "xlsum_100", "xlsum_1000"]:
            preprocess_data(language, language, directory, directory)
        preprocess_data(language, language, "xlsum", "xlsum", add_test_data=True)
    for language in ["en_XX", "es_XX", "ru_RU"]:
        preprocess_data(language, language, "xlsum_10000", "xlsum_10000")
    for translation_language in ["es", "ru", "gu"]:
        directory = "xlsum_{}_en".format(translation_language)
        preprocess_data("en_XX", "en_XX", directory, directory, add_train_data=False, add_test_data=True)


if __name__ == '__main__':
    main()
