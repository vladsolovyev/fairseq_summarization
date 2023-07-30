from summarization_scripts.utils import preprocess_data


def main():
    for source_language, target_language in zip(["es_XX", "ru_RU", "tr_TR", "es_XX", "en_XX", "tr_TR"],
                                                ["en_XX", "en_XX", "en_XX", "ru_RU", "tr_TR", "tr_TR"]):
        preprocess_data(source_language,
                        target_language,
                        "wikilingua_{}-{}".format(source_language, target_language),
                        "wikilingua",
                        add_test_data=True,
                        add_validation_data=True)
        for data_size in [10, 100, 1000]:
            preprocess_data(source_language,
                            target_language,
                            "wikilingua_{}-{}_{}".format(source_language, target_language, data_size),
                            "wikilingua_{}".format(data_size))
    for source_language, target_language in zip(["es_XX", "ru_RU", "es_XX"], ["en_XX", "en_XX", "ru_RU"]):
        preprocess_data(source_language,
                        target_language,
                        "wikilingua_{}-{}_{}".format(source_language, target_language, 10000),
                        "wikilingua_{}".format(10000))
    for language in ["en_XX", "es_XX", "ru_RU"]:
        preprocess_data(language, language, "wikilingua_{}-{}".format(language, language),
                        "wikilingua", add_validation_data=True)
    for source_language, translation_language in zip(["es", "ru", "tr", "es", "en", "tr"],
                                                     ["en", "en", "en", "ru", "tr", "tr"]):
        directory = "wikilingua_{}_{}".format(source_language, translation_language)
        preprocess_data("en_XX", "en_XX", directory, directory, add_train_data=False, add_test_data=True)


if __name__ == '__main__':
    main()
