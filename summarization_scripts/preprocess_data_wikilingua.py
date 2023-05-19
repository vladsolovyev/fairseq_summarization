from summarization_scripts.utils import preprocess_data


def main():
    for language in ["es_XX", "ru_RU"]:
        for data_size in [10, 100, 1000, 10000]:
            preprocess_data(language,
                            "en_XX",
                            "wikilingua_{}-en_XX_{}".format(language, data_size),
                            "wikilingua_{}".format(data_size))
            preprocess_data(language,
                            "en_XX",
                            "wikilingua_{}-en_XX".format(language),
                            "wikilingua_{}".format(data_size),
                            add_validation_data=True,
                            add_train_data=False)
        preprocess_data(language,
                        "en_XX",
                        "wikilingua_{}-en_XX".format(language),
                        "wikilingua",
                        add_test_data=True,
                        add_validation_data=True)
    for language in ["en_XX", "es_XX", "ru_RU", "de_DE"]:
        preprocess_data(language, language, "wikilingua_{}-{}".format(language, language),
                        "wikilingua", add_validation_data=True)
    for language in ["en_XX", "es_XX", "ru_RU"]:
        preprocess_data(language,
                        "de_DE",
                        "wikilingua_{}-de_DE".format(language),
                        "wikilingua",
                        add_validation_data=True)
        preprocess_data("de_DE",
                        language,
                        "wikilingua_de_DE-{}".format(language),
                        "wikilingua",
                        add_validation_data=True)


if __name__ == '__main__':
    main()
