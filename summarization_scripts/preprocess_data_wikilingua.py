from summarization_scripts.utils import preprocess_data


def main():
    for language in ["es_XX", "ru_RU"]:
        for data_size in [10, 100, 1000, 10000]:
            preprocess_data(language,
                            "en_XX",
                            "wikilingua_cross_{}-en_XX_{}".format(language, data_size),
                            "wikilingua_cross_{}".format(data_size))
            preprocess_data(language,
                            "en_XX",
                            "wikilingua_cross_{}-en_XX".format(language),
                            "wikilingua_cross_{}".format(data_size),
                            add_validation_data=True,
                            add_train_data=False)
        preprocess_data(language,
                        "en_XX",
                        "wikilingua_cross_{}-en_XX".format(language),
                        "wikilingua_cross",
                        add_test_data=True,
                        add_validation_data=True)
    for language in ["en_XX", "es_XX", "ru_RU"]:
        preprocess_data(language, language, "wikilingua_mono", "wikilingua_mono", add_validation_data=True)


if __name__ == '__main__':
    main()
