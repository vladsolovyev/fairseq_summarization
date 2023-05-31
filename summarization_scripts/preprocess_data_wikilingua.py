from summarization_scripts.utils import preprocess_data


def main():
    for language in ["es_XX", "ru_RU", "tr_TR"]:
        preprocess_data(language,
                        "en_XX",
                        "wikilingua_{}-en_XX".format(language),
                        "wikilingua",
                        add_test_data=True,
                        add_validation_data=True)
        for data_size in [10, 100, 1000]:
            preprocess_data(language,
                            "en_XX",
                            "wikilingua_{}-en_XX_{}".format(language, data_size),
                            "wikilingua_{}".format(data_size))
    for language in ["es_XX", "ru_RU"]:
        preprocess_data(language,
                        "en_XX",
                        "wikilingua_{}-en_XX_{}".format(language, 10000),
                        "wikilingua_{}".format(10000))
    for language in ["en_XX", "es_XX", "ru_RU"]:
        preprocess_data(language, language, "wikilingua_{}-{}".format(language, language),
                        "wikilingua", add_validation_data=True)


if __name__ == '__main__':
    main()
