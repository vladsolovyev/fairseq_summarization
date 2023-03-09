from summarization_scripts.utils import preprocess_data


def main():
    for directory in ["wikilingua_cross_es_XX-en_XX_10",
                      "wikilingua_cross_es_XX-en_XX_100", "wikilingua_cross_es_XX-en_XX_1000",
                      "wikilingua_cross_es_XX-en_XX_10000"]:
        preprocess_data("es_XX", "en_XX", directory)
    for directory in ["wikilingua_cross_ru_RU-en_XX_10",
                      "wikilingua_cross_ru_RU-en_XX_100", "wikilingua_cross_ru_RU-en_XX_1000",
                      "wikilingua_cross_ru_RU-en_XX_10000"]:
        preprocess_data("ru_RU", "en_XX", directory)
    preprocess_data("es_XX", "en_XX", "wikilingua_cross_es_XX-en_XX", add_test_data=True)
    preprocess_data("ru_RU", "en_XX", "wikilingua_cross_ru_RU-en_XX", add_test_data=True)
    for language in ["en_XX", "es_XX", "ru_RU"]:
        preprocess_data(language, language, "wikilingua_mono")


if __name__ == '__main__':
    main()
