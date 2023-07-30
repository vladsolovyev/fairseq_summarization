import pandas as pd
from datasets import load_dataset

from summarization_datasets.utils import encode_dataset_and_create_statistics, create_translated_data

data_types = ["train", "sampled_validation", "test"]
new_data_types = ["train", "validation", "test"]
columns = ["source", "target"]
dataset_name = "wikilingua"
mono_languages = ["en_XX", "es_XX", "ru_RU"]
crosslingual_languages = ["es_XX", "ru_RU", "tr_TR"]
datasets_monolingual = [load_dataset("GEM/wiki_lingua", "en", cache_dir="./cache"),
                        load_dataset("GEM/wiki_lingua", "es", cache_dir="./cache"),
                        load_dataset("GEM/wiki_lingua", "ru", cache_dir="./cache")]
datasets_crosslingual = [load_dataset("GEM/wiki_lingua", name="es_en", cache_dir="./cache"),
                         load_dataset("GEM/wiki_lingua", name="ru_en", cache_dir="./cache"),
                         load_dataset("GEM/wiki_lingua", name="tr_en", cache_dir="./cache")]
dataset_es_ru = load_dataset("GEM/wiki_lingua", name="es_ru", cache_dir="./cache")
dataset_en_tr = load_dataset("GEM/wiki_lingua", name="en_tr", cache_dir="./cache")
dataset_tr_tr = load_dataset("GEM/wiki_lingua", name="tr", cache_dir="./cache")


def main():
    statistics = dict()
    for dataset, source_language, target_language in zip(
            datasets_crosslingual + [dataset_es_ru, dataset_en_tr, dataset_tr_tr],
            ["es", "ru", "tr", "es", "en", "tr"],
            ["en", "en", "en", "ru", "tr", "tr"]):
        filtered_dataset = dataset["test"].\
            filter(lambda sample: sample["target_language"] == target_language and sample["source_language"] == source_language)
        directory = "wikilingua_{}_{}".format(source_language, target_language)
        create_translated_data(filtered_dataset["source"], filtered_dataset["target"], directory, source_language)
    for language, dataset in zip(crosslingual_languages, datasets_crosslingual):
        for data_type, new_data_type in zip(data_types, new_data_types):
            filtered_dataset = dataset[data_type]. \
                filter(lambda sample: sample["target_language"] == "en" and sample["source_language"] != "en")
            statistics["{}-en_XX-{}".format(language, data_type)] = \
                encode_dataset_and_create_statistics(dataset_name, filtered_dataset, language, "en_XX",
                                                     new_data_type, columns, add_few_shot=True)

    for language, dataset in zip(mono_languages, datasets_monolingual):
        for data_type, new_data_type in zip(data_types[:2], new_data_types[:2]):
            statistics["{}-{}-{}".format(language, language, data_type)] = \
                encode_dataset_and_create_statistics(dataset_name, dataset[data_type], language, language,
                                                     new_data_type, columns)

    for data_type, new_data_type in zip(data_types, new_data_types):
        filtered_dataset = dataset_es_ru[data_type]. \
            filter(lambda sample: sample["target_language"] == "ru" and sample["source_language"] == "es")
        statistics["es_XX-ru_RU-{}".format(data_type)] = \
            encode_dataset_and_create_statistics(dataset_name, filtered_dataset, "es_XX", "ru_RU",
                                                 new_data_type, columns, add_few_shot=True)

    for data_type, new_data_type in zip(data_types, new_data_types):
        filtered_dataset = dataset_en_tr[data_type]. \
            filter(lambda sample: sample["target_language"] == "tr" and sample["source_language"] == "en")
        statistics["en_XX-tr_TR-{}".format(data_type)] = \
            encode_dataset_and_create_statistics(dataset_name, filtered_dataset, "en_XX", "tr_TR",
                                                 new_data_type, columns, add_few_shot=True)

    for data_type, new_data_type in zip(data_types, new_data_types):
        statistics["tr_TR-tr_TR-{}".format(data_type)] = \
            encode_dataset_and_create_statistics(dataset_name, dataset_tr_tr[data_type], "tr_TR", "tr_TR",
                                                 new_data_type, columns, add_few_shot=True)

    statistics_df = pd.DataFrame.from_dict(statistics, orient="index").sort_index()
    statistics_df.to_csv("stat_wikilingua.txt")


if __name__ == '__main__':
    main()
