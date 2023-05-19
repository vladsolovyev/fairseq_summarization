import pandas as pd
from datasets import load_dataset
from sentencepiece import SentencePieceProcessor

from summarization_datasets.utils import encode_dataset_and_create_statistics

data_types = ["train", "sampled_validation", "test"]
new_data_types = ["train", "validation", "test"]
columns = ["source", "target"]
dataset_name = "wikilingua"
mono_languages = ["en_XX", "es_XX", "ru_RU", "de_DE"]
crosslingual_source_languages_en = ["es_XX", "ru_RU"]
crosslingual_source_languages_de = ["en_XX", "es_XX", "ru_RU"]
datasets_monolingual = [load_dataset("GEM/wiki_lingua", "en", cache_dir="./cache"),
                        load_dataset("GEM/wiki_lingua", "es", cache_dir="./cache"),
                        load_dataset("GEM/wiki_lingua", "ru", cache_dir="./cache"),
                        load_dataset("GEM/wiki_lingua", "de", cache_dir="./cache")]
datasets_crosslingual_en = [load_dataset("GEM/wiki_lingua", name="es_en", cache_dir="./cache"),
                            load_dataset("GEM/wiki_lingua", name="ru_en", cache_dir="./cache")]
datasets_crosslingual_de = [load_dataset("GEM/wiki_lingua", name="ru_de", cache_dir="./cache"),
                            load_dataset("GEM/wiki_lingua", name="es_de", cache_dir="./cache"),
                            load_dataset("GEM/wiki_lingua", name="en_de", cache_dir="./cache")]
spp = SentencePieceProcessor(model_file="mbart.cc25.v2/sentence.bpe.model")


def main():
    statistics = dict()
    for language, dataset in zip(crosslingual_source_languages_en, datasets_crosslingual_en):
        for data_type, new_data_type in zip(data_types, new_data_types):
            filtered_dataset = dataset[data_type]. \
                filter(lambda sample: sample["target_language"] == "en" and sample["source_language"] != "en")
            statistics["{}-en_XX-{}".format(language, data_type)] = \
                encode_dataset_and_create_statistics(dataset_name, filtered_dataset, language, "en_XX",
                                                     new_data_type, columns, add_few_shot=True)

    for language, dataset in zip(crosslingual_source_languages_de, datasets_crosslingual_de):
        for data_type, new_data_type in zip(data_types, new_data_types):
            filtered_dataset = dataset[data_type]. \
                filter(lambda sample: sample["target_language"] == "de" and sample["source_language"] != "de")
            statistics["{}-de_DE-{}".format(language, data_type)] = \
                encode_dataset_and_create_statistics(dataset_name, filtered_dataset, language,
                                                     "de_DE", new_data_type, columns)
            filtered_dataset = dataset[data_type]. \
                filter(lambda sample: sample["target_language"] != "de" and sample["source_language"] == "de")
            statistics["{}-de_DE-{}".format(language, data_type)] = \
                encode_dataset_and_create_statistics(dataset_name, filtered_dataset, "de_DE",
                                                     language, new_data_type, columns)

    for language, dataset in zip(mono_languages, datasets_monolingual):
        for data_type, new_data_type in zip(data_types[:2], new_data_types[:2]):
            statistics["{}-{}-{}".format(language, language, data_type)] = \
                encode_dataset_and_create_statistics(dataset_name, dataset[data_type], language, language, new_data_type, columns)

    statistics_df = pd.DataFrame.from_dict(statistics, orient="index")
    statistics_df.to_csv("stat_wikilingua.txt")


if __name__ == '__main__':
    main()
