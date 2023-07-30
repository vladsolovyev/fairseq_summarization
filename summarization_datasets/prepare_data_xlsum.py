import pandas as pd
from datasets import load_dataset

from summarization_datasets.utils import encode_dataset_and_create_statistics, create_translated_data

data_types = ["train", "validation", "test"]
columns = ["text", "target"]
languages = ["en_XX", "es_XX", "ru_RU", "gu_IN"]
datasets = [load_dataset("GEM/xlsum", "english", cache_dir="./cache"),
            load_dataset("GEM/xlsum", "spanish", cache_dir="./cache"),
            load_dataset("GEM/xlsum", "russian", cache_dir="./cache"),
            load_dataset("GEM/xlsum", "gujarati", cache_dir="./cache")]


def filter_datasets():
    for dataset in datasets:
        dataset["train"] = dataset["train"].filter(
            lambda sample: len(sample["text"].split()) > 20 and len(sample["target"].split()) > 10)


def main():
    filter_datasets()
    for dataset, lang in zip(datasets[1:], ["es", "ru", "gu"]):
        directory = "xlsum_{}_en".format(lang)
        create_translated_data(dataset, directory, lang)
    statistics = dict()
    for language, dataset in zip(languages, datasets):
        for data_type in data_types:
            statistics["{}-{}".format(language, data_type)] =\
                encode_dataset_and_create_statistics("xlsum", dataset[data_type], language, language,
                                                     data_type, columns, add_few_shot=(language != "en_XX"))
    statistics_df = pd.DataFrame.from_dict(statistics, orient="index")
    statistics_df.to_csv("stat_xlsum.txt")


if __name__ == '__main__':
    main()
