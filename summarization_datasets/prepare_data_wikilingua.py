from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentencepiece import SentencePieceProcessor

from summarization_datasets.utils import write_to_file

data_types = ["train", "test"]
columns = ["source", "target"]
new_columns = ["input_text", "summary"]
mono_languages = ["en_XX", "es_XX", "ru_RU"]
crosslingual_source_languages = ["es_XX", "ru_RU"]
datasets_monolingual = [load_dataset("GEM/wiki_lingua", "en", cache_dir="./cache"),
                        load_dataset("GEM/wiki_lingua", "es", cache_dir="./cache"),
                        load_dataset("GEM/wiki_lingua", "ru", cache_dir="./cache")]
datasets_crosslingual = [load_dataset("GEM/wiki_lingua", name="es_en", cache_dir="./cache"),
                         load_dataset("GEM/wiki_lingua", name="ru_en", cache_dir="./cache")]
spp = SentencePieceProcessor(model_file="mbart.cc25.v2/sentence.bpe.model")


def main():
    statistics = dict()
    for language, dataset in zip(crosslingual_source_languages, datasets_crosslingual):
        for data_type in data_types:
            dataset[data_type] = dataset[data_type].\
                filter(lambda sample: sample["target_language"] == "en" and sample["source_language"] != "en")
            dataset_statistics = dict()
            dataset_statistics["samples_num"] = len(dataset[data_type])
            for column, new_column in zip(columns, new_columns):
                dataset_statistics["{} average length (words)".format(new_column)] =\
                    np.mean([len(sentence.split())for sentence in dataset[data_type][column]])
                encoded_column = spp.encode(dataset[data_type][column], out_type=str)
                dataset_statistics["{} average length (tokens)".format(new_column)] =\
                    np.mean([len(encoded_sample)for encoded_sample in encoded_column])
                lang = language if column == "source" else "en_XX"
                encoded_texts = [" ".join(encoded_parts) for encoded_parts in encoded_column]
                if data_type == "train":
                    Path("wikilingua_cross_{}-en_XX".format(language)).mkdir(exist_ok=True)
                    for data_size in [10, 100, 1000, 10000]:
                        output_dir = "wikilingua_cross_{}-en_XX_{}".format(language, data_size)
                        Path(output_dir).mkdir(exist_ok=True)
                        write_to_file(encoded_texts[:data_size],
                                      "{}/{}.{}.{}".format(output_dir, data_type, new_column, lang))
                write_to_file(encoded_texts, "wikilingua_cross_{}-en_XX/{}.{}.{}".format(language, data_type, new_column, lang))
            statistics["{}-en_XX-{}".format(language, data_type)] = dataset_statistics

    for language, dataset in zip(mono_languages, datasets_monolingual):
        dataset_statistics = dict()
        dataset_statistics["samples_num"] = len(dataset["train"])
        for column, new_column in zip(columns, new_columns):
            dataset_statistics["{} average length (words)".format(new_column)] = \
                np.mean([len(sentence.split()) for sentence in dataset["train"][column]])
            encoded_column = spp.encode(dataset["train"][column], out_type=str)
            dataset_statistics["{} average length (tokens)".format(new_column)] = \
                np.mean([len(encoded_sample) for encoded_sample in encoded_column])
            encoded_texts = [" ".join(encoded_parts) for encoded_parts in encoded_column]
            Path("wikilingua_mono").mkdir(exist_ok=True)
            write_to_file(encoded_texts, "wikilingua_mono/{}.{}.{}".format("train", new_column, language))
        statistics["{}-{}-train".format(language, language)] = dataset_statistics

    statistics_df = pd.DataFrame.from_dict(statistics, orient="index")
    statistics_df.to_csv("stat_wikilingua.txt")


if __name__ == '__main__':
    main()
