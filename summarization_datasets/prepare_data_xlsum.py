from pathlib import Path

import pandas as pd
from datasets import load_dataset
from easynmt import EasyNMT
from sentencepiece import SentencePieceProcessor

from summarization_datasets.utils import write_to_file, encode_dataset_and_create_statistics

data_types = ["train", "validation", "test"]
columns = ["text", "target"]
languages = ["en_XX", "es_XX", "ru_RU", "gu_IN"]
datasets = [load_dataset("GEM/xlsum", "english", cache_dir="./cache"),
            load_dataset("GEM/xlsum", "spanish", cache_dir="./cache"),
            load_dataset("GEM/xlsum", "russian", cache_dir="./cache"),
            load_dataset("GEM/xlsum", "gujarati", cache_dir="./cache")]
spp = SentencePieceProcessor(model_file="mbart.cc25.v2/sentence.bpe.model")
translation_model = EasyNMT("mbart50_m2en", cache_folder="./cache")


def create_translated_data(dataset, source_lang):
    directory = "xlsum_{}_en".format(source_lang)
    Path(directory).mkdir(exist_ok=True)
    translated_input_text = translation_model.translate(dataset["test"]["text"],
                                                        source_lang=source_lang,
                                                        target_lang="en",
                                                        show_progress_bar=True)
    encoded_translated_input = spp.encode(translated_input_text, out_type=str)
    encoded_texts = [" ".join(encoded_parts) for encoded_parts in encoded_translated_input]
    write_to_file(encoded_texts, "{}/test.input_text.en_XX".format(directory))
    encoded_summary = spp.encode(dataset["test"]["target"], out_type=str)
    encoded_texts = [" ".join(encoded_parts) for encoded_parts in encoded_summary]
    # it's actually not english, but has to have en_XX suffix for generation.
    # After generation, it will be translated into spanish or russian.
    write_to_file(encoded_texts, "{}/test.summary.en_XX".format(directory))


def filter_datasets():
    for dataset in datasets:
        dataset["train"] = dataset["train"].filter(
            lambda sample: len(sample["text"].split()) > 20 and len(sample["target"].split()) > 10)


def main():
    filter_datasets()
    for dataset, lang in zip(datasets[1:], ["es", "ru", "gu"]):
        create_translated_data(dataset, lang)
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
