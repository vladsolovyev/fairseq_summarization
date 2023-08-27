from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sentencepiece import SentencePieceProcessor

from summarization_datasets.utils import write_to_file

languages = ["en", "es", "ru", "tr"]
languages_mbart = ["en_XX", "es_XX", "ru_RU", "tr_TR"]
spp = SentencePieceProcessor(model_file="mbart.cc25.v2/sentence.bpe.model")


def map_datasets(mono_dataset, cross_dataset):
    input_texts = dict(zip(mono_dataset["source"], mono_dataset["target"]))
    input_texts_keys = set(input_texts.keys())
    summaries_input = [input_texts[cross_sample["source"]] for cross_sample in cross_dataset if cross_sample["source"] in input_texts_keys]
    summaries_output = [cross_sample["target"] for cross_sample in cross_dataset if cross_sample["source"] in input_texts_keys]
    summaries = dict(zip(mono_dataset["target"], mono_dataset["source"]))
    summaries_keys = set(summaries.keys())
    input_texts_input = [summaries[cross_sample["target"]] for cross_sample in cross_dataset if cross_sample["target"] in summaries_keys]
    input_texts_output = [cross_sample["source"] for cross_sample in cross_dataset if cross_sample["target"] in summaries_keys]
    return summaries_input + input_texts_input, summaries_output + input_texts_output


def parse_datasets(mono_dataset, cross_dataset):
    train_mono_dataset = mono_dataset["train"]
    train_cross_dataset = cross_dataset["train"]
    train_input, train_output = map_datasets(train_mono_dataset, train_cross_dataset)
    validation_mono_dataset = mono_dataset["sampled_validation"]
    validation_cross_dataset = cross_dataset["sampled_validation"]
    validation_input, validation_output = map_datasets(validation_mono_dataset, validation_cross_dataset)
    return train_input, train_output, validation_input, validation_output


def main():
    statistics = dict()
    for i in range(len(languages))[:-1]:
        for k in range(len(languages))[i + 1:]:
            mono_dataset = load_dataset("GEM/wiki_lingua", languages[i], cache_dir="./cache")
            cross_dataset = load_dataset("GEM/wiki_lingua", name="{}_{}".format(languages[k], languages[i]),
                                         cache_dir="./cache")
            train_input, train_output, validation_input, validation_output = parse_datasets(mono_dataset, cross_dataset)
            print("{}-{}-{}-{}".format(languages[i], languages[k], len(train_input), len(validation_input)))
            statistics["{}-{}_train".format(languages[i], languages[k])] = len(train_input)
            dataset = load_dataset("ted_talks_iwslt", language_pair=(languages[i], languages[k]),
                                   year="2016", cache_dir="./cache")
            train_data = pd.DataFrame(dataset["train"]["translation"]).to_dict(orient="list")
            train_input += train_data[languages[i]]
            train_output += train_data[languages[k]]
            print("{}-{}-{}-{}".format(languages[i], languages[k], len(train_input), len(validation_input)))
            statistics["{}-{}_train_with_ted".format(languages[i], languages[k])] = len(train_input)
            statistics["{}-{}_valid".format(languages[i], languages[k])] = len(validation_input)
            create_translated_data(train_input, train_output, "train",
                                   languages_mbart[i], languages_mbart[k])
            create_translated_data(validation_input, validation_output, "validation",
                                   languages_mbart[i], languages_mbart[k])
    statistics_df = pd.DataFrame.from_dict(statistics, orient="index", columns=["number"])
    statistics_df.to_csv("stat_nmt.txt")


def create_translated_data(input, target, data_type, source_lang, target_lang):
    directory = "translated/{}_{}".format(source_lang, target_lang)
    Path(directory).mkdir(exist_ok=True, parents=True)
    encoded_input = [" ".join(encoded_parts) for encoded_parts in spp.encode(input, out_type=str)]
    write_to_file(encoded_input, "{}/{}.input_text.{}".format(directory, data_type, source_lang))
    # it's actually a translation, not a summary, but the task is implemented now in this way
    write_to_file(encoded_input, "{}/{}.summary.{}".format(directory, data_type, source_lang))
    encoded_output = [" ".join(encoded_parts) for encoded_parts in spp.encode(target, out_type=str)]
    write_to_file(encoded_output, "{}/{}.summary.{}".format(directory, data_type, target_lang))
    # it's actually a translation, not a summary, but the task is implemented now in this way
    write_to_file(encoded_output, "{}/{}.input_text.{}".format(directory, data_type, target_lang))


if __name__ == "__main__":
    main()
