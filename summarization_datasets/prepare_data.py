from pathlib import Path

from datasets import load_dataset
from sentencepiece import SentencePieceProcessor

data_types = ["train", "test", "validation"]
new_data_types = ["train", "test", "valid"]
columns = ["text", "target"]
new_columns = ["input_text", "summary"]
languages = ["en_XX", "es_XX", "ru_RU"]
datasets = [load_dataset("GEM/xlsum", "english"),
            load_dataset("GEM/xlsum", "spanish"),
            load_dataset("GEM/xlsum", "russian")]


def filter_datasets():
    for dataset in datasets:
        for data_type in data_types:
            dataset[data_type] = dataset[data_type].filter(
                lambda sample: len(sample["text"].split()) > 20 and len(sample["target"].split()) > 10)


def main():
    filter_datasets()
    spp = SentencePieceProcessor(model_file="mbart.cc25.v2/sentence.bpe.model")
    for data_size in [-1, 10, 100, 1000, 10000]:
        for data_type, new_data_type in zip(data_types, new_data_types):
            for column, new_column in zip(columns, new_columns):
                for language, dataset in zip(languages, datasets):
                    data = dataset[data_type][column][:data_size] if data_type == "train" else dataset[data_type][
                        column]
                    encoded_texts = [" ".join(encoded_parts) for encoded_parts in spp.encode(data, out_type=str)]
                    output_dir = "xlsum" if data_size == -1 else "xlsum_{}".format(data_size)
                    Path(output_dir).mkdir(exist_ok=True)
                    with open("{}/{}.{}.{}".format(output_dir, new_data_type, new_column, language), "w",
                              encoding="utf-8") as f:
                        for encoded_text in encoded_texts:
                            f.write("{}\n".format(encoded_text))


if __name__ == '__main__':
    main()
