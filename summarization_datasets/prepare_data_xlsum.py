from pathlib import Path

from datasets import load_dataset
from sentencepiece import SentencePieceProcessor

from summarization_datasets.utils import write_to_file

data_types = ["train", "test"]
columns = ["text", "target"]
new_columns = ["input_text", "summary"]
languages = ["en_XX", "es_XX", "ru_RU"]
datasets = [load_dataset("GEM/xlsum", "english", cache_dir="./cache"),
            load_dataset("GEM/xlsum", "spanish", cache_dir="./cache"),
            load_dataset("GEM/xlsum", "russian", cache_dir="./cache")]
spp = SentencePieceProcessor(model_file="mbart.cc25.v2/sentence.bpe.model")


def filter_datasets():
    for dataset in datasets:
        dataset["train"] = dataset["train"].filter(
            lambda sample: len(sample["text"].split()) > 20 and len(sample["target"].split()) > 10)


def main():
    filter_datasets()
    for language, dataset in zip(languages, datasets):
        for data_type in data_types:
            for column, new_column in zip(columns, new_columns):
                encoded_texts = [" ".join(encoded_parts) for encoded_parts in
                                 spp.encode(dataset[data_type][column], out_type=str)]
                if data_type == "train":
                    Path("xlsum").mkdir(exist_ok=True)
                    for data_size in [10, 100, 1000, 10000]:
                        output_dir = "xlsum_{}".format(data_size)
                        Path(output_dir).mkdir(exist_ok=True)
                        write_to_file(encoded_texts[:data_size],
                                      "{}/{}.{}.{}".format(output_dir, data_type, new_column, language))
                write_to_file(encoded_texts, "xlsum/{}.{}.{}".format(data_type, new_column, language))


if __name__ == '__main__':
    main()
