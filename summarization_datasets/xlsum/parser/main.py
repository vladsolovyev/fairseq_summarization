from datasets import load_dataset
from sentencepiece import SentencePieceProcessor

data_types = ["train", "test", "validation"]
new_data_types = ["train", "test", "valid"]
columns = ["text", "target"]
new_columns = ["input_text", "summary"]
languages = ["en", "es", "ru"]
datasets = [load_dataset("GEM/xlsum", "english"),
            load_dataset("GEM/xlsum", "spanish"),
            load_dataset("GEM/xlsum", "russian")]


def main():
    spp = SentencePieceProcessor(model_file="../../mbart.cc25.v2/sentence.bpe.model")
    for data_type, new_data_type in zip(data_types, new_data_types):
        for column, new_column in zip(columns, new_columns):
            for language, dataset in zip(languages, datasets):
                encoded_texts = [" ".join(encoded_parts) for encoded_parts in spp.encode(dataset[data_type][column], out_type=str)]
                with open("{}.{}.{}".format(new_data_type, new_column, language), "w", encoding="utf-8") as f:
                    for encoded_text in encoded_texts:
                        f.write("{}\n".format(encoded_text))


if __name__ == '__main__':
    main()
