from pathlib import Path

from datasets import load_dataset
from sentencepiece import SentencePieceProcessor

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


def write_to_file(lines, file):
    with open(file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write("{}\n".format(line))


def main():
    for language, dataset in zip(crosslingual_source_languages, datasets_crosslingual):
        for data_type in data_types:
            for column, new_column in zip(columns, new_columns):
                lang = language if column == "source" else "en_XX"
                encoded_texts = [" ".join(encoded_parts) for encoded_parts in
                                 spp.encode(dataset[data_type][column], out_type=str)]
                if data_type == "train":
                    Path("wikilingua_cross_{}-en_XX".format(language)).mkdir(exist_ok=True)
                    for data_size in [10, 100, 1000, 10000]:
                        output_dir = "wikilingua_cross_{}-en_XX_{}".format(language, data_size)
                        Path(output_dir).mkdir(exist_ok=True)
                        write_to_file(encoded_texts[:data_size],
                                      "{}/{}.{}.{}".format(output_dir, data_type, new_column, lang))
                write_to_file(encoded_texts, "wikilingua_cross_{}-en_XX/{}.{}.{}".format(language, data_type, new_column, lang))
    for language, dataset in zip(mono_languages, datasets_monolingual):
        for column, new_column in zip(columns, new_columns):
            encoded_texts = [" ".join(encoded_parts) for encoded_parts in
                             spp.encode(dataset[data_type][column], out_type=str)]
            Path("wikilingua_mono").mkdir(exist_ok=True)
            write_to_file(encoded_texts, "wikilingua_mono/{}.{}.{}".format("train", new_column, language))


if __name__ == '__main__':
    main()
