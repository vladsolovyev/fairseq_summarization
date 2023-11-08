from pathlib import Path

import nltk
import numpy as np
import torch
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

spp = SentencePieceProcessor(model_file="mbart.cc25.v2/sentence.bpe.model")
columns = ["input_text", "summary"]
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)

lang_to_nllb = dict({"es": "spa_Latn",
                     "ru": "rus_Cyrl",
                     "gu": "guj_Gujr",
                     "tr": "tur_Latn"})


def write_to_file(lines, file):
    with open(file, "w", encoding="utf-8") as f:
        for line in lines:
            f.write("{}\n".format(line))


def encode_dataset_and_create_statistics(dataset_name, dataset, source_language, target_language, data_type,
                                         original_columns,
                                         add_few_shot=False):
    dataset_statistics = dict()
    dataset_statistics["samples_num"] = len(dataset)
    if dataset_name == "wikilingua":
        dataset_name = "{}_{}-{}".format(dataset_name, source_language, target_language)
    for original_column, new_column in zip(original_columns, columns):
        dataset_statistics["{} average length (words)".format(new_column)] = \
            np.mean([len(sentence.split()) for sentence in dataset[original_column]])
        encoded_column = spp.encode(dataset[original_column], out_type=str)
        dataset_statistics["{} average length (tokens)".format(new_column)] = \
            np.mean([len(encoded_sample) for encoded_sample in encoded_column])
        lang = source_language if original_column == "source" else target_language
        encoded_texts = [" ".join(encoded_parts) for encoded_parts in encoded_column]
        Path(dataset_name).mkdir(exist_ok=True)
        if add_few_shot and data_type == "train":
            for data_size in [10, 100, 1000, 10000]:
                if (source_language in ["gu_IN", "tr_TR"] or target_language in ["gu_IN", "tr_TR"]) \
                        and data_size == 10000:
                    continue
                output_dir = "{}_{}".format(dataset_name, data_size)
                Path(output_dir).mkdir(exist_ok=True)
                write_to_file(encoded_texts[:data_size], "{}/{}.{}.{}".format(output_dir, data_type, new_column, lang))
        write_to_file(encoded_texts, "{}/{}.{}.{}".format(dataset_name, data_type, new_column, lang))
    return dataset_statistics


def create_translated_data(input, target, directory, source_lang):
    Path(directory).mkdir(exist_ok=True)
    if source_lang != "en":
        tokenizer.src_lang = lang_to_nllb[source_lang]
        translated = list()
        for text in tqdm(input):
            sentences = nltk.sent_tokenize(text)
            inputs = tokenizer(sentences, return_tensors="pt", truncation=True, padding=True).to(device)
            translated_tokens = model.generate(**inputs,
                                               forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"],
                                               max_new_tokens=200)
            translated.append(" ".join(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)))
        input = translated
    encoded_translated_input = spp.encode(input, out_type=str)
    encoded_texts = [" ".join(encoded_parts) for encoded_parts in encoded_translated_input]
    write_to_file(encoded_texts, "{}/test.input_text.en_XX".format(directory))
    encoded_summary = spp.encode(target, out_type=str)
    encoded_texts = [" ".join(encoded_parts) for encoded_parts in encoded_summary]
    # it's actually not english, but has to have en_XX suffix for generation.
    # After generation, it will be translated into another language
    write_to_file(encoded_texts, "{}/test.summary.en_XX".format(directory))
