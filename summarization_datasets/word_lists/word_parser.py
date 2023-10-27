import string

from sentencepiece import SentencePieceProcessor

spp = SentencePieceProcessor(model_file="../mbart.cc25.v2/sentence.bpe.model")
masked_labels = [0] * (spp.vocab_size() - 3)

symbols = "{}{}{}".format(string.punctuation, string.digits, string.whitespace)
encoded_sample_ids = spp.encode(symbols)
for id in encoded_sample_ids:
    masked_labels[id - 3] = 1
for symbol in symbols:
    encoded_sample_ids = spp.encode(symbol)
    for id in encoded_sample_ids:
        masked_labels[id - 3] = 1


with open("turkish.txt", "r", encoding="utf-8") as file:
    line = file.readline()
    for word in line.split(","):
        encoded_sample_ids = spp.encode(word)
        for id in encoded_sample_ids:
            masked_labels[id - 3] = 1


with open("turkish_2.txt", "r") as file:
    for line in file:
        encoded_sample_ids = spp.encode(line)
        for id in encoded_sample_ids:
            masked_labels[id - 3] = 1

with open("mask_tr_TR.txt", "w") as file:
    for id in masked_labels:
        file.write("{}\n".format(id))