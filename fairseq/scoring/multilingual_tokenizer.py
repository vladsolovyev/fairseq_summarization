import string

from nltk import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from rouge_score.tokenizers import Tokenizer

languages = dict({"en_XX": "english",
                  "es_XX": "spanish",
                  "ru_RU": "russian"})


class MultilingualTokenizer(Tokenizer):
    def __init__(self, language="en_XX", use_stemmer=False, to_lowercase=False):
        self.use_stemmer = use_stemmer
        self.to_lowercase = to_lowercase
        self.stemmer = SnowballStemmer(languages[language]) if use_stemmer else None

    def tokenize(self, text):
        words = word_tokenize(text)
        words = [word for word in words if word not in string.punctuation]
        if self.to_lowercase:
            words = [word.lower() for word in words]
        if self.use_stemmer:
            words = [self.stemmer.stem(word) for word in words]
        return words
