# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import shutil
from dataclasses import dataclass, field

import evaluate
import nltk
import numpy as np
from easynmt import EasyNMT
from langid.langid import LanguageIdentifier, model
from rouge_score import rouge_scorer

from fairseq.dataclass import FairseqDataclass
from fairseq.scoring import BaseScorer, register_scorer

nltk.download("stopwords")
translation_model = EasyNMT("mbart50_en2m")
translation_to_mbart_language = dict({"es": "es_XX",
                                      "ru": "ru_RU",
                                      "gu": "gu_IN"})
mbart_lang_to_rouge_lang = dict({"en_XX": "english",
                                 "es_XX": "spanish",
                                 "ru_RU": "russian",
                                 "gu_IN": "gujarati",
                                 "tr_TR": "turkish"})
languages = ["en", "es", "ru", "gu", "tr"]


@dataclass
class RougeBertScoreScorerConfig(FairseqDataclass):
    lang: str = field(default="en_XX", metadata={"help": "Language"})
    translate_to_lang: str = field(default="", metadata={"help": "translation language"})
    rouge_scorer: str = field(default="huggingface", metadata={"help": "rouge scorer implementation"})


@register_scorer("rougebert", dataclass=RougeBertScoreScorerConfig)
class RougeBertScoreScorer(BaseScorer):
    def __init__(self, cfg):
        super(RougeBertScoreScorer, self).__init__(cfg)
        self.cfg = cfg
        self.scores = None

    def add_string(self, ref, pred):
        self.ref.append(ref)
        self.pred.append(pred)

    def score(self, order=4):
        return 0.0

    def result_string(self, order=4):
        return f"BERTScore: {self.rouge_and_bert_score()}"

    def calculate_rouge_scores(self):
        rouge_result_with_stemming = dict()
        rouge_result_without_stemming = dict()
        if self.cfg.rouge_scorer == "huggingface":
            shutil.rmtree("./cache")
            rouge_result_with_stemming = evaluate.load("rouge", cache_dir="./cache").compute(predictions=self.pred,
                                                                                             references=self.ref,
                                                                                             use_stemmer=True)
            shutil.rmtree("./cache")
            rouge_result_without_stemming = evaluate.load("rouge", cache_dir="./cache").compute(predictions=self.pred,
                                                                                                references=self.ref,
                                                                                                use_stemmer=False)
            shutil.rmtree("./cache")
        elif self.cfg.rouge_scorer == "multilingual":
            rouge_result_with_stemming = self.calculate_multilingual_rouge_scores(use_stemmer=True)
            rouge_result_without_stemming = self.calculate_multilingual_rouge_scores(use_stemmer=False)
        rouge_result_with_stemming = {"{}_with_stemming".format(k): v for k, v in
                                      rouge_result_with_stemming.items()}
        rouge_result_without_stemming = {"{}_without_stemming".format(k): v for k, v in
                                         rouge_result_without_stemming.items()}
        return rouge_result_with_stemming | rouge_result_without_stemming

    def calculate_multilingual_rouge_scores(self, use_stemmer=False):
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"],
                                          use_stemmer=use_stemmer,
                                          lang=mbart_lang_to_rouge_lang[self.cfg.lang])
        scores_all_samples = [scorer.score(pred, ref) for pred, ref in zip(self.pred, self.ref)]
        return {metric: np.mean([score_per_sample[metric] for score_per_sample in scores_all_samples])
                for metric in scores_all_samples[0]}

    def calculate_language_probabilities(self):
        identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        identifier.set_languages(languages)
        results = [dict(identifier.rank(summary)) for summary in self.pred]
        return {"{}_prob".format(language): np.mean([result[language] for result in results]) for language in languages}

    def calculate_bert_score(self):
        shutil.rmtree("./cache")
        bert_result = evaluate.load("bertscore", cache_dir="./cache").compute(predictions=self.pred,
                                                                              references=self.ref,
                                                                              model_type="bert-base-multilingual-cased")
        shutil.rmtree("./cache")
        if bert_result["hashcode"]:
            del bert_result["hashcode"]
        return {"bert_score_{}".format(k): np.mean(v) for k, v in bert_result.items()}

    def rouge_and_bert_score(self):
        print("number of samples: {}".format(len(self.pred)))
        if self.cfg.translate_to_lang in languages[1:4]:
            self.cfg.lang = translation_to_mbart_language[self.cfg.translate_to_lang]
            self.pred = translation_model.translate(self.pred,
                                                    source_lang="en",
                                                    target_lang=self.cfg.translate_to_lang,
                                                    show_progress_bar=True)
        results = self.calculate_rouge_scores() | self.calculate_bert_score() | self.calculate_language_probabilities()
        results = {key: value * 100 for key, value in results.items()}
        results["gen_len"] = np.mean([len(sentence.split()) for sentence in self.pred])
        self.scores = {k: round(v, 4) for k, v in results.items()}
