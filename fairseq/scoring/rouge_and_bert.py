# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import evaluate
import numpy as np

from fairseq.dataclass import FairseqDataclass
from fairseq.scoring import BaseScorer, register_scorer


#check if it is possible to delete it later
@dataclass
class RougeBertScoreScorerConfig(FairseqDataclass):
    bert_score_lang: str = field(default="en", metadata={"help": "BERTScore language"})


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

    def rouge_and_bert_score(self):
        rouge_result = evaluate.load("rouge").compute(predictions=self.pred,
                                                      references=self.ref,
                                                      use_stemmer=True)
        bert_result = evaluate.load("bertscore").compute(predictions=self.pred,
                                                         references=self.ref,
                                                         model_type="bert-base-multilingual-cased")
        if bert_result["hashcode"]:
            del bert_result["hashcode"]
        bert_result = {"bert_score_{}".format(k): np.mean(v) for k, v in bert_result.items()}
        results = rouge_result | bert_result
        results = {key: value * 100 for key, value in results.items()}
        results["gen_len"] = np.mean([len(sentence.split()) for sentence in self.pred])
        self.scores = {k: round(v, 4) for k, v in results.items()}
