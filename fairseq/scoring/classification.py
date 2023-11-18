from dataclasses import dataclass

import numpy as np
import torch.nn.functional as F

from fairseq.dataclass import FairseqDataclass
from fairseq.scoring import BaseScorer, register_scorer

languages = ["en", "es", "ru", "tr"]


@dataclass
class ClassificationScorerConfig(FairseqDataclass):
    pass


@register_scorer("classification", dataclass=ClassificationScorerConfig)
class ClassificationScorer(BaseScorer):
    def __init__(self, cfg):
        super(ClassificationScorer, self).__init__(cfg)
        self.cfg = cfg
        self.classifications = list()
        self.scores = dict()

    def add_classification_out(self, classification_out, src_pad_idx):
        classification_out = F.softmax(classification_out.float(), dim=-1)
        self.classifications.extend(classification_out[~src_pad_idx].tolist())

    def score(self, order=4):
        return 0.0

    def result_string(self, order=4):
        classifications = np.array(self.classifications)
        scores = np.round(classifications.mean(axis=0), 4)
        for language, score in zip(languages, scores):
            self.scores[language] = score
        return "Encoder output classification: {}".format(self.scores)
