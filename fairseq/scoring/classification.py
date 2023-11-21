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
        self.classifications = None
        self.samples_number = 0
        self.scores = dict()

    def add_classification_out(self, classification_out, src_pad_idx):
        classifications = F.softmax(classification_out.float(), dim=-1)[~src_pad_idx]
        self.samples_number += len(classifications)
        if self.classifications is None:
            self.classifications = classifications.sum(axis=0).cpu()
        else:
            self.classifications = np.add(self.classifications, classifications.sum(axis=0).cpu())

    def score(self, order=4):
        return 0.0

    def result_string(self, order=4):
        scores = np.round(self.classifications / self.samples_number, 4)
        for language, score in zip(languages, scores):
            self.scores[language] = score
        return "Encoder output classification: {}".format(self.scores)
