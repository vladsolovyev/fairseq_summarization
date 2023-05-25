# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from fairseq import metrics
import warnings


@register_task("multilingual_translation_similarity")
class MultilingualTranslationSimilarityTask(TranslationMultiSimpleEpochTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        TranslationMultiSimpleEpochTask.add_args(parser)

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        if not any("nsentences" in log for log in logging_outputs):
            warnings.warn(
                "nsentences not found in Criterion logging outputs, cannot log bsz and similarity loss"
            )
        else:
            nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)

            if any("similarity_loss" in log for log in logging_outputs):
                similarity_loss = sum(log.get("similarity_loss", 0) for log in logging_outputs) / nsentences
                metrics.log_scalar("similarity_loss", similarity_loss, priority=200, round=1)

        criterion.__class__.reduce_metrics(logging_outputs)