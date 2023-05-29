# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq.tasks import register_task
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask


@register_task("translation_multi_simple_epoch_task_with_adversarial_loss")
class TranslationMultiSimpleEpochTaskWithAdversarialLoss(TranslationMultiSimpleEpochTask):
    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)
        assert args.encoder_langtok == "src"

    def train_step_with_classifier(
            self, sample, model, criterion, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            loss, classifier_loss, encoder_loss, sample_size, logging_output =\
                criterion(model, sample, print_predictions=(update_num % 500 == 0))
        if ignore_grad:
            loss *= 0
            classifier_loss *= 0
        return loss, classifier_loss, encoder_loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, classifier_loss, encoder_loss, sample_size, logging_output = criterion(model, sample)
        return encoder_loss, sample_size, logging_output
