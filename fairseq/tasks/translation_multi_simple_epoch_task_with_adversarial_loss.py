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

        self.language_classifier_steps = args.language_classifier_steps
        self.language_classifier_one_vs_rest = args.language_classifier_one_vs_rest
        self.use_kldivloss = args.use_kldivloss

    def classification_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        # Improve classifier to distinguish source languages
        # Based on train_step from FairseqTask
        model.train()
        model.set_num_updates(update_num)

        with torch.autograd.profiler.record_function("forward"):
            loss, classifier_loss, sample_size, logging_output = \
                criterion(model,
                          sample,
                          classification_step=True,
                          language_classifier_one_vs_rest=self.language_classifier_one_vs_rest,
                          print_predictions=(update_num % 500 == 0),
                          use_kldivloss=self.use_kldivloss)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def translation_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        # Improve translation quality and trick classifier using reversed gradients
        model.train()
        model.set_num_updates(update_num)

        with torch.autograd.profiler.record_function("forward"):
            loss, classifier_loss, sample_size, logging_output = \
                criterion(model,
                          sample,
                          classification_step=False,
                          language_classifier_one_vs_rest=self.language_classifier_one_vs_rest,
                          print_predictions=(update_num % 500 == 0),
                          use_kldivloss=self.use_kldivloss)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        # Alternate
        if (update_num + 1) % self.language_classifier_steps == 0:
            return self.translation_step(sample, model, criterion, optimizer, update_num, ignore_grad)
        else:
            return self.classification_step(sample, model, criterion, optimizer, update_num, ignore_grad)

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, classifier_loss, sample_size, logging_output = \
                criterion(model,
                          sample,
                          classification_step=True,
                          language_classifier_one_vs_rest=self.language_classifier_one_vs_rest,
                          use_kldivloss=self.use_kldivloss)
            logging_output["loss"] = classifier_loss.data
        return classifier_loss, sample_size, logging_output
