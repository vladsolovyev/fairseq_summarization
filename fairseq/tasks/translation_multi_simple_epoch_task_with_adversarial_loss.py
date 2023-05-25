# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.tasks import register_task
import torch

from fairseq.tasks.multilingual_translation_similarity_task import MultilingualTranslationSimilarityTask


@register_task("translation_multi_simple_epoch_task_with_adversarial_loss")
class MultilingualTranslationAdversarial(MultilingualTranslationSimilarityTask):
    def __init__(self, args, langs, dicts, training):
        super().__init__(args, langs, dicts, training)
        assert args.encoder_langtok == "src"

        self.language_classifier_steps = args.language_classifier_steps
        self.language_classifier_one_vs_rest = args.language_classifier_one_vs_rest

    def build_model(self, args):
        model = super().build_model(args)
        print("Encoder type", type(model.encoder))
        print("Decoder type", type(model.decoder))

        return model

    def classification_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        # Improve classifier to distinguish source languages
        # Based on train_step from FairseqTask
        model.train()
        model.set_num_updates(update_num)

        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = \
                criterion(model,
                          sample,
                          classification_step=True,
                          language_classifier_one_vs_rest=self.language_classifier_one_vs_rest)
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
            loss, sample_size, logging_output = \
                criterion(model,
                          sample,
                          classification_step=False,
                          language_classifier_one_vs_rest=self.language_classifier_one_vs_rest)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    #

    def train_step(
            self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        # Alternate
        if update_num % self.language_classifier_steps == 0:
            return self.translation_step(sample, model, criterion, optimizer, update_num, ignore_grad)
        else:
            return self.classification_step(sample, model, criterion, optimizer, update_num, ignore_grad)

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output = \
                criterion(model,
                          sample,
                          classification_step=True,
                          language_classifier_one_vs_rest=self.language_classifier_one_vs_rest)
        return loss, sample_size, logging_output
