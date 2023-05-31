# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import tensor
from torch.nn import KLDivLoss

from fairseq import metrics
from fairseq import utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import label_smoothed_nll_loss, LabelSmoothedCrossEntropyCriterion

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
lang_dict = dict({250004: tensor(0).to(device), 250005: tensor(1).to(device), 250021: tensor(2).to(device)})


@register_criterion("language_classification_cross_entropy")
class LanguageClassificationCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )

    def forward(self, model, sample, reduce=True, classification_step=False, language_classifier_one_vs_rest=-1, print_predictions=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        ####################################################
        # sample has the following keys:
        # id
        # nsentences
        # ntokens   --> number of total target tokens
        # net_input
        # src_tokens    --> source token indices
        # src_lengths   --> source length tensor (for each instance)
        # prev_output_tokens --> target (indices) shifted left
        # target --> target token (indices)
        ####################################################
        """
        # 1) forward pass for src -> tgt
        net_output = model(**sample["net_input"])
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        if classification_step:
            with torch.no_grad():
                # This loss won't take part of backward
                loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

            logging_output = {
                "loss": loss.data,
                "nll_loss": nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }

            classifier_loss, classifier_nll_loss, n_correct, total, stats_per_lang = \
                self.compute_encoder_classification_loss(sample["net_input"],
                                                         net_output,
                                                         classification_step=True,
                                                         language_classifier_one_vs_rest=language_classifier_one_vs_rest,
                                                         print_predictions=print_predictions)

            logging_output["classifier_loss"] = classifier_loss.data
            logging_output["classifier_nll_loss"] = classifier_nll_loss.data

            if self.report_accuracy:
                logging_output["n_correct"] = utils.item(n_correct.data)
                logging_output["total"] = utils.item(total.data)
                logging_output["per_lang_n_correct"] = defaultdict(int)
                logging_output["per_lang_total"] = defaultdict(int)

                if len(stats_per_lang) > 0:
                    for lang in stats_per_lang:
                        logging_output["per_lang_n_correct"][lang] += stats_per_lang[lang][0]
                        logging_output["per_lang_total"][lang] += stats_per_lang[lang][1]

            return classifier_loss, sample_size, logging_output

        else:
            # Translation step, together with fooling classifier
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            logging_output = {
                "loss": loss.data,
                "nll_loss": nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }

            classifier_loss, classifier_nll_loss, _, _, _ = \
                self.compute_encoder_classification_loss(sample["net_input"],
                                                         net_output,
                                                         classification_step=False,
                                                         language_classifier_one_vs_rest=language_classifier_one_vs_rest,
                                                         print_predictions=print_predictions)
            loss += classifier_loss
            logging_output["classifier_loss"] = classifier_loss.data
            logging_output["classifier_nll_loss"] = classifier_nll_loss.data

            return loss, sample_size, logging_output

    def compute_encoder_classification_loss(self,
                                            net_input,
                                            net_output,
                                            reduce=True,
                                            classification_step=True,
                                            language_classifier_one_vs_rest=-1,
                                            print_predictions=False):
        encoder_classification_out = net_output[1]["classification_out"]
        max_len, batch_size, _ = encoder_classification_out.shape
        lprobs = F.log_softmax(encoder_classification_out.float(), dim=-1)
        target = tensor([lang_dict[x.item()] for x in net_input["src_lang_id"]]).to(device)
        if print_predictions:
            print("Target: {}".format(target))
            print("Predictions: {}".format(torch.mean(lprobs, 0)))
        target = target.repeat(max_len, 1)
        src_pad_idx = net_input["src_tokens"].eq(self.padding_idx).transpose(0, 1)
        if language_classifier_one_vs_rest > -1:  # Change target to binary
            src_one_lang_idx = target == language_classifier_one_vs_rest
            target[src_one_lang_idx] = 0
            target[~src_one_lang_idx] = 1
        lprobs, target, src_pad_idx = lprobs.view(-1, lprobs.size(-1)), target.view(-1), src_pad_idx.contiguous().view(-1)
        lprobs = lprobs[~src_pad_idx]
        target = target[~src_pad_idx]

        if classification_step:
            loss, nll_loss = label_smoothed_nll_loss(lprobs, target, self.eps, reduce=reduce)
        else:
            equal_probabilities = tensor(1 / len(lang_dict)).repeat(len(lang_dict))
            equal_probabilities = F.log_softmax(equal_probabilities, -1)
            target_equal_probabilities = equal_probabilities.repeat(len(lprobs), 1).to(device)
            loss = KLDivLoss(reduction="sum", log_target=True)(lprobs, target_equal_probabilities)
            nll_loss = loss

        stats_per_lang = {}
        unique_targets = torch.unique(target, sorted=True)

        for id in unique_targets:
            mask = target.eq(id)
            n_correct = torch.sum(
                lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
            )
            n_total = torch.sum(mask)
            stats_per_lang[utils.item(id.data)] = [utils.item(n_correct.data), utils.item(n_total.data)]

        # Calc overall accuracy
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        n_total = torch.sum(mask)

        return loss, nll_loss, n_correct, n_total, stats_per_lang

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        classifier_loss_sum = sum(log.get("classifier_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        metrics.log_scalar(
            "classifier_loss", classifier_loss_sum / ntokens / math.log(2), ntokens, round=3
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

        t = logging_outputs[0].get("per_lang_n_correct", None)
        if t is not None:
            for lang_id in t:
                n_correct = sum(log.get("per_lang_n_correct").get(lang_id, 0) for log in logging_outputs)
                n_total = sum(log.get("per_lang_total").get(lang_id, 0) for log in logging_outputs)
                metrics.log_scalar(
                    f"accuracy_lang_{lang_id}",
                    round(
                        n_correct * 100.0 / n_total, 3
                    ) if n_total > 0 else float("nan"),
                    priority=100,
                )
                metrics.log_scalar(
                    f"n_total_lang_{lang_id}",
                    n_total,
                    priority=100,
                )

    @staticmethod
    def add_args(parser):
        super(
            LanguageClassificationCrossEntropyCriterion,
            LanguageClassificationCrossEntropyCriterion,
        ).add_args(parser)
        # fmt: off

        """Add criterion-specific arguments to the parser."""
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )

        parser.add_argument(
            "--report-accuracy",
            action="store_true",
            help="report accuracy metric",
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return False
