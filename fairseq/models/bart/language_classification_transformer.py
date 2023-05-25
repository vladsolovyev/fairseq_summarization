# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch

from fairseq.models import register_model, register_model_architecture
from fairseq.models.bart import AdapterTransformerModel, AdapterTransformerEncoder, mbart_large_architecture
from fairseq.modules.classifier import ClassificationLayer


@register_model("language_classification_transformer")
class LanguageProbingTransformerModel(AdapterTransformerModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        AdapterTransformerModel.add_args(parser)
        parser.add_argument(
            "--classifier-middle-layer-size",
            default=256,
            type=int,
            help="TBA",
        )
        parser.add_argument(
            "--num-language-to-classify",
            required=True,
            type=int,
            help="TBA",
        )
        parser.add_argument(
            "--grad-reversal-scaling-factor",
            default=1.0,
            type=float,
            help="TBA",
        )
        parser.add_argument(
            "--language-classifier-steps",
            default=2,
            type=int,
            help="Do this many updates before each translation loss update",
        )

        parser.add_argument(
            "--language-classifier-one-vs-rest",
            default=0,
            type=int,
            help="If non-zero, language classification will be binary (this class vs rest)",
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return LanguageProbingTransformerEncoder(args, src_dict, embed_tokens)

    def load_state_dict(
            self,
            state_dict,
            strict=True,
            model_cfg=None,
            args=None,
    ):
        # Setting strict to False due to newly added parameters
        return super().load_state_dict(state_dict, strict=False)


class LanguageProbingTransformerEncoder(AdapterTransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        # +1 due to padding
        output_dim = args.num_language_to_classify + 1 if args.language_classifier_one_vs_rest == 0 else 3
        self.language_classifier = ClassificationLayer(args=args,
                                                       input_dim=args.encoder_embed_dim,
                                                       middle_dim=args.classifier_middle_layer_size,
                                                       output_dim=output_dim)

    def forward_scriptable(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        enc_out_dict = super().forward_scriptable(src_tokens,
                                                  src_lengths,
                                                  return_all_hiddens,
                                                  token_embeddings)
        # Add encoder and classification outputs
        enc_out = enc_out_dict["encoder_out"][0]   # T x B x C
        lang_classifier_out = self.language_classifier(enc_out)     # T x B x num_lan
        enc_out_dict["classification_out"] = lang_classifier_out

        return enc_out_dict


@register_model_architecture("language_classification_transformer", "language_classification_transformer")
def language_classification_transformer(args):
    mbart_large_architecture(args)