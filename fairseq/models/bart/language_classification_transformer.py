# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Dict, List

import torch
from torch import Tensor

from fairseq.models import register_model, register_model_architecture
from fairseq.models.bart import mbart_large_architecture, BARTResidualDropModel, ResidualDropTransformerEncoder, \
    ResidualDropTransformerDecoder
from fairseq.modules.classifier import ClassificationLayer


@register_model("language_classification_transformer")
class LanguageClassificationTransformerModel(BARTResidualDropModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        BARTResidualDropModel.add_args(parser)
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
            default=-1,
            type=int,
            help="If not -1, language classification will be binary (this class vs rest)",
        )

        parser.add_argument(
            "--use-kldivloss",
            action="store_true", help="Use kldivloss for adversarial loss", default=False
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return LanguageClassificationTransformerEncoder(args, src_dict, embed_tokens, cls.lang_dict())

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return LanguageClassificationTransformerDecoder(args, tgt_dict, embed_tokens, cls.lang_dict())


class LanguageClassificationTransformerEncoder(ResidualDropTransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens, lang_dict):
        super().__init__(args, dictionary, embed_tokens, lang_dict)

        output_dim = args.num_language_to_classify if args.language_classifier_one_vs_rest == -1 else 2
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
        enc_out_dict = super().forward_scriptable(src_tokens, src_lengths, return_all_hiddens, token_embeddings)
        # Add encoder and classification outputs
        enc_out = enc_out_dict["encoder_out"][0]  # T x B x C
        lang_classifier_out = self.language_classifier(enc_out)  # T x B x num_lan
        enc_out_dict["classification_out"] = lang_classifier_out
        return enc_out_dict


class LanguageClassificationTransformerDecoder(ResidualDropTransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, lang_dict):
        super().__init__(args, dictionary, embed_tokens, lang_dict)

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            tgt_lang_id=None
    ):
        x, extra = super().extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
            tgt_lang_id=tgt_lang_id
        )
        # Additionally return encoder output
        extra["encoder_out"] = encoder_out["encoder_out"]
        extra["encoder_padding_mask"] = encoder_out["encoder_padding_mask"]
        if "classification_out" in encoder_out:
            extra["classification_out"] = encoder_out["classification_out"]
        return x, extra


@register_model_architecture("language_classification_transformer", "language_classification_transformer")
def language_classification_transformer(args):
    mbart_large_architecture(args)
