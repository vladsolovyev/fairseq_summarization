# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch.nn as nn
from torch import Tensor

from fairseq.models import register_model, register_model_architecture
from fairseq.models.bart import BARTModel, mbart_large_architecture
from fairseq.models.transformer import TransformerEncoder, TransformerDecoder

from fairseq.modules.adapter_transformer_layer import AdapterTransformerDecoderLayer, AdapterTransformerEncoderLayer


@register_model("adapter_transformer")
class AdapterTransformerModel(BARTModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        BARTModel.add_args(parser)
        parser.add_argument(
            "--bottleneck-dim",
            default=256,
            type=int,
            help="bottleneck size of adapter",
        )
        parser.add_argument(
            "--num-src-lang",
            default=1,
            type=int,
            help="number of unique adapters",
        )


    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return AdapterTransformerEncoder(args, src_dict, embed_tokens)


    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return AdapterTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
            )


    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg=None,
        args=None,
    ):
        # Setting strict to False due to newly added adapters
        return super().load_state_dict(state_dict, strict=False)


class AdapterTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.layers = nn.ModuleList(
            [self._build_encoder_layer(args, idx, args.encoder_drop_residual) for idx in range(args.encoder_layers)]
        )

        if args.encoder_adapter:
            for name, child in (self.named_children()):
                # Freeze everything other than the adapter
                for p_name, param in child.named_parameters():
                    param.requires_grad = False

            for l in self.layers:
                l.activate_adapters()

    def _build_encoder_layer(self, args, layer_idx, encoder_drop_residual_at_layer=None):
        drop_residual_after_att = (layer_idx == encoder_drop_residual_at_layer)
        return AdapterTransformerEncoderLayer(args, drop_residual_after_att)


class AdapterTransformerDecoder(TransformerDecoder):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, output_projection)
        self.layers = nn.ModuleList(
            [AdapterTransformerDecoderLayer(args) for idx in range(args.decoder_layers)]
        )

        if args.decoder_adapter:
            for name, child in (self.named_children()):
                # Freeze everything other than the adapter
                for p_name, param in child.named_parameters():
                    param.requires_grad = False

            for l in self.layers:
                l.activate_adapters()


    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        x, extra = super().extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        # Additionally return encoder output
        extra["encoder_out"] = encoder_out["encoder_out"]
        extra["encoder_padding_mask"] = encoder_out["encoder_padding_mask"]
        if "classification_out" in encoder_out:
            extra["classification_out"] = encoder_out["classification_out"]

        return x, extra


@register_model_architecture(
    "adapter_transformer", "adapter_transformer"
)
def adapter_transformer_architecture(args):
    mbart_large_architecture(args)