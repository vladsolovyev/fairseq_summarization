# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq.modules import LayerNorm
from fairseq.modules import TransformerEncoderLayer, TransformerDecoderLayer
from fairseq.modules.fairseq_dropout import FairseqDropout


class AdapterTransformerEncoderLayer(TransformerEncoderLayer):
    """Encoder layer block. TODO: comment
    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.
    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, drop_residual_after_att=False):
        super().__init__(args)
        self.lang_idx = None

        if args.encoder_adapter:
            self.adapters = MultilingualAdapter(args, args.encoder_embed_dim)
            self.drop_adapters_for_inference = args.drop_adapters_for_inference
        else:
            self.adapters = None

        print('****************************', drop_residual_after_att)
        self.drop_residual_after_att = drop_residual_after_att


    def set_lang_idx(self, lang_idx):
        self.lang_idx = lang_idx


    def activate_adapters(self):
        # Freeze all original parameters
        if self.adapters is not None:
            for name, child in (self.named_children()):

                if isinstance(child, MultilingualAdapter):
                    for p_name, param in child.named_parameters():
                        param.requires_grad = True
                else:
                    if "layer_norm" in name: #isinstance(child, FusedLayerNorm):
                        child.eval()



    def forward(self, x, encoder_padding_mask: Optional[Tensor], attn_mask: Optional[Tensor] = None, lang = 0):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)

        if not self.drop_residual_after_att:
            x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

#	x = super().forward(x, encoder_padding_mask, attn_mask)

        if self.adapters is None:
            return x

        if not self.training and self.drop_adapters_for_inference:
            return x

        x = self.adapters(x, 0)

        return x


class AdapterTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, args):
        super().__init__(args)

        if args.decoder_adapter:
            self.adapters = MultilingualAdapter(args, args.decoder_embed_dim)
            self.drop_adapters_for_inference = args.drop_adapters_for_inference
        else:
            self.adapters = None

    def activate_adapters(self):
        # Freeze all original parameters
        if self.adapters is not None:
            for name, child in (self.named_children()):
                if isinstance(child, MultilingualAdapter):
                    for p_name, param in child.named_parameters():
                        param.requires_grad = True
                else:
                    if "layer_norm" in name: ##isinstance(child, FusedLayerNorm):
                        child.eval()


    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        x, attn, self_attn_state = super().forward(x, encoder_out, encoder_padding_mask,
                incremental_state, prev_self_attn_state, prev_attn_state,
                self_attn_mask, self_attn_padding_mask, need_attn, need_head_weights)

        if self.adapters is None:
            return x, attn, self_attn_state

        if not self.training and self.drop_adapters_for_inference:
            return x, attn, self_attn_state

        x = self.adapters(x, 0)

        return x, attn, self_attn_state



class MultilingualAdapter(nn.Module):

    def __init__(self, args, input_dim):
        super(MultilingualAdapter, self).__init__()

        self.all_adapters = torch.nn.ModuleList()
        bottleneck_dim = args.bottleneck_dim

        for i in range(args.num_src_lang):
            adapter_layer_norm = LayerNorm(input_dim)
            feed_forward = FeedForward(args, input_dim, bottleneck_dim)
            adapter = nn.Sequential(adapter_layer_norm, feed_forward)

            self.all_adapters.append(adapter)


    def forward(self, x, lang_idx):
        adapter = self.all_adapters[lang_idx]

        return x + adapter(x)


class FeedForward(nn.Module):
    def __init__(self, args, input_dim, bottleneck_dim):
        super(FeedForward, self).__init__()
        self.fc_1 = nn.Linear(input_dim, bottleneck_dim)
        self.fc_2 = nn.Linear(bottleneck_dim, input_dim)
        self.dropout = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )

    def forward(self, x):
        x = F.relu(self.fc_1(x), inplace=True)
        x = self.dropout(x)
        x = self.fc_2(x)
        return x