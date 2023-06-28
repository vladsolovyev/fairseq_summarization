from typing import Optional, Dict, List

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from fairseq.modules import TransformerDecoderLayer, LayerNorm, FairseqDropout


class AdapterTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(self, args, num_languages, lang_dict):
        super().__init__(args)

        self.use_decoder_adapter = args.use_decoder_adapter
        if self.use_decoder_adapter:
            self.lang_dict = lang_dict
            self.adapters = MultilingualAdapter(args, args.decoder_embed_dim, num_languages)

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
            tgt_lang_id=None
    ):
        x, attn, self_attn_state = super().forward(x, encoder_out, encoder_padding_mask,
                                                   incremental_state, prev_self_attn_state, prev_attn_state,
                                                   self_attn_mask, self_attn_padding_mask, need_attn, need_head_weights)

        if self.use_decoder_adapter:
            x = self.adapters(x, self.lang_dict[tgt_lang_id[0].item()])

        return x, attn, self_attn_state


class MultilingualAdapter(nn.Module):

    def __init__(self, args, input_dim, num_languages):
        super(MultilingualAdapter, self).__init__()

        self.all_adapters = torch.nn.ModuleList()

        for i in range(num_languages):
            adapter_layer_norm = LayerNorm(input_dim)
            feed_forward = FeedForward(args, input_dim, args.decoder_embed_dim // 2)
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
        x = F.relu(self.fc_1(x))
        x = self.dropout(x)
        return self.fc_2(x)
