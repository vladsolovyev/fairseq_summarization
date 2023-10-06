from typing import Optional, Dict, List

import torch
from torch import Tensor

from fairseq.modules import TransformerDecoderLayer
from fairseq.modules.quantization.multilingual_adapter import MultilingualAdapter


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
