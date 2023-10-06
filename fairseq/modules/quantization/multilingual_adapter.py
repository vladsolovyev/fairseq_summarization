import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm

from fairseq.modules import FairseqDropout


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
