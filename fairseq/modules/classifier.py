import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules.fairseq_dropout import FairseqDropout


class ClassificationLayer(nn.Module):
    def __init__(self, args, input_dim, middle_dim, output_dim):
        super(ClassificationLayer, self).__init__()
        self.fc_1 = nn.Linear(input_dim, middle_dim)
        self.fc_2 = nn.Linear(middle_dim, output_dim)
        self.dropout = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = self.dropout(x)
        return self.fc_2(x)
