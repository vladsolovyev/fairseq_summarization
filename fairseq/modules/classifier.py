import torch
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
        self.grad_reversal_scaling_factor = args.grad_reversal_scaling_factor

    def forward(self, x):
        if self.training:   # Gradient reversal on input to classifier
            x = grad_reverse(x, self.grad_reversal_scaling_factor)

        x = F.relu(self.fc_1(x), inplace=True)
        x = self.dropout(x)
        x = self.fc_2(x)
        return x


class GradReverse(torch.autograd.Function):
    scale = 1.0

    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return GradReverse.scale * grad_output.neg()


def grad_reverse(x, scale=1.0):
    GradReverse.scale = scale
    return GradReverse.apply(x)