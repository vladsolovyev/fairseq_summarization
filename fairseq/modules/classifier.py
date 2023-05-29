# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn

from fairseq.modules.fairseq_dropout import FairseqDropout


class ClassificationLayer(nn.Module):
    def __init__(self, args, input_dim, middle_dim, output_dim):
        super(ClassificationLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, middle_dim)
        self.fc2 = nn.Linear(middle_dim, output_dim)
        self.dropout = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.clone_input = args.use_language_embeddings_encoder_output

    def forward(self, x):
        if self.clone_input:
            x = x.clone()
        if self.training:
            x = grad_reverse(x, self.grad_reversal_scaling_factor)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
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
