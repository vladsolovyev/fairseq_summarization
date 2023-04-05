from torch import nn

from fairseq.models import register_model_architecture, register_model
from fairseq.models.bart import mbart_large_architecture, BARTModel
from fairseq.models.transformer import TransformerEncoder
from fairseq.modules.residual_drop_transformer_layer import ResidualDropTransformerEncoderLayer


@register_model("bart_residual_drop")
class BARTResidualDropModel(BARTModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        BARTModel.add_args(parser)
        parser.add_argument(
            "--encoder-drop-residual",
            type=int,
            help="drop residual after self-attention in this encoder layer"
        )

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return ResidualDropTransformerEncoder(args, src_dict, embed_tokens)


class ResidualDropTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        self.layers = nn.ModuleList(
            [self._build_encoder_layer(args, idx, args.encoder_drop_residual) for idx in range(args.encoder_layers)]
        )

    def _build_encoder_layer(self, args, layer_idx, encoder_drop_residual_at_layer=None):
        drop_residual_after_att = (layer_idx == encoder_drop_residual_at_layer)
        return ResidualDropTransformerEncoderLayer(args, drop_residual_after_att)


@register_model_architecture("bart_residual_drop", "mbart_large_residual_drop")
def mbart_large_residual_drop_architecture(args):
    mbart_large_architecture(args)
