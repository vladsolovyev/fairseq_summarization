import torch
from torch import nn, tensor

from fairseq.models import register_model_architecture, register_model
from fairseq.models.bart import mbart_large_architecture, BARTModel
from fairseq.models.transformer import TransformerEncoder, TransformerDecoder
from fairseq.modules import FairseqDropout
from fairseq.modules.residual_drop_transformer_layer import ResidualDropTransformerEncoderLayer

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


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

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return ResidualDropTransformerDecoder(args, tgt_dict, embed_tokens)

    def load_state_dict(self, state_dict, strict=True, model_cfg=None, args=None, ):
        return super().load_state_dict(state_dict, strict=False)


class ResidualDropTransformerEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        self.layers = nn.ModuleList(
            [self._build_encoder_layer(args, idx, args.encoder_drop_residual) for idx in range(args.encoder_layers)]
        )

    def _build_encoder_layer(self, args, layer_idx, encoder_drop_residual_at_layer=None):
        drop_residual_after_att = (layer_idx == encoder_drop_residual_at_layer)
        return ResidualDropTransformerEncoderLayer(args, drop_residual_after_att)


class ResidualDropTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.lang_dict = dict({250004: tensor(0).to(device), 250005: tensor(1).to(device),
                               250009: tensor(2).to(device), 250021: tensor(3).to(device)})
        self.language_embeddings = nn.Embedding(4, args.encoder_embed_dim)
        self.language_embeddings_encoder_output = nn.Embedding(4, args.encoder_embed_dim)
        self.fc_language_embeddings = nn.ModuleList(
            [nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
             nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
             nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim),
             nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)])
        self.dropout = FairseqDropout(args.dropout, module_name=self.__class__.__name__)


@register_model_architecture("bart_residual_drop", "mbart_large_residual_drop")
def mbart_large_residual_drop_architecture(args):
    mbart_large_architecture(args)
