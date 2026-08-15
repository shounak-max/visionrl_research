"""
Encoders package initialization.
"""

from encoders.base_encoder import BaseEncoder
from encoders.resnet_encoder import FrozenResNet18Encoder, PartialFineTuneResNet18Encoder
from encoders.cnn_encoder import EndToEndCNNEncoder


def make_encoder(encoder_id: str, embedding_dim: int = 512) -> BaseEncoder:
    """Factory function to build perception encoders.

    Supported encoder_ids:
    - 'frozen_resnet18', 'resnet18' -> FrozenResNet18Encoder
    - 'partial_resnet18' -> PartialFineTuneResNet18Encoder
    - 'end2end_cnn', 'cnn' -> EndToEndCNNEncoder
    """
    if encoder_id in ["frozen_resnet18", "resnet18"]:
        return FrozenResNet18Encoder(embedding_dim=embedding_dim)
    elif encoder_id == "partial_resnet18":
        return PartialFineTuneResNet18Encoder(embedding_dim=embedding_dim)
    elif encoder_id in ["end2end_cnn", "cnn"]:
        return EndToEndCNNEncoder(embedding_dim=embedding_dim)
    else:
        raise ValueError(f"Unknown encoder_id: '{encoder_id}'")
