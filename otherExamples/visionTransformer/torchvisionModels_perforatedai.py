import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn

from torchvision.ops.misc import Conv2dNormActivation, MLP
from torchvision.transforms._presets import ImageClassification, InterpolationMode
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface


__all__ = [
    "VisionTransformer",
    "ViT_B_16_Weights",
    "ViT_B_32_Weights",
    "ViT_L_16_Weights",
    "ViT_L_32_Weights",
    "ViT_H_14_Weights",
    "vit_b_16",
    "vit_b_32",
    "vit_l_16",
    "vit_l_32",
    "vit_h_14",
]


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., nn.Module] = nn.BatchNorm2d
    activation_layer: Callable[..., nn.Module] = nn.ReLU


class MLPBlock(MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        originalEncoderBlock,
        lastOne,
        originalEncoderNorm
    ):
        super().__init__()
        self.num_heads = originalEncoderBlock.num_heads

        # Attention block
        self.ln_1 = originalEncoderBlock.ln_1
        self.self_attention = originalEncoderBlock.self_attention
        self.dropout = originalEncoderBlock.dropout

        # MLP block
        self.ln_2 = originalEncoderBlock.ln_2
        self.mlp = originalEncoderBlock.mlp
        self.extra_norm = lastOne
        if(self.extra_norm):
            self.ln3 = originalEncoderNorm

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        if(self.extra_norm):
            self.ln3(x+y)
        else:
            return x + y
    

class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        originalEncoder
    ):
        super().__init__()
        self.pos_embedding = originalEncoder.pos_embedding
        self.dropout = originalEncoder.dropout
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        num_layers = len(originalEncoder.layers)
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                originalEncoder.layers[i],
                lastOne = (1 == num_layers-1),
                originalEncoderNorm = originalEncoder.ln)
        self.layers = nn.Sequential(layers)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        #return self.ln(self.layers(self.dropout(input)))
        return self.layers(self.dropout(input))


class VisionTransformer(nn.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        originalVisionTransformer,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.image_size = originalVisionTransformer.image_size
        self.patch_size = originalVisionTransformer.patch_size
        self.hidden_dim = originalVisionTransformer.hidden_dim
        self.mlp_dim = originalVisionTransformer.mlp_dim
        self.attention_dropout = originalVisionTransformer.attention_dropout
        self.dropout = originalVisionTransformer.dropout
        self.num_classes = originalVisionTransformer.num_classes
        self.representation_size = originalVisionTransformer.representation_size
        self.norm_layer = originalVisionTransformer.norm_layer
        self.conv_proj = originalVisionTransformer.conv_proj
        self.class_token = originalVisionTransformer.class_token
        self.encoder = Encoder(originalVisionTransformer.encoder)
        self.seq_length = originalVisionTransformer.seq_length
        self.heads = originalVisionTransformer.heads

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x
    
def interpolate_embeddings(
    image_size: int,
    patch_size: int,
    model_state: "OrderedDict[str, torch.Tensor]",
    interpolation_mode: str = "bicubic",
    reset_heads: bool = False,
) -> "OrderedDict[str, torch.Tensor]":
    """This function helps interpolate positional embeddings during checkpoint loading,
    especially when you want to apply a pre-trained model on images with different resolution.

    Args:
        image_size (int): Image size of the new model.
        patch_size (int): Patch size of the new model.
        model_state (OrderedDict[str, torch.Tensor]): State dict of the pre-trained model.
        interpolation_mode (str): The algorithm used for upsampling. Default: bicubic.
        reset_heads (bool): If true, not copying the state of heads. Default: False.

    Returns:
        OrderedDict[str, torch.Tensor]: A state dict which can be loaded into the new model.
    """
    # Shape of pos_embedding is (1, seq_length, hidden_dim)
    pos_embedding = model_state["encoder.pos_embedding"]
    n, seq_length, hidden_dim = pos_embedding.shape
    if n != 1:
        raise ValueError(f"Unexpected position embedding shape: {pos_embedding.shape}")

    new_seq_length = (image_size // patch_size) ** 2 + 1

    # Need to interpolate the weights for the position embedding.
    # We do this by reshaping the positions embeddings to a 2d grid, performing
    # an interpolation in the (h, w) space and then reshaping back to a 1d grid.
    if new_seq_length != seq_length:
        # The class token embedding shouldn't be interpolated, so we split it up.
        seq_length -= 1
        new_seq_length -= 1
        pos_embedding_token = pos_embedding[:, :1, :]
        pos_embedding_img = pos_embedding[:, 1:, :]

        # (1, seq_length, hidden_dim) -> (1, hidden_dim, seq_length)
        pos_embedding_img = pos_embedding_img.permute(0, 2, 1)
        seq_length_1d = int(math.sqrt(seq_length))
        if seq_length_1d * seq_length_1d != seq_length:
            raise ValueError(
                f"seq_length is not a perfect square! Instead got seq_length_1d * seq_length_1d = {seq_length_1d * seq_length_1d } and seq_length = {seq_length}"
            )

        # (1, hidden_dim, seq_length) -> (1, hidden_dim, seq_l_1d, seq_l_1d)
        pos_embedding_img = pos_embedding_img.reshape(1, hidden_dim, seq_length_1d, seq_length_1d)
        new_seq_length_1d = image_size // patch_size

        # Perform interpolation.
        # (1, hidden_dim, seq_l_1d, seq_l_1d) -> (1, hidden_dim, new_seq_l_1d, new_seq_l_1d)
        new_pos_embedding_img = nn.functional.interpolate(
            pos_embedding_img,
            size=new_seq_length_1d,
            mode=interpolation_mode,
            align_corners=True,
        )

        # (1, hidden_dim, new_seq_l_1d, new_seq_l_1d) -> (1, hidden_dim, new_seq_length)
        new_pos_embedding_img = new_pos_embedding_img.reshape(1, hidden_dim, new_seq_length)

        # (1, hidden_dim, new_seq_length) -> (1, new_seq_length, hidden_dim)
        new_pos_embedding_img = new_pos_embedding_img.permute(0, 2, 1)
        new_pos_embedding = torch.cat([pos_embedding_token, new_pos_embedding_img], dim=1)

        model_state["encoder.pos_embedding"] = new_pos_embedding

        if reset_heads:
            model_state_copy: "OrderedDict[str, torch.Tensor]" = OrderedDict()
            for k, v in model_state.items():
                if not k.startswith("heads"):
                    model_state_copy[k] = v
            model_state = model_state_copy

    return model_state
