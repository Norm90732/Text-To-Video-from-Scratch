import torch.nn as nn
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor

"""
Implementation of 2-D and 1-D Rope Embeddings and Tubelet Patching
"""


class TubeletPatching(nn.Module):
    def __init__(
        self,
        inputChannels: int,
        embedDim: int,
        patchEmbedSize: tuple[int, int, int] = (1, 2, 2),
    ) -> None:
        super().__init__()

        self.patchEmbedSize = patchEmbedSize

        self.patchEmbed = nn.Conv3d(
            in_channels=inputChannels,
            out_channels=embedDim,
            kernel_size=patchEmbedSize,
            stride=patchEmbedSize,
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self, latentVideo: Float[Tensor, "batch channels time height width"]
    ) -> Float[Tensor, "batch seq_len embed_dim"]:
        x = self.patchEmbed(latentVideo)

        x: Tensor = rearrange(x, "b c t h w -> b (t h w) c")

        return x
