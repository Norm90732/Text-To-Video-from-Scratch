import torch
import torch.nn as nn
from beartype import beartype
from einops import rearrange, repeat
from jaxtyping import Float, jaxtyped
from torch import Tensor

"""
Implementation of 3-D Rope Embeddings and Tubelet Patching

"""


class TubeletPatching(nn.Module):
    """
    Uses standard Conv3D for patch embedding to create a sequence of latent vector
    """

    @jaxtyped(typechecker=beartype)
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
    ) -> Float[Tensor, "batch SeqLen dim"]:
        x = self.patchEmbed(latentVideo)

        x: Tensor = rearrange(x, "b c t h w -> b (t h w) c")

        return x


# https://arxiv.org/pdf/2104.09864
@jaxtyped(typechecker=beartype)
def buildRopeTable1D(
    headDim: int, maxSeqLength: int
) -> Float[Tensor, "maxSeqLen halfDim"]:
    """
    Creates The Position and Angle Matrices
    """
    halfDim: int = headDim // 2
    frequencyGen: Tensor = (-8 * halfDim / headDim) * torch.log(
        torch.tensor(10.0)
    )  # numerical stability idea I think works
    frequencyTableLog: Tensor = torch.linspace(0, frequencyGen, steps=halfDim)
    frequencyTable: Tensor = torch.exp(frequencyTableLog)
    positionTable: Tensor = torch.arange(0, maxSeqLength, dtype=torch.float32)
    angleTable: Tensor = torch.outer(positionTable, frequencyTable)

    return angleTable


@jaxtyped(typechecker=beartype)
def create3DFrequencyTable3D(
    t: int, h: int, w: int, headDim: int
) -> Float[Tensor, "maxSeqLen halfDim"]:
    """
    maxSeqLen is a T*H*W tensor
    """

    angleTableT: Tensor = buildRopeTable1D(
        headDim=headDim, maxSeqLength=t
    )  # maxSeqLen, halfdim
    angleTableH: Tensor = buildRopeTable1D(
        headDim=headDim, maxSeqLength=h
    )  # maxSeqLen, halfdim
    angleTableW: Tensor = buildRopeTable1D(
        headDim=headDim, maxSeqLength=w
    )  # maxSeqLen, halfdim

    angleTableT: Tensor = repeat(
        angleTableT,
        "maxSeqlenTime halfdim -> maxSeqlenTime 1 1 halfdim",
    )

    angleTableH: Tensor = repeat(
        angleTableH,
        "maxSeqlenHeight halfdim -> 1 maxSeqlenHeight 1 halfdim",
    )

    angleTableW: Tensor = repeat(
        angleTableW,
        "maxSeqlenWidth halfdim -> 1 1 maxSeqlenWidth halfdim",
    )

    totalTable: Tensor = angleTableT + angleTableH + angleTableW

    concatTable: Tensor = rearrange(totalTable, "t h w halfdim -> (t h w) halfdim")

    return concatTable


class ThreeDRope(nn.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(self, t: int, h: int, w: int, headDim: int) -> None:
        """
        Class to apply Rope for Attention Blocks.
        """
        super().__init__()
        angleTable: Tensor = create3DFrequencyTable3D(t=t, h=h, w=w, headDim=headDim)
        self.register_buffer("angleTableCache", angleTable, persistent=False)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, latentVideo: Float[Tensor, "batch seq_len numHeads headDim"]
    ) -> Float[Tensor, "batch seq_len numHeads headDim"]:
        pairLatent: Tensor = rearrange(
            latentVideo,
            "batch seq_len numHeads (headDim p) -> batch seq_len numHeads headDim p",
            p=2,
        )

        pairLatent = torch.view_as_complex(pairLatent)

        angles = self.angleTableCache
        angles = rearrange(torch.as_tensor(angles), "seq halfDim -> 1 seq 1 halfDim")
        rotatorMatrix: Tensor = torch.polar(
            abs=torch.ones_like(angles),
            angle=angles,
        )

        rotated: Tensor = torch.mul(pairLatent, rotatorMatrix)

        result: Tensor = torch.view_as_real(rotated)  # Batch, seqLen, heads, dim//2, 2
        result: Tensor = rearrange(
            result,
            "batch seq_len numHeads headDim p -> batch seq_len numHeads (headDim p)",
            p=2,
        )

        return result
