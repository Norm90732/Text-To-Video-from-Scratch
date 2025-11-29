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
    Uses standard Conv3D for patch embedding to create temporal batches of latent vector
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
    embedDimSection: int, maxSeqLength: int
) -> Float[Tensor, "maxSeqLen halfDim"]:
    """
    Creates The Position and Angle Matrices
    """
    halfDim: int = embedDimSection // 2
    frequencyGen: Tensor = (-8 * halfDim / embedDimSection) * torch.log(
        torch.tensor(10.0)
    )  # numerical stability idea I think works
    frequencyTableLog: Tensor = torch.linspace(0, frequencyGen, steps=halfDim)
    frequencyTable: Tensor = torch.exp(frequencyTableLog)
    positionTable: Tensor = torch.arange(0, maxSeqLength, dtype=torch.float32)
    angleTable: Tensor = torch.outer(positionTable, frequencyTable)

    return angleTable


@jaxtyped(typechecker=beartype)
def create3DFrequencyTable3D(
    t: int, h: int, w: int, embedDim: int
) -> Float[Tensor, "batch maxSeqLen halfDim"]:
    """
    maxSeqLen is a T*H*W tensor, and halfDim*3 is result.
    """
    tSection: int = embedDim // 3
    hSection: int = embedDim // 3
    wSection: int = embedDim - tSection - hSection

    angleTableT: Tensor = buildRopeTable1D(
        embedDimSection=tSection, maxSeqLength=t
    ).unsqueeze(0)  # Batch, maxSeqLen, halfdim
    angleTableH: Tensor = buildRopeTable1D(
        embedDimSection=hSection, maxSeqLength=h
    ).unsqueeze(0)  # Batch maxSeqLen, halfdim
    angleTableW: Tensor = buildRopeTable1D(
        embedDimSection=wSection, maxSeqLength=w
    ).unsqueeze(0)  # Batch, maxSeqLen, halfdim

    angleTableT: Tensor = repeat(
        angleTableT,
        "batch maxSeqlenTime halfdim -> batch maxSeqlenTime H W halfdim",
        H=h,
        W=w,
    )
    angleTableT: Tensor = rearrange(
        angleTableT,
        "batch maxSeqlenTime H W halfdim -> batch (maxSeqlenTime H W) halfdim ",
    )

    angleTableH: Tensor = repeat(
        angleTableH,
        "batch maxSeqlenHeight halfdim -> batch T maxSeqlenHeight W halfdim",
        T=t,
        W=w,
    )
    angleTableH: Tensor = rearrange(
        angleTableH,
        "batch T maxSeqlenHeight W halfdim -> batch (T maxSeqlenHeight W) halfdim",
    )

    angleTableW: Tensor = repeat(
        angleTableW,
        "batch maxSeqlenWidth halfdim -> batch T H maxSeqlenWidth halfdim",
        T=t,
        H=h,
    )
    angleTableW: Tensor = rearrange(
        angleTableW,
        "batch T H maxSeqlenWidth halfdim -> batch (T H maxSeqlenWidth) halfdim",
    )

    # B, (T*H*W), halfDim*3
    concatTable: Tensor = torch.cat([angleTableT, angleTableH, angleTableW], dim=-1)
    return concatTable


class ThreeDRope(nn.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(self, t: int, h: int, w: int, embedDim: int) -> None:
        """
        Class to apply Rope for Attention Blocks.
        """
        super().__init__()
        angleTable: Tensor = create3DFrequencyTable3D(t=t, h=h, w=w, embedDim=embedDim)
        self.register_buffer("angleTableCache", angleTable)

    @jaxtyped(typechecker=beartype)
    def forward(
        self, latentVideo: Float[Tensor, "batch seq_len embedDim"]
    ) -> Float[Tensor, "batch maxSeqLen embedDim"]:
        pairLatent: Tensor = rearrange(
            latentVideo, "batch seq_len (embedDim p) -> batch seq_len embedDim p", p=2
        )

        pairLatent = torch.view_as_complex(pairLatent)
        rotatorMatrix: Tensor = torch.polar(
            abs=torch.ones_like(torch.as_tensor(self.angleTableCache)),
            angle=torch.as_tensor(self.angleTableCache),
        )

        rotated: Tensor = torch.mul(pairLatent, rotatorMatrix)

        result: Tensor = torch.view_as_real(rotated)  # Batch, seqLen, dim//2, 2
        result: Tensor = rearrange(
            result, "batch seq_len embedDim p -> batch seq_len (embedDim p)", p=2
        )

        return result
