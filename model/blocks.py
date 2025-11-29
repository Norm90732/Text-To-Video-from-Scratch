import torch
import torch.nn as nn
from beartype import beartype
from einops import rearrange
from jaxtyping import Float, jaxtyped
from torch import Tensor


# Time step Embedding
class TimeStepEmbedding(nn.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(self, embedDim: int) -> None:
        super().__init__()
        self.halfDim: int = embedDim // 2
        i: Tensor = torch.arange(0, self.halfDim, step=1, dtype=torch.float32)
        invFrequencies: Tensor = 10000 ** ((-2 * i) / embedDim)
        self.register_buffer("invFreqCache", invFrequencies)
        # B, embedDim//2

    @jaxtyped(typechecker=beartype)
    def forward(self, t: Float[Tensor, "batch"]) -> Float[Tensor, "batch embedDim"]:
        # T is broadcast to B, 1
        t = rearrange(t, "b -> b 1")
        angleMatrix: Tensor = torch.mul(t, torch.as_tensor(self.invFreqCache))
        # Make a Sin and Cos now
        Cos: Tensor = torch.cos(angleMatrix)  # PE(t,2i)
        Sin: Tensor = torch.sin(angleMatrix)  # Pe(t,2i+1)
        concat: Tensor = torch.cat([Cos, Sin], dim=-1)  # B, E

        return concat


# Combining Text and Time Step Conditioning
class ConditioningMLP(nn.Module):
    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        tEmbedDim: int,
        textEmbedDim: int,
        hiddenDimMultiplier: int,
        conditioningDim: int,
    ) -> None:
        super().__init__()
        self.baseDim = tEmbedDim + textEmbedDim
        self.MLP = nn.Sequential(
            nn.Linear(self.baseDim, hiddenDimMultiplier * self.baseDim),
            nn.GELU(),
            nn.Linear(self.baseDim * hiddenDimMultiplier, conditioningDim),
        )

    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        tEmbed: Float[Tensor, "batch timeEmbed"],
        textEmbed: Float[Tensor, "batch textEmbed"],
    ) -> Float[Tensor, "batch conditioningDim"]:
        concat: Tensor = torch.cat([tEmbed, textEmbed], dim=-1)  # B, E

        result = self.MLP(concat)

        return result  # B, E


# Adaln

# Attention

# FeedForward
