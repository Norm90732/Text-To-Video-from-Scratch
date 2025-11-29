import torch
from einops import repeat
from torch import Tensor

from model.blocks import ConditioningMLP, TimeStepEmbedding


def test_TimeStepEmbed() -> None:
    embedDim = 50
    model = TimeStepEmbedding(embedDim)
    t: Tensor = torch.randn(4)
    output: Tensor = model.forward(t)
    t = repeat(t, "b -> b e", e=embedDim)
    assert output.shape == t.shape, "Not the same size"


def test_ConditioningMLP() -> None:
    tEmbedDim = 200
    textEmbedDim = 1024
    hiddenDimMultiplier = 2
    conditioningDim = 1000
    batchSize = 20
    model = ConditioningMLP(
        tEmbedDim=tEmbedDim,
        textEmbedDim=textEmbedDim,
        hiddenDimMultiplier=hiddenDimMultiplier,
        conditioningDim=conditioningDim,
    )
    tEmbed = torch.randn(batchSize, tEmbedDim)
    textEmbed = torch.randn(batchSize, textEmbedDim)
    target = torch.rand(batchSize, conditioningDim)
    output = model.forward(tEmbed, textEmbed)

    assert target.shape == output.shape, "Target and Output Shapes do not match"
