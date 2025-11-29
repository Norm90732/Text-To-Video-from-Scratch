import torch
from torch import Tensor

from model.embeddings import (
    ThreeDRope,
    TubeletPatching,
    buildRopeTable1D,
    create3DFrequencyTable3D,
)


def test_tubeletPatchingShape() -> None:
    B, C, T, H, W = 64, 3, 8, 64, 64
    patchSize = (1, 2, 2)
    embedDim = 768
    layer = TubeletPatching(
        inputChannels=C, embedDim=embedDim, patchEmbedSize=patchSize
    )
    x: Tensor = torch.randn(B, C, T, H, W)

    y = layer(x)

    tP: int = T // patchSize[0]
    hP: int = H // patchSize[1]
    wP: int = W // patchSize[2]

    expectedLen: int = tP * hP * wP

    assert y.shape == (B, expectedLen, embedDim)


def test_buildRopeTable1D() -> None:
    embedDim: int = 8
    maxSeqLength: int = 10

    halfDim: int = embedDim // 2

    angleTable: Tensor = buildRopeTable1D(
        embedDimSection=embedDim, maxSeqLength=maxSeqLength
    )

    result: Tensor = torch.randn(maxSeqLength, halfDim)

    assert angleTable.shape == result.shape


def test_applyRope3D() -> None:
    t = 4
    h = 5
    w = 10
    embedDim = 300
    layer: Tensor = create3DFrequencyTable3D(t=t, h=h, w=w, embedDim=embedDim)

    targetSeqLen = t * h * w
    target: Tensor = torch.randn(1, targetSeqLen, embedDim // 2)

    assert layer.shape == target.shape


def test_ThreeDRope():
    # Shape should match
    t = 16
    h = 400
    w = 300
    embedDim = 900
    ropeModel = ThreeDRope(t=t, h=h, w=w, embedDim=embedDim)
    videoTensor: Tensor = torch.randn(1, t * h * w, embedDim)
    output: Tensor = ropeModel.forward(videoTensor)

    assert output.shape == videoTensor.shape, "Shapes do not match"

    # Input First tensor should not be rotated
    t = 4
    h = 5
    w = 10
    embedDim = 300
    ropeModel = ThreeDRope(t=t, h=h, w=w, embedDim=embedDim)
    videoTensor: Tensor = torch.randn(1, t * h * w, embedDim)
    output: Tensor = ropeModel.forward(videoTensor)
    result = (videoTensor[0, 0, :] - output[0, 0, :]).abs().sum()
    assert result < 1e-5, "Difference in first postion is greater than 1e-5"

    # Rotation Should Preserve Magnitude
    t = 15
    h = 80
    w = 120
    embedDim = 3072
    ropeModel = ThreeDRope(t=t, h=h, w=w, embedDim=embedDim)
    videoTensor: Tensor = torch.randn(1, t * h * w, embedDim)
    output: Tensor = ropeModel.forward(videoTensor)
    # Per Token Norm
    inputTokenNorm = videoTensor.norm(p=2, dim=-1)
    outputTokenNorm = output.norm(p=2, dim=-1)

    maximumDifference = (inputTokenNorm - outputTokenNorm).abs().max()

    assert maximumDifference < 1e-4, "Norm is greater than 1e-5 difference"
