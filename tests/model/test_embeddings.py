import torch

from model.embeddings import TubeletPatching


def test_tubeletPatchingShape():
    B, C, T, H, W = 64, 3, 8, 64, 64
    patchSize = (1, 2, 2)
    embedDim = 768
    layer = TubeletPatching(
        inputChannels=C, embedDim=embedDim, patchEmbedSize=patchSize
    )
    x = torch.randn(B, C, T, H, W)

    y = layer(x)

    tP = T // patchSize[0]
    hP = H // patchSize[1]
    wP = W // patchSize[2]

    expectedLen = tP * hP * wP

    assert y.shape == (B, expectedLen, embedDim)
