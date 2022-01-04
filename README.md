# DeepSaki
Welcome AI enthuisiasts to DeepSaki, a collection reusable machine learning code. :muscle::robot::metal:

The ML framework used is tensorflow and the entire code is suitable to run Google's TPUs.

![GitHub](https://img.shields.io/github/license/saki1309/deepsaki)

# Installation

## Git
```
git clone https://github.com/SaKi1309/DeepSaki.git
```

## Pip
![PyPI](https://img.shields.io/pypi/v/deepsaki)
![PyPI - Status](https://img.shields.io/pypi/status/deepsaki)
![PyPI - Downloads](https://img.shields.io/pypi/dm/deepsaki?label=downloads%20pip)
```
pip install DeepSaki
```

# Content
- initializer
  - HeAlphaNormal
  - HeAlphaUniform
- layers
  - GlobalSumPooling2D
  - ReflectionPadding (suitable for TPU)
  - Conv2DBlock
  - Conv2DSplitted
  - DenseBlock
  - DownSampleBlock
  - UpSampleBlock
  - ResidualIdentityBlock
  - ResBlockDown
  - ResBlockUp
  - ScaleLayer
  - ScalarGatedSelfAttention
  - Encoder
  - Bottleneck
  - Decoder
  - helper
    - GetInitializer
    - pad_func
    - dropout_func
    - PlotLayer
- loss
  - PixelDistanceLoss
  - StructuralSimilarityLoss
- models
  - LayoutContentDiscriminator 
  - PatchDiscriminator
  - ResNet
  - UNet
  - UNetDiscriminator
- optimizer
  - SWATS_ADAM
  - SWATS_NADAM
- regularization
  - CutMix
  - CutOut 
  - GetMask
- utils
  - DetectHw
  - EnableXlaAcceleration
  - EnableMixedPrecision
