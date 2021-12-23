# DeepSaki
Welcome AI enthuisiasts to DeepSaki, a collection reusable machine learning code. :muscle::robot::metal:

The ML framework used is tensorflow and the entire code is suitable to run Google's TPUs.

# Installation

## Git
```
!git clone https://github.com/SaKi1309/DeepSaki.git
```

## Pip
not supported yet

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
