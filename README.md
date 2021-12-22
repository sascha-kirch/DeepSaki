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
- layer
  - GlobalSumPooling2D
  - ReflectionPadding (suitable for TPU)
  - Conv2DBlock
  - Conv2DSplitted
  - helper
    - GetInitializer
    - pad_func
    - activation_func
    - dropout_func
    - PlotLayer
- loss
  - PixelDistanceLoss
  - StructuralSimilarityLoss
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
