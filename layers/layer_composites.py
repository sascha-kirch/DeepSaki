import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from DeepSaki import layers
from DeepSaki import initializer
from DeepSaki.layers import helper

class Conv2DSplitted(tf.keras.layers.Layer):
  '''
  To decrease the number of parameters, a convolution with the kernel_size (kernel,kernel) can be splitted into two consecutive convolutions with the kernel_size (kernel,1) and (1,kernel) respectivly
  args:
    - filters: number of filters in the output feature map 
    - kernels: size of the convolutions kernels, which will be translated to (kernels, 1) & (1,kernels) for the first and seccond convolution respectivly
    - useSpecNorm (optional, default: False): applies spectral normalization to convolutional  layers
    - strides (optional, default: (1,1)): stride of the filter
    - use_bias (optional, default: True): determines whether convolutions layers include a bias or not
    - kernel_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the convolutions kernels.
  '''
  def __init__(self,
                filters, 
                kernels,
                useSpecNorm = False, 
                strides = (1,1),
                use_bias = True,
                kernel_initializer = DeepSaki.initializer.HeAlphaUniform()
               ):
    super(Conv2DSplitted, self).__init__()
    self.filters = filters 
    self.kernels = kernels 
    self.useSpecNorm = useSpecNorm
    self.strides = strides 
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer

    self.conv1 = tf.keras.layers.Conv2D(filters = filters, kernel_size = (1, kernels), kernel_initializer = kernel_initializer,use_bias = use_bias,strides=strides)
    self.conv2 = tf.keras.layers.Conv2D(filters = filters, kernel_size = (kernels, 1),kernel_initializer = kernel_initializer,use_bias = use_bias,strides=strides)

    if useSpecNorm:
      self.conv1 = tfa.layers.SpectralNormalization(self.conv1)
      self.conv2 = tfa.layers.SpectralNormalization(self.conv2)

  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.conv2(x)
    return x

  def get_config(self):
    config = super(Conv2DSplitted, self).get_config()
    config.update({
        "filters":self.filters,
        "kernels":self.kernels,
        "useSpecNorm":self.useSpecNorm,
        "strides":self.strides, 
        "use_bias":self.use_bias,
        "kernel_initializer":self.kernel_initializer,
        })
    return config

class Conv2DBlock(tf.keras.layers.Layer):
  '''
  Wraps a two-dimensional convolution into a more complex building block
  args:
    - filters: number of filters in the output feature map
    - kernels: size of the convolutions kernels
    - useResidualConv2DBlock (optional, default: False):
    - split_kernels (optional, default: False): to decrease the number of parameters, a convolution with the kernel_size (kernel,kernel) can be splitted into two consecutive convolutions with the kernel_size (kernel,1) and (1,kernel) respectivly
    - numberOfConvs (optional, default: 1): number of consecutive convolutional building blocks, i.e. Conv2DBlock.
    - activation (optional, default: "leaky_relu"): string literal to obtain activation function
    - dropout_rate (optional, default: 0): probability of the dropout layer. If the preceeding layer has more than one channel, spatial dropout is applied, otherwise standard dropout 
    - final_activation (optional, default: True): whether or not to activate the output of this layer
    - useSpecNorm (optional, default: False): applies spectral normalization to convolutional and dense layers
    - strides (optional, default: (1,1)): stride of the filter
    - padding (optional, default: "none"): padding type. Options are "none", "zero" or "reflection"
    - applyFinalNormalization (optional, default: True): Whether or not to place a normalization on the layer's output
    - use_bias (optional, default: True): determines whether convolutions and dense layers include a bias or not
    - kernel_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the convolutions kernels.
    - gamma_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the normalization layers.
  '''
  def __init__(self,
                filters,
                kernels,  
                useResidualConv2DBlock = False, 
                split_kernels = False, 
                numberOfConvs = 1, 
                activation = "leaky_relu", 
                dropout_rate=0, 
                final_activation = True, 
                useSpecNorm = False, 
                strides = (1,1), 
                padding = "zero", 
                applyFinalNormalization = True,
                use_bias = True,
                kernel_initializer = DeepSaki.initializer.HeAlphaUniform()
                gamma_initializer = DeepSaki.initializer.HeAlphaUniform()
               ):
    super(Conv2DBlock, self).__init__()
    self.filters = filters 
    self.useResidualConv2DBlock = useResidualConv2DBlock 
    self.kernels = kernels 
    self.split_kernels = split_kernels 
    self.numberOfConvs = numberOfConvs
    self.activation = activation 
    self.dropout_rate= dropout_rate 
    self.final_activation = final_activation 
    self.useSpecNorm = useSpecNorm
    self.strides = strides 
    self.padding = padding 
    self.applyFinalNormalization = applyFinalNormalization
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.gamma_initializer = gamma_initializer

    self.pad = int((kernels-1)/2) # assumes odd kernel size, which is typical!

    if split_kernels:
        self.convs = [Conv2DSplitted(filters = filters, kernels = kernels,use_bias = use_bias,strides=strides,useSpecNorm=useSpecNorm) for _ in range(numberOfConvs)]
    else:  
      if useSpecNorm:
        self.convs = [tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(filters = filters, kernel_size = (kernels, kernels),kernel_initializer = kernel_initializer,use_bias = use_bias,strides=strides)) for _ in range(numberOfConvs)]
      else:
        self.convs = [tf.keras.layers.Conv2D(filters = filters, kernel_size = (kernels, kernels),kernel_initializer = kernel_initializer,use_bias = use_bias,strides=strides) for _ in range(numberOfConvs)]

    if applyFinalNormalization:
      num_instancenorm_blocks = numberOfConvs 
    else:
      num_instancenorm_blocks = numberOfConvs - 1
    self.IN_blocks = [tfa.layers.InstanceNormalization(gamma_initializer = gamma_initializer) for _ in range(num_instancenorm_blocks)]
    self.dropout = DeepSaki.layers.helper.dropout_func(filters, dropout_rate)

  def build(self, input_shape):
    super(Conv2DBlock, self).build(input_shape)
    #print("Model built with shape: {}".format(input_shape))
    self.residualConv = None
    if input_shape[-1] != self.filters and self.useResidualConv2DBlock:
      # split kernels for kernel_size = 1 not required
      self.residualConv = tf.keras.layers.Conv2D(filters = self.filters, kernel_size = 1,kernel_initializer = self.kernel_initializer, use_bias = self.use_bias, strides=self.strides)
      if self.useSpecNorm:
        self.residualConv = tfa.layers.SpectralNormalization(self.residualConv)

  def call(self, inputs):
    if not self.built:
      raise ValueError('This model has not yet been built.')
    x = inputs

    for block in range(self.numberOfConvs):
      residual = x

      if self.pad != 0 and self.padding != "none":
        x = DeepSaki.layers.helper.pad_func(padValues = (self.pad,self.pad), padding=self.padding)(x)
      x = self.convs[block](x)

      if self.useResidualConv2DBlock:
        if block == 0 and self.residualConv != None: #after the first conf, the channel depth matches between input and output
          residual = self.residualConv(residual)
        x = tf.keras.layers.Add()([x, residual]) 

      if block != (self.numberOfConvs - 1) or self.applyFinalNormalization:
        x = self.IN_blocks[block](x)

      if block != (self.numberOfConvs - 1) or self.final_activation:
        x = DeepSaki.layers.helper.activation_func(self.activation)(x)
    
    if self.dropout_rate > 0:
      x = self.dropout(x)

    return x

  def get_config(self):
    config = super(Conv2DBlock, self).get_config()
    config.update({
        "filters":self.filters,
        "useResidualConv2DBlock":self.useResidualConv2DBlock,
        "kernels":self.kernels,
        "split_kernels":self.split_kernels,
        "numberOfConvs":self.numberOfConvs,
        "activation":self.activation,
        "dropout_rate":self.dropout_rate,
        "final_activation":self.final_activation,
        "useSpecNorm":self.useSpecNorm,
        "strides":self.strides, 
        "padding":self.padding,
        "applyFinalNormalization":self.applyFinalNormalization,
        "use_bias":self.use_bias,
        "pad":self.pad,
        "kernel_initializer":self.kernel_initializer,
        "gamma_initializer":self.gamma_initializer
        })
    return config
    
#testcode
#layer = Conv2DBlock(filters = 128, useResidualConv2DBlock = True, kernels = 3, split_kernels  = True, useSpecNorm = True, numberOfConvs = 3, activation = "relu", dropout_rate =0.1, final_activation = False, applyFinalNormalization = True)
#print(layer.get_config())
#DeepSaki.layers.helper.PlotLayer(layer,inputShape=(256,256,64))
