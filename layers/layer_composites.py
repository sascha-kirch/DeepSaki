import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import DeepSaki.layers
import DeepSaki.initializer
import DeepSaki.layers.helper

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
    - padding (optional, default: "zero"): padding type. Options are "none", "zero" or "reflection"
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
                kernel_initializer = DeepSaki.initializer.HeAlphaUniform(),
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


class DenseBlock(tf.keras.layers.Layer):
  '''
  Wraps a dense layer into a more complex building block
  args:
    - units: number of units of each dense block
    - numberOfLayers (optional, default: 1): number of consecutive convolutional building blocks, i.e. Conv2DBlock.
    - activation (optional, default: "leaky_relu"): string literal to obtain activation function
    - dropout_rate (optional, default: 0): probability of the dropout layer. If the preceeding layer has more than one channel, spatial dropout is applied, otherwise standard dropout 
    - final_activation (optional, default: True): whether or not to activate the output of this layer
    - useSpecNorm (optional, default: False): applies spectral normalization to convolutional and dense layers
    - applyFinalNormalization (optional, default: True): Whether or not to place a normalization on the layer's output
    - use_bias (optional, default: True): determines whether convolutions and dense layers include a bias or not
    - kernel_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the convolutions kernels.
    - gamma_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the normalization layers.
  '''
  def __init__(self,
               units, 
               numberOfLayers = 1, 
               activation = "leaky_relu", 
               dropout_rate=0, 
               final_activation = True, 
               useSpecNorm = False, 
               applyFinalNormalization = True, 
               use_bias = True,
               kernel_initializer = DeepSaki.initializer.HeAlphaUniform(),
               gamma_initializer =  DeepSaki.initializer.HeAlphaUniform()
               ):
    super(DenseBlock, self).__init__()
    self.units = units 
    self.numberOfLayers = numberOfLayers 
    self.dropout_rate = dropout_rate
    self.activation = activation 
    self.final_activation = final_activation 
    self.useSpecNorm = useSpecNorm
    self.applyFinalNormalization = applyFinalNormalization 
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.gamma_initializer = gamma_initializer

    if useSpecNorm:
      self.DenseBlocks = [tfa.layers.SpectralNormalization(tf.keras.layers.Dense(units=units,use_bias =use_bias,kernel_initializer = kernel_initializer)) for _ in range(numberOfLayers)]
    else:
      self.DenseBlocks = [tf.keras.layers.Dense(units=units,use_bias =use_bias,kernel_initializer = kernel_initializer) for _ in range(numberOfLayers)]
    
    if applyFinalNormalization:
      num_instancenorm_blocks = numberOfLayers 
    else:
      num_instancenorm_blocks = numberOfLayers - 1
    self.IN_blocks = [tfa.layers.InstanceNormalization(gamma_initializer = gamma_initializer) for _ in range(num_instancenorm_blocks)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, inputs):
    x = inputs

    for block in range(self.numberOfLayers):
      x = self.DenseBlocks[block](x)

      if block != (self.numberOfLayers - 1) or self.applyFinalNormalization:
        x = self.IN_blocks[block](x)

      if block != (self.numberOfLayers - 1) or self.final_activation:
        x = DeepSaki.layers.helper.activation_func(self.activation)(x)
    
    if self.dropout_rate > 0:
      x = self.dropout(x)
    return x

  def get_config(self):
    config = super(DenseBlock, self).get_config()
    config.update({
        "units":self.units,
        "numberOfLayers":self.numberOfLayers,
        "dropout_rate":self.dropout_rate,
        "activation":self.activation,
        "activation":self.activation,
        "final_activation":self.final_activation,
        "useSpecNorm":self.useSpecNorm,
        "applyFinalNormalization":self.applyFinalNormalization,
        "use_bias":self.use_bias,
        "kernel_initializer":self.kernel_initializer,
        "gamma_initializer":self.gamma_initializer
        })
    return config

#testcode
#layer = DenseBlock(units = 512, numberOfLayers = 3, activation = "leaky_relu", applyFinalNormalization=False)
#print(layer.get_config())
#DeepSaki.layers.helper.PlotLayer(layer,inputShape=(256,256,64))

class DownSampleBlock(tf.keras.layers.Layer):
  '''
   Spatial down-sampling for grid-like data
   args:
     - downsampling (optional, default: "average_pooling"): 
     - activation (optional, default: "leaky_relu"): string literal to obtain activation function
     - kernels (optional, default: 3): size of the convolution's kernels when using downsampling = "conv_stride_2"
     - useSpecNorm (optional, default: False): applies spectral normalization to convolutional and dense layers
     - padding (optional, default: "zero"): padding type. Options are "none", "zero" or "reflection"
     - use_bias (optional, default: True): determines whether convolutions and dense layers include a bias or not
     - kernel_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the convolutions kernels.
     - gamma_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the normalization layers.
  '''
  def __init__(self,
               downsampling = "average_pooling", 
               activation = "leaky_relu", 
               kernels = 3, 
               useSpecNorm = False, 
               padding = "zero",
               use_bias = True,
               kernel_initializer = DeepSaki.initializer.HeAlphaUniform(),
               gamma_initializer =  DeepSaki.initializer.HeAlphaUniform()
               ):
    super(DownSampleBlock, self).__init__()
    self.kernels = kernels 
    self.downsampling = downsampling
    self.activation = activation
    self.useSpecNorm = useSpecNorm
    self.padding = padding 
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.gamma_initializer = gamma_initializer

  def build(self, input_shape):
    super(DownSampleBlock, self).build(input_shape)

    self.layers = []
    if self.downsampling == "conv_stride_2":
      self.layers.append(DeepSaki.layers.Conv2DBlock(input_shape[-1], self.kernels, activation = self.activation, strides = (2,2), useSpecNorm=self.useSpecNorm,padding=self.padding,use_bias = self.use_bias, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer))
    elif self.downsampling == "max_pooling":
      #Only spatial downsampling, increase in features is done by the conv2D_block specified later!
      self.layers.append(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
    elif self.downsampling =="average_pooling":
      self.layers.append(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
    elif self.downsampling =="space_to_depth":
      pass
    else:
      raise Exception("Undefined downsampling provided")

  def call(self, inputs):
    x = inputs
    if self.downsampling == "space_to_depth":
      x = tf.nn.space_to_depth(x, block_size = 2)
    else:
      for layer in self.layers:
        x = layer(x)

    return x

  def get_config(self):
    config = super(DownSampleBlock, self).get_config()
    config.update({
        "kernels":self.kernels,
        "downsampling":self.downsampling,
        "activation":self.activation,
        "useSpecNorm":self.useSpecNorm,
        "padding":self.padding,
        "use_bias":self.use_bias,
        "kernel_initializer":self.kernel_initializer,
        "gamma_initializer":self.gamma_initializer
        })
    return config

#layer = DownSampleBlock(numOfChannels = 3, kernels = 3, downsampling = "conv_stride_2", activation = "leaky_relu", useSpecNorm = True)
#print(layer.get_config())
#DeepSaki.layers.helper.PlotLayer(layer,inputShape=(256,256,64))


class UpSampleBlock(tf.keras.layers.Layer):
  '''
   Spatial down-sampling for grid-like data
   args:
     - upsampling (optional, default: "2D_upsample_and_conv"): 
     - activation (optional, default: "leaky_relu"): string literal to obtain activation function
     - kernels (optional, default: 3): size of the convolution's kernels when using upsampling = "2D_upsample_and_conv" or "transpose_conv"
     - split_kernels (optional, default: False): to decrease the number of parameters, a convolution with the kernel_size (kernel,kernel) can be splitted into two consecutive convolutions with the kernel_size (kernel,1) and (1,kernel) respectivly. applies to upsampling = "2D_upsample_and_conv"
     - useSpecNorm (optional, default: False): applies spectral normalization to convolutional and dense layers
     - padding (optional, default: "zero"): padding type. Options are "none", "zero" or "reflection"
     - use_bias (optional, default: True): determines whether convolutions and dense layers include a bias or not
     - kernel_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the convolutions kernels.
     - gamma_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the normalization layers.
  '''
  def __init__(self,
               upsampling ="2D_upsample_and_conv",
               activation = "leaky_relu", 
               kernels = 3,
               split_kernels = False, 
               useSpecNorm = False,
               use_bias = True,
               padding = "zero",
               kernel_initializer = DeepSaki.initializer.HeAlphaUniform(),
               gamma_initializer =  DeepSaki.initializer.HeAlphaUniform()
               ):
    super(UpSampleBlock, self).__init__()
    self.kernels = kernels 
    self.split_kernels = split_kernels
    self.activation = activation 
    self.useSpecNorm = useSpecNorm
    self.upsampling = upsampling
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.gamma_initializer = gamma_initializer
    self.padding = padding

  def build(self, input_shape):
    super(UpSampleBlock, self).build(input_shape)
    self.layers = []

    if self.upsampling == "2D_upsample_and_conv":
      self.layers.append(tf.keras.layers.UpSampling2D(interpolation='bilinear'))
      self.layers.append(DeepSaki.layers.Conv2DBlock(filters = input_shape[-1], useResidualConv2DBlock = False, kernels = 1, split_kernels = self.split_kernels, numberOfConvs = 1, activation = self.activation,useSpecNorm=self.useSpecNorm, use_bias = self.use_bias,padding = self.padding, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer))
    elif self.upsampling == "transpose_conv":
      self.layers.append(tf.keras.layers.Conv2DTranspose(input_shape[-1],kernel_size = (self.kernels, self.kernels),strides=(2,2),kernel_initializer = self.kernel_initializer,padding='same',use_bias = self.use_bias))
      self.layers.append(tfa.layers.InstanceNormalization(gamma_initializer = self.gamma_initializer))
      self.layers.append(DeepSaki.layers.helper.activation_func(self.activation))
    elif self.upsampling =="depth_to_space":
      self.layers.append(DeepSaki.layers.Conv2DBlock(filters = 4 * input_shape[-1], useResidualConv2DBlock = False, kernels = 1, split_kernels = False, numberOfConvs = 1, activation = self.activation,useSpecNorm=self.useSpecNorm, use_bias = self.use_bias,padding = self.padding, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer))
    else:
      raise Exception("Undefined upsampling provided")

  def call(self, inputs):
    x = inputs
    for layer in self.layers:
      x = layer(x)
    if self.upsampling == "depth_to_space":
      x = tf.nn.depth_to_space(x, block_size = 2)
    return x

  def get_config(self):
    config = super(UpSampleBlock, self).get_config()
    config.update({
        "kernels":self.kernels,
        "split_kernels":self.split_kernels,
        "activation":self.activation,
        "useSpecNorm":self.useSpecNorm,
        "upsampling":self.upsampling,
        "use_bias":self.use_bias,
        "padding": self.padding,
        "kernel_initializer": self.kernel_initializer,
        "gamma_initializer": self.gamma_initializer
        })
    return config

#Testcode
#layer = UpSampleBlock(kernels = 3, upsampling = "transpose_conv", activation = "leaky_relu", useSpecNorm = True, split_kernels = False)
#print(layer.get_config())
#DeepSaki.layers.helper.PlotLayer(layer,inputShape=(256,256,64))
