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
        x = tf.keras.layers.Activation(self.activation)(x)
    
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
        x = tf.keras.layers.Activation(self.activation)(x)
    
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
   Spatial up-sampling for grid-like data
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
      self.layers.append(tf.keras.layers.Activation(self.activation))
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

class ResidualIdentityBlock(tf.keras.layers.Layer):
  '''
  Residual identity block with configurable cardinality
  args:
    - filters: number of filters in the output feature map
    - kernels: size of the convolutions kernels
    - numberOfBlocks (optional, default: 1): number of consecutive convolutional building blocks.
    - activation (optional, default: "leaky_relu"): string literal to obtain activation function
    - dropout_rate (optional, default: 0): probability of the dropout layer. If the preceeding layer has more than one channel, spatial dropout is applied, otherwise standard dropout
    - useSpecNorm (optional, default: False): applies spectral normalization to convolutional and dense layers
    - residual_cardinality (optional, default: 1): number of parallel convolution blocks
    - padding (optional, default: "zero"): padding type. Options are "none", "zero" or "reflection"
    - use_bias (optional, default: True): determines whether convolutions and dense layers include a bias or not
    - kernel_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the convolutions kernels.
    - gamma_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the normalization layers.
  '''
  def __init__(self,
               filters,
               kernels,
               activation = "leaky_relu", 
               numberOfBlocks = 1,
               useSpecNorm = False, 
               residual_cardinality = 1, 
               dropout_rate = 0,
               use_bias = True,
               padding = "zero",
               kernel_initializer = DeepSaki.initializer.HeAlphaUniform(),
               gamma_initializer =  DeepSaki.initializer.HeAlphaUniform() 
               ):
    super(ResidualIdentityBlock, self).__init__()
    self.activation = activation 
    self.filters = filters
    self.kernels = kernels 
    self.numberOfBlocks = numberOfBlocks
    self.useSpecNorm = useSpecNorm 
    self.residual_cardinality = residual_cardinality
    self.dropout_rate = dropout_rate
    self.use_bias=use_bias
    self.padding = padding
    self.kernel_initializer = kernel_initializer
    self.gamma_initializer = gamma_initializer

    self.pad = int((kernels-1)/2) # assumes odd kernel size, which is typical!

    if residual_cardinality > 1:
      self.intermediateFilters = int(max(filters/32, max(filters/16, max(filters/8, max(filters/4, max(filters/2, 1))))))
    else:
      self.intermediateFilters = int(max(filters/4, max(filters/2, 1))) 

    # for each block, add several con
    self.blocks = []
    for i in range(numberOfBlocks):
      cardinals = []
      for _ in range(residual_cardinality):
        cardinals.append(
            [
            DeepSaki.layers.Conv2DBlock(filters=self.intermediateFilters, useResidualConv2DBlock=False, kernels=1, split_kernels=False, numberOfConvs=1, activation=activation,useSpecNorm=useSpecNorm, use_bias=use_bias,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer),
            DeepSaki.layers.Conv2DBlock(filters=self.intermediateFilters, useResidualConv2DBlock=False, kernels=kernels, split_kernels=False, numberOfConvs=1, activation=activation,padding="none",useSpecNorm=useSpecNorm, use_bias=use_bias, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer),
            DeepSaki.layers.Conv2DBlock(filters=filters, useResidualConv2DBlock=False, kernels=1, split_kernels=False, numberOfConvs=1, activation=activation,useSpecNorm=useSpecNorm, use_bias=use_bias,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
            ]
        )
      self.blocks.append(cardinals)

    self.dropout = DeepSaki.layers.helper.dropout_func(filters, dropout_rate)

  def build(self, input_shape):
    super(ResidualIdentityBlock, self).build(input_shape)
    self.conv0 = None
    if input_shape[-1] != self.filters:
      self.conv0 = DeepSaki.layers.Conv2DBlock(filters=self.filters, useResidualConv2DBlock=False, kernels=1, split_kernels=False, numberOfConvs=1, activation=self.activation,useSpecNorm=self.useSpecNorm, use_bias=self.use_bias,padding = self.padding, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer)

  def call(self, inputs):
    if not self.built:
      raise ValueError('This model has not yet been built.')
    x = inputs

    if self.conv0 != None:
      x = self.conv0(x)

    for block in range(self.numberOfBlocks):
      residual = x

      if self.pad != 0 and self.padding != "none":
        x = DeepSaki.layers.helper.pad_func(padValues = (self.pad,self.pad), padding=self.padding)(x)

      # y is allways the footpoint of the cardinality blocks
      y = x
      for cardinal in range(self.residual_cardinality):
        # after the first iteration c is the previous cardinality block output
        # its used to iterativly add the result, rather than summing all at once. 
        # performance reasons, since otherwise multiple arrays must be stored at once!
        c = x
        x = self.blocks[block][cardinal][0](y)
        x = self.blocks[block][cardinal][1](x)
        x = self.blocks[block][cardinal][2](x)
        if cardinal > 0:
          x = tf.keras.layers.Add()([x, c])

      x = tf.keras.layers.Add()([x, residual])

    if self.dropout_rate > 0:
      x = self.dropout(x)

    return x

  def get_config(self):
    config = super(ResidualIdentityBlock, self).get_config()
    config.update({
        "activation":self.activation,
        "filters":self.filters,
        "kernels":self.kernels,
        "numberOfBlocks":self.numberOfBlocks,
        "useSpecNorm":self.useSpecNorm,
        "residual_cardinality":self.residual_cardinality,
        "dropout_rate":self.dropout_rate,
        "use_bias":self.use_bias,
        "padding":self.padding,
        "kernel_initializer":self.kernel_initializer,
        "gamma_initializer":self.gamma_initializer
        })
    return config

#Testcode
#layer = ResidualIdentityBlock(filters =64, activation = "leaky_relu", kernels = 3, numberOfBlocks=2,useSpecNorm = False, residual_cardinality =5, dropout_rate=0.2)
#print(layer.get_config())
#DeepSaki.layers.helper.PlotLayer(layer,inputShape=(256,256,64))

class ResBlockDown(tf.keras.layers.Layer):
  '''
   Spatial down-sampling with residual connection for grid-like data
   args:
     - activation (optional, default: "leaky_relu"): string literal to obtain activation function
     - useSpecNorm (optional, default: False): applies spectral normalization to convolutional and dense layers
     - use_bias (optional, default: True): determines whether convolutions and dense layers include a bias or not
     - padding (optional, default: "zero"): padding type. Options are "none", "zero" or "reflection"
     - kernel_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the convolutions kernels.
     - gamma_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the normalization layers.
  '''
  def __init__(self,
               activation = "leaky_relu", 
               useSpecNorm = False, 
               use_bias = True,
               padding = "zero",
               kernel_initializer = DeepSaki.initializer.HeAlphaUniform(),
               gamma_initializer =  DeepSaki.initializer.HeAlphaUniform()
               ):
    super(ResBlockDown, self).__init__()
    self.activation = activation 
    self.useSpecNorm = useSpecNorm
    self.use_bias = use_bias
    self.padding = padding
    self.kernel_initializer = kernel_initializer
    self.gamma_initializer = gamma_initializer

  def build(self, input_shape):
    super(ResBlockDown, self).build(input_shape)

    self.convRes = DeepSaki.layers.Conv2DBlock(filters = input_shape[-1], useResidualConv2DBlock=False, kernels=1, split_kernels=False, numberOfConvs=1, activation=self.activation,useSpecNorm=self.useSpecNorm, use_bias=self.use_bias,padding = self.padding, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer)
    self.conv1 = DeepSaki.layers.Conv2DBlock(filters = input_shape[-1],   useResidualConv2DBlock=False, kernels=3, split_kernels=False, numberOfConvs=1, activation=self.activation,useSpecNorm=self.useSpecNorm, use_bias=self.use_bias,padding = self.padding, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer)
    self.conv2 = DeepSaki.layers.Conv2DBlock(filters = input_shape[-1],   useResidualConv2DBlock=False, kernels=3, split_kernels=False, numberOfConvs=1, activation=self.activation,useSpecNorm=self.useSpecNorm, use_bias=self.use_bias,padding = self.padding, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer)
    
  def call(self, inputs):
    path1 = inputs
    path2 = inputs

    path1 = self.convRes(path1)
    path1 = tf.keras.layers.AveragePooling2D()(path1)

    path2 = self.conv1(path2)
    path2 = self.conv2(path2)
    path2 = tf.keras.layers.AveragePooling2D()(path2)

    x = tf.keras.layers.Add()([path1, path2])
    return x

  def get_config(self):
    config = super(ResBlockDown, self).get_config()
    config.update({
        "activation":self.activation,
        "useSpecNorm":self.useSpecNorm,
        "use_bias":self.use_bias,
        "padding":self.padding,
        "kernel_initializer":self.kernel_initializer,
        "gamma_initializer":self.gamma_initializer
        })
    return config

#Testcode
#layer = ResBlockDown( activation = "leaky_relu", useSpecNorm = True)
#print(layer.get_config())
#DeepSaki.layers.helper.PlotLayer(layer,inputShape=(256,256,64))

class ResBlockUp(tf.keras.layers.Layer):
  '''
   Spatial down-sampling with residual connection for grid-like data
   args:
     - activation (optional, default: "leaky_relu"): string literal to obtain activation function
     - useSpecNorm (optional, default: False): applies spectral normalization to convolutional and dense layers
     - padding (optional, default: "zero"): padding type. Options are "none", "zero" or "reflection"
     - use_bias (optional, default: True): determines whether convolutions and dense layers include a bias or not
     - kernel_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the convolutions kernels.
     - gamma_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the normalization layers.
  '''
  def __init__(self,
               activation = "leaky_relu", 
               useSpecNorm = False,
               use_bias = True,
               padding = "zero",
               kernel_initializer = DeepSaki.initializer.HeAlphaUniform(),
               gamma_initializer =  DeepSaki.initializer.HeAlphaUniform()
               ):
    super(ResBlockUp, self).__init__()
    self.activation = activation 
    self.useSpecNorm = useSpecNorm
    self.use_bias = use_bias
    self.padding = padding
    self.kernel_initializer = kernel_initializer
    self.gamma_initializer = gamma_initializer

  def build(self, input_shape):
    super(ResBlockUp, self).build(input_shape)
    self.convRes = DeepSaki.layers.Conv2DBlock(filters=input_shape[-1], useResidualConv2DBlock=False, kernels=1, split_kernels=False, numberOfConvs=1, activation=self.activation,useSpecNorm=self.useSpecNorm,use_bias = self.use_bias,padding = self.padding, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer)
    self.conv1 = DeepSaki.layers.Conv2DBlock(filters=input_shape[-1], useResidualConv2DBlock=False, kernels=3, split_kernels=False, numberOfConvs=1, activation=self.activation,useSpecNorm=self.useSpecNorm,use_bias = self.use_bias,padding = self.padding, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer)
    self.conv2 = DeepSaki.layers.Conv2DBlock(filters=input_shape[-1], useResidualConv2DBlock=False, kernels=3, split_kernels=False, numberOfConvs=1, activation=self.activation,useSpecNorm=self.useSpecNorm,use_bias = self.use_bias,padding = self.padding, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer)
    
  def call(self, inputs):
    path1 = inputs
    path2 = inputs

    path1 = tf.keras.layers.UpSampling2D()(path1)
    path1 = self.convRes(path1)

    path2 = tf.keras.layers.UpSampling2D()(path2)
    path2 = self.conv1(path2)
    path2 = self.conv2(path2)

    x = tf.keras.layers.Add()([path1, path2])
    return x

  def get_config(self):
    config = super(ResBlockUp, self).get_config()
    config.update({
        "activation":self.activation,
        "useSpecNorm":self.useSpecNorm,
        "use_bias": self.use_bias,
        "padding": self.padding,
        "kernel_initializer":self.kernel_initializer,
        "gamma_initializer":self.gamma_initializer
        })
    return config

#Testcode
#layer = ResBlockUp(activation = "leaky_relu", useSpecNorm = True)
#print(layer.get_config())
#DeepSaki.layers.helper.PlotLayer(layer,inputShape=(256,256,64))

class NonNegative(tf.keras.constraints.Constraint):
  '''
  constraint that enforces positive activations
  '''
  def __call__(self, w):
    return w * tf.cast(tf.math.greater_equal(w, 0.), w.dtype)

@tf.keras.utils.register_keras_serializable(package='Custom', name='scale')
class ScaleLayer(tf.keras.layers.Layer):
  '''
  trainable scalar that can act as trainable gate
  args:
    - initializer (optional, default: tf.keras.initializers.Ones()): initializes the scalar weight
  '''
  def __init__(self,
               initializer = tf.keras.initializers.Ones()
               ):
    super(ScaleLayer, self).__init__()
    #self.scale = tf.Variable([0.],shape = (1),trainable =True)
    self.initializer = initializer
    self.scale = self.add_weight(shape=[1],initializer = initializer, constraint=NonNegative() , trainable=True)

  def call(self, inputs):
    return inputs * self.scale

  def get_config(self):
    config = super(ScaleLayer, self).get_config()
    config.update({
        "scale": self.scale,
        "initializer":self.initializer
        })
    return config
 
class ScalarGatedSelfAttention(tf.keras.layers.Layer):
  '''
  Scaled dot-product self attention that is gated by a learnable scalar.
  args:
  - useSpecNorm (optional, default: False): applies spectral normalization to convolutional and dense layers
  - intermediateChannel (optional, default: None): Integer that determines the intermediate channels within the self-attention model. If None, intermediate channels = inputChannels/8 
  - kernel_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the convolutions kernels.
  - gamma_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the normalization layers.
  '''
  def __init__(self, 
               useSpecNorm = False,
               intermediateChannel = None,
               kernel_initializer = DeepSaki.initializer.HeAlphaUniform(),
               gamma_initializer =  DeepSaki.initializer.HeAlphaUniform()
               ):
    super(ScalarGatedSelfAttention, self).__init__()
    self.useSpecNorm = useSpecNorm
    self.intermediateChannel = intermediateChannel
    self.kernel_initializer = kernel_initializer
    self.gamma_initializer=gamma_initializer

  def build(self, input_shape):
    super(ScalarGatedSelfAttention, self).build(input_shape)
    batchSize, height, width, numChannel = input_shape
    if self.intermediateChannel == None:
      self.intermediateChannel = int(numChannel/8)

    self.w_f = DeepSaki.layers.DenseBlock(units = self.intermediateChannel, useSpecNorm = self.useSpecNorm, numberOfLayers = 1, activation = None, applyFinalNormalization = False, use_bias = False, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer)
    self.w_g = DeepSaki.layers.DenseBlock(units = self.intermediateChannel, useSpecNorm = self.useSpecNorm, numberOfLayers = 1, activation = None, applyFinalNormalization = False, use_bias = False, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer)
    self.w_h = DeepSaki.layers.DenseBlock(units = self.intermediateChannel, useSpecNorm = self.useSpecNorm, numberOfLayers = 1, activation = None, applyFinalNormalization = False, use_bias = False, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer)
    self.w_fgh = DeepSaki.layers.DenseBlock(units = numChannel, useSpecNorm = self.useSpecNorm, numberOfLayers = 1, activation = None, applyFinalNormalization = False, use_bias = False, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer)

    self.LN_f = tf.keras.layers.LayerNormalization(gamma_initializer = self.gamma_initializer)
    self.LN_g = tf.keras.layers.LayerNormalization(gamma_initializer = self.gamma_initializer)
    self.LN_h = tf.keras.layers.LayerNormalization(gamma_initializer = self.gamma_initializer)
    self.LN_fgh = tf.keras.layers.LayerNormalization(gamma_initializer = self.gamma_initializer)
    self.scale = DeepSaki.layers.ScaleLayer()

  def call(self, inputs):
    if not self.built:
      raise ValueError('This model has not yet been built.')

    f = self.w_f(inputs)
    f = self.LN_f(f)
    f = tf.keras.layers.Permute(dims=(2, 1, 3))(f)

    g = self.w_f(inputs)
    g = self.LN_f(g)

    h = self.w_h(inputs)
    h = self.LN_h(h)

    f_g = tf.keras.layers.Multiply()([f, g])
    f_g = tf.keras.layers.Softmax(axis=1)(f_g)

    f_g_h = tf.keras.layers.Multiply()([f_g, h])
    f_g_h = self.w_fgh(f_g_h)
    f_g_h = self.LN_fgh(f_g_h)
    f_g_h = self.scale(f_g_h)

    z = tf.keras.layers.Add()([f_g_h, inputs])

    return z

  def get_config(self):
    config = super(ScalarGatedSelfAttention, self).get_config()
    config.update({
        "useSpecNorm":self.useSpecNorm,
        "intermediateChannel":self.intermediateChannel,
        "gamma_initializer":self.gamma_initializer,
        "kernel_initializer":self.kernel_initializer
        })
    return config

#Testcode
#layer =ScalarGatedSelfAttention(useSpecNorm = True,intermediateChannel = None)
#print(layer.get_config())
#DeepSaki.layers.helper.PlotLayer(layer,inputShape=(32,32,512))
