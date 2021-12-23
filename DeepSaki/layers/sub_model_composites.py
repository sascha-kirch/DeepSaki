import tensorflow as tf
import DeepSaki.layers

class Encoder(tf.keras.layers.Layer):
  '''
  Encoder sub-model combines convolutional blocks with down sample blocks. The spatial width is halfed with every level while the channel depth is doubled.
  args:
    - number_of_levels (optional, default:3): number of conv2D -> Downsampling pairs
    - filters (optional, default:64): defines the number of filters to which the input is exposed.
    - kernels: size of the convolutions kernels
    - limit_filters (optional, default:1024): limits the number of filters, which is doubled with every downsampling block 
    - useResidualConv2DBlock (optional, default: False): ads a residual connection in parallel to the Conv2DBlock
    - downsampling(optional, default: "conv_stride_2"): describes the downsampling method used
    - split_kernels (optional, default: False): to decrease the number of parameters, a convolution with the kernel_size (kernel,kernel) can be splitted into two consecutive convolutions with the kernel_size (kernel,1) and (1,kernel) respectivly
    - numberOfConvs (optional, default: 1): number of consecutive convolutional building blocks, i.e. Conv2DBlock.
    - activation (optional, default: "leaky_relu"): string literal or tensorflow activation function object to obtain activation function
    - first_kernel (optional, default: 5): The first convolution can have a different kernel size, to e.g. increase the perceptive field, while the channel depth is still low.
    - useResidualIdentityBlock (optional, default: False): Whether or not to use the ResidualIdentityBlock instead of the Conv2DBlock
    - residual_cardinality (optional, default: 1): cardinality for the ResidualIdentityBlock
    - channelList (optional, default:None): alternativly to number_of_layers and filters, a list with the disired filters for each level can be provided. e.g. channel_list = [64, 128, 256] results in a 3-level Encoder with 64, 128, 256 filters for level 1, 2 and 3 respectivly.
    - useSpecNorm (optional, default: False): applies spectral normalization to convolutional and dense layers
    - use_bias (optional, default: True): determines whether convolutions and dense layers include a bias or not
    - dropout_rate (optional, default: 0): probability of the dropout layer. If the preceeding layer has more than one channel, spatial dropout is applied, otherwise standard dropout 
    - useSelfAttention (optional, default: False): Determines whether to apply self-attention after the encoder before branching.
    - omit_skips (optional, default: 0): defines how many layers should not output a skip connection output. Requires outputSkips to be True. E.g. if omit_skips = 2, the first two levels do not output a skip connection, it starts at level 3.
    - padding (optional, default: "none"): padding type. Options are "none", "zero" or "reflection"
    - outputSkips (optional, default: False): Whether or not to output skip connections at each level
    - kernel_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the convolutions kernels.
    - gamma_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the normalization layers.
  '''
  def __init__(self,
            number_of_levels = 3, 
            filters = 64, 
            limit_filters = 1024, 
            useResidualConv2DBlock = False, 
            downsampling = "conv_stride_2", 
            kernels = 3,
            split_kernels = False,
            numberOfConvs = 2,
            activation = "leaky_relu",
            first_kernel = None,  
            useResidualIdentityBlock = False,
            residual_cardinality = 1, 
            channelList = None,
            useSpecNorm=False, 
            use_bias = True,
            dropout_rate=0,
            useSelfAttention=False,
            omit_skips = 0,
            padding = "zero",
            outputSkips = False,
            kernel_initializer = DeepSaki.initializer.HeAlphaUniform(),
            gamma_initializer =  DeepSaki.initializer.HeAlphaUniform()
            ):
    super(Encoder, self).__init__()
    self.number_of_levels = number_of_levels
    self.filters = filters
    self.limit_filters = limit_filters
    self.useResidualConv2DBlock = useResidualConv2DBlock
    self.downsampling = downsampling
    self.kernels = kernels
    self.split_kernels = split_kernels
    self.numberOfConvs = numberOfConvs
    self.activation = activation
    self.first_kernel = first_kernel
    self.useResidualIdentityBlock = useResidualIdentityBlock
    self.residual_cardinality = residual_cardinality
    self.channelList = channelList
    self.useSpecNorm = useSpecNorm
    self.dropout_rate = dropout_rate
    self.useSelfAttention = useSelfAttention
    self.omit_skips = omit_skips
    self.padding = padding
    self.outputSkips = outputSkips
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.gamma_initializer = gamma_initializer

  def build(self, input_shape):
    super(Encoder, self).build(input_shape)

    if self.channelList == None:
      self.channelList = [min(self.filters * 2**i, self.limit_filters) for i in range(self.number_of_levels)]
    else:
      self.number_of_levels = len(self.channelList)

    self.encoderBlocks = []
    self.downSampleBlocks = []

    if self.useSelfAttention:
        self.SA = DeepSaki.layers.ScalarGatedSelfAttention(useSpecNorm=self.useSpecNorm, intermediateChannel=None, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer)
    else: 
      self.SA = None

    for i, ch in enumerate(self.channelList):
      if i == 0 and self.first_kernel:
        encoder_kernels = self.first_kernel
      else:
        encoder_kernels = self.kernels

      if self.useResidualIdentityBlock:
        self.encoderBlocks.append(DeepSaki.layers.ResidualIdentityBlock(filters =ch, activation = self.activation, kernels = encoder_kernels,numberOfBlocks=self.numberOfConvs, useSpecNorm=self.useSpecNorm,dropout_rate=self.dropout_rate, use_bias = self.use_bias, residual_cardinality = self.residual_cardinality,padding = self.padding, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer))
        self.downSampleBlocks.append(DeepSaki.layers.ResBlockDown(activation = self.activation, useSpecNorm=self.useSpecNorm, use_bias = self.use_bias,padding = self.padding, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer))
      else:
        self.encoderBlocks.append(DeepSaki.layers.Conv2DBlock(filters=ch, useResidualConv2DBlock = self.useResidualConv2DBlock,kernels = encoder_kernels,split_kernels = self.split_kernels, activation = self.activation, numberOfConvs=self.numberOfConvs,useSpecNorm=self.useSpecNorm,dropout_rate=self.dropout_rate,padding=self.padding,use_bias = self.use_bias, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer))
        self.downSampleBlocks.append(DeepSaki.layers.DownSampleBlock( downsampling = self.downsampling, activation=self.activation,kernels = encoder_kernels,useSpecNorm=self.useSpecNorm,padding=self.padding, use_bias = self.use_bias, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer))

  def call(self, inputs):
    if not self.built:
      raise ValueError('This model has not yet been built.')

    x = inputs
    skips = []

    for level in range(self.number_of_levels):
      if level == 3 and self.SA is not None:
        x = self.SA(x) 
      skip = self.encoderBlocks[level](x)
      x = self.downSampleBlocks[level](skip)
      if self.outputSkips:
        if level >= self.omit_skips: # omit the first skip connection
          skips.append(skip)
        else:
          skips.append(None)

    if self.outputSkips:
      return x, skips
    else:
      return x

  def get_config(self):
    config = super(Encoder, self).get_config()
    config.update({
        "number_of_levels":self.number_of_levels,
        "filters":self.filters,
        "limit_filters":self.limit_filters,
        "useResidualConv2DBlock":self.useResidualConv2DBlock,
        "downsampling":self.downsampling,
        "kernels":self.kernels,
        "split_kernels":self.split_kernels,
        "numberOfConvs":self.numberOfConvs,
        "activation":self.activation,
        "first_kernel":self.first_kernel,
        "useResidualIdentityBlock":self.useResidualIdentityBlock,
        "residual_cardinality":self.residual_cardinality,
        "channelList":self.channelList,
        "useSpecNorm":self.useSpecNorm,
        "use_bias":self.use_bias,
        "dropout_rate":self.dropout_rate,
        "useSelfAttention":self.useSelfAttention,
        "omit_skips":self.omit_skips,
        "padding":self.padding,
        "outputSkips":self.outputSkips,
        "kernel_initializer":self.kernel_initializer,
        "gamma_initializer":self.gamma_initializer
        })
    return config

#Testcode
#layer = Encoder( number_of_levels = 5, filters = 64, limit_filters = 512, useSelfAttention = True,useResidualConv2DBlock = True, downsampling="max_pooling", kernels=3, split_kernels = True,  numberOfConvs = 2,activation = "leaky_relu", first_kernel=3,useResidualIdentityBlock = True,useSpecNorm=True, omit_skips=2)
#print(layer.get_config())
#DeepSaki.layers.helper.PlotLayer(layer,inputShape=(256,256,4))


class Bottleneck(tf.keras.layers.Layer):
  '''
  Bottlenecks are sub-model blocks in auto-encoder-like models such as UNet or ResNet. It is composed of multiple convolution blocks which might have residuals
  args:
    - n_bottleneck_blocks (optional, default: 3): Number of consecutive convolution blocks
    - kernels: size of the convolutions kernels
    - split_kernels (optional, default: False): to decrease the number of parameters, a convolution with the kernel_size (kernel,kernel) can be splitted into two consecutive convolutions with the kernel_size (kernel,1) and (1,kernel) respectivly
    - numberOfConvs (optional, default: 2): number of consecutive convolutional building blocks, i.e. Conv2DBlock.
    - useResidualConv2DBlock (optional, default: True): ads a residual connection in parallel to the Conv2DBlock
    - useResidualIdentityBlock (optional, default: False): Whether or not to use the ResidualIdentityBlock instead of the Conv2DBlock
    - activation (optional, default: "leaky_relu"): string literal or tensorflow activation function object to obtain activation function
    - dropout_rate (optional, default: 0): probability of the dropout layer. If the preceeding layer has more than one channel, spatial dropout is applied, otherwise standard dropout
    - channelList (optional, default:None): alternativly to number_of_layers and filters, a list with the disired filters for each block can be provided. e.g. channel_list = [64, 128, 256] results in a 3-staged Bottleneck with 64, 128, 256 filters for stage 1, 2 and 3 respectivly.
    - useSpecNorm (optional, default: False): applies spectral normalization to convolutional and dense layers
    - use_bias (optional, default: True): determines whether convolutions and dense layers include a bias or not
    - residual_cardinality (optional, default: 1): cardinality for the ResidualIdentityBlock
    - padding (optional, default: "none"): padding type. Options are "none", "zero" or "reflection"
    - kernel_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the convolutions kernels.
    - gamma_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the normalization layers.
  '''
  def __init__(self,
               n_bottleneck_blocks = 3, 
               kernels = 3,
               split_kernels = False,
               numberOfConvs = 2, 
               useResidualConv2DBlock = True, 
               useResidualIdentityBlock = False, 
               activation = "leaky_relu", 
               dropout_rate = 0.2, 
               channelList = None,
               useSpecNorm = False,
               use_bias = True,
               residual_cardinality = 1,
               padding = "zero",
               kernel_initializer = DeepSaki.initializer.HeAlphaUniform(),
               gamma_initializer =  DeepSaki.initializer.HeAlphaUniform()
               ):
    super(Bottleneck, self).__init__()
    self.useResidualIdentityBlock = useResidualIdentityBlock 
    self.n_bottleneck_blocks = n_bottleneck_blocks
    self.useResidualConv2DBlock = useResidualConv2DBlock 
    self.kernels = kernels
    self.split_kernels = split_kernels 
    self.numberOfConvs = numberOfConvs
    self.activation = activation
    self.dropout_rate = dropout_rate
    self.channelList = channelList
    self.useSpecNorm = useSpecNorm
    self.use_bias = use_bias
    self.residual_cardinality = residual_cardinality
    self.padding = padding
    self.kernel_initializer = kernel_initializer
    self.gamma_initializer = gamma_initializer

  def build(self, input_shape):
    super(Bottleneck, self).build(input_shape)

    if self.channelList == None:
      ch = input_shape[-1]
      self.channelList = [ch for i in range(self.n_bottleneck_blocks)]

    self.layers = []
    for ch in self.channelList:
      if self.useResidualIdentityBlock:
        self.layers.append(DeepSaki.layers.ResidualIdentityBlock(activation = self.activation,filters=ch, kernels = self.kernels,numberOfBlocks=self.numberOfConvs,useSpecNorm=self.useSpecNorm, use_bias = self.use_bias,residual_cardinality = self.residual_cardinality,padding = self.padding, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer))  
      else:
        self.layers.append(DeepSaki.layers.Conv2DBlock(filters=ch, useResidualConv2DBlock = self.useResidualConv2DBlock, kernels = self.kernels, split_kernels = self.split_kernels,numberOfConvs=self.numberOfConvs,activation=self.activation,useSpecNorm=self.useSpecNorm, use_bias = self.use_bias,padding = self.padding, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer))

    self.dropout = DeepSaki.layers.helper.dropout_func(self.channelList[-1], self.dropout_rate)

  def call(self, inputs):
    if not self.built:
      raise ValueError('This model has not yet been built.')
    x = inputs

    for layer in self.layers:
      x = layer(x)

    if self.dropout_rate > 0:
      x = self.dropout(x)

    return x

  def get_config(self):
    config = super(Bottleneck, self).get_config()
    config.update({
        "useResidualIdentityBlock":self.useResidualIdentityBlock,
        "n_bottleneck_blocks":self.n_bottleneck_blocks,
        "useResidualConv2DBlock":self.useResidualConv2DBlock,
        "kernels":self.kernels,
        "split_kernels":self.split_kernels,
        "numberOfConvs":self.numberOfConvs,
        "activation":self.activation,
        "dropout_rate":self.dropout_rate,
        "useSpecNorm":self.useSpecNorm,
        "use_bias":self.use_bias,
        "channelList":self.channelList,
        "residual_cardinality":self.residual_cardinality,
        "padding": self.padding,
        "kernel_initializer":self.kernel_initializer,
        "gamma_initializer":self.gamma_initializer
        })
    return config

#Testcode
#layer = Bottleneck(True, 3, False, 3,False,1, "leaky_relu" , dropout_rate = 0.2, channelList = None)
#print(layer.get_config())
#DeepSaki.layers.helper.PlotLayer(layer,inputShape=(256,256,64))

class Decoder(tf.keras.layers.Layer):
  '''
  Decoder sub-model combines convolutional blocks with up sample blocks. The spatial width is double with every level while the channel depth is halfed.
  args:
    - number_of_levels (optional, default:3): number of conv2D -> Upsampling pairs
    - upsampling(optional, default: "2D_upsample_and_conv"): describes the upsampling method used
    - filters (optional, default:64): defines the number of filters to which the input is exposed.
    - limit_filters (optional, default:1024): limits the number of filters 
    - useResidualConv2DBlock (optional, default: False): ads a residual connection in parallel to the Conv2DBlock
    - kernels: size of the convolutions kernels
    - split_kernels (optional, default: False): to decrease the number of parameters, a convolution with the kernel_size (kernel,kernel) can be splitted into two consecutive convolutions with the kernel_size (kernel,1) and (1,kernel) respectivly
    - numberOfConvs (optional, default: 1): number of consecutive convolutional building blocks, i.e. Conv2DBlock.
    - activation (optional, default: "leaky_relu"): string literal or tensorflow activation function object to obtain activation function
    - dropout_rate (optional, default: 0): probability of the dropout layer. If the preceeding layer has more than one channel, spatial dropout is applied, otherwise standard dropout. In the decoder only applied to the first half of levels. 
    - useResidualIdentityBlock (optional, default: False): Whether or not to use the ResidualIdentityBlock instead of the Conv2DBlock
    - residual_cardinality (optional, default: 1): cardinality for the ResidualIdentityBlock
    - channelList (optional, default:None): alternativly to number_of_layers and filters, a list with the disired filters for each level can be provided. e.g. channel_list = [64, 128, 256] results in a 3-level Decoder with 64, 128, 256 filters for level 1, 2 and 3 respectivly.
    - useSpecNorm (optional, default: False): applies spectral normalization to convolutional and dense layers
    - use_bias (optional, default: True): determines whether convolutions and dense layers include a bias or not
    - useSelfAttention (optional, default: False): Determines whether to apply self-attention after the encoder before branching.
    - enableSkipConnectionsInput (optional, default: False): Whether or not to input skip connections at each level
    - padding (optional, default: "none"): padding type. Options are "none", "zero" or "reflection"
    - kernel_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the convolutions kernels.
    - gamma_initializer (optional, default: DeepSaki.initializer.HeAlphaUniform()): Initialization of the normalization layers.
  '''
  def __init__(self,
            number_of_levels = 3, 
            upsampling = "2D_upsample_and_conv", 
            filters = 64, 
            limit_filters = 1024, 
            useResidualConv2DBlock = False,
            kernels = 3,
            split_kernels = False,
            numberOfConvs = 2,
            activation = "leaky_relu", 
            dropout_rate = 0.2, 
            useResidualIdentityBlock = False,
            residual_cardinality = 1,
            channelList = None, 
            useSpecNorm=False, 
            use_bias = True,
            useSelfAttention=False,
            enableSkipConnectionsInput = False,
            padding = "zero",
            kernel_initializer = DeepSaki.initializer.HeAlphaUniform(),
            gamma_initializer =  DeepSaki.initializer.HeAlphaUniform()
            ):
    super(Decoder, self).__init__()
    self.number_of_levels = number_of_levels
    self.filters = filters
    self.upsampling = upsampling
    self.limit_filters = limit_filters
    self.useResidualConv2DBlock = useResidualConv2DBlock
    self.kernels = kernels
    self.split_kernels = split_kernels
    self.numberOfConvs = numberOfConvs
    self.activation = activation
    self.useResidualIdentityBlock = useResidualIdentityBlock
    self.channelList = channelList
    self.useSpecNorm = useSpecNorm
    self.use_bias = use_bias
    self.dropout_rate = dropout_rate
    self.useSelfAttention = useSelfAttention
    self.enableSkipConnectionsInput = enableSkipConnectionsInput
    self.residual_cardinality = residual_cardinality
    self.padding = padding
    self.kernel_initializer = kernel_initializer
    self.gamma_initializer = gamma_initializer

  def build(self, input_shape):
    super(Decoder, self).build(input_shape)

    if self.channelList == None:
      self.channelList = [min(self.filters * 2**i, self.limit_filters) for i in reversed(range(self.number_of_levels))]
    else:
      self.number_of_levels = len(self.channelList)

    self.decoderBlocks = []
    self.upSampleBlocks = []
    self.dropouts = []

    if self.useSelfAttention:
        self.SA =DeepSaki.layers.ScalarGatedSelfAttention(useSpecNorm=self.useSpecNorm, intermediateChannel=None, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer)
    else: 
      self.SA = None

    for i, ch in enumerate(self.channelList):
      if i < int(self.number_of_levels/2): # ">" since i is reversed
        dropout_rate = self.dropout_rate
      else:
        dropout_rate = 0

      if self.useResidualIdentityBlock:
        self.decoderBlocks.append(DeepSaki.layers.ResidualIdentityBlock(filters =ch, activation = self.activation, kernels = self.kernels,numberOfBlocks=self.numberOfConvs,useSpecNorm=self.useSpecNorm,dropout_rate=dropout_rate, use_bias = self.use_bias, residual_cardinality = self.residual_cardinality,padding = self.padding, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer))
        self.upSampleBlocks.append(DeepSaki.layers.ResBlockUp(activation=self.activation,useSpecNorm=self.useSpecNorm, use_bias = self.use_bias,padding = self.padding, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer))
      else:
        self.decoderBlocks.append(DeepSaki.layers.Conv2DBlock(filters = ch,useResidualConv2DBlock=self.useResidualConv2DBlock, kernels = self.kernels,split_kernels=self.split_kernels, activation = self.activation,numberOfConvs=self.numberOfConvs, dropout_rate=dropout_rate,useSpecNorm=self.useSpecNorm, use_bias = self.use_bias,padding = self.padding, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer))
        self.upSampleBlocks.append(DeepSaki.layers.UpSampleBlock(kernels = self.kernels, upsampling = self.upsampling, split_kernels = self.split_kernels,activation=self.activation,useSpecNorm=self.useSpecNorm, use_bias = self.use_bias,padding = self.padding, kernel_initializer = self.kernel_initializer, gamma_initializer = self.gamma_initializer))

  def call(self, inputs):
    if not self.built:
      raise ValueError('This model has not yet been built.')
    skipConnections = None
    if self.enableSkipConnectionsInput:
      x, skipConnections = inputs
    else:
      x = inputs

    for level in range(self.number_of_levels):
      if level == 3 and self.SA is not None:
        x = self.SA(x) 
      x = self.upSampleBlocks[level](x)
      if skipConnections is not None: 
        x = tf.keras.layers.concatenate([x, skipConnections[self.number_of_levels - (level+1)]])
      x = self.decoderBlocks[level](x)

    return x

  def get_config(self):
    config = super(Decoder, self).get_config()
    config.update({
        "number_of_levels":self.number_of_levels,
        "filters":self.filters,
        "limit_filters":self.limit_filters,
        "useResidualConv2DBlock":self.useResidualConv2DBlock,
        "upsampling":self.upsampling,
        "kernels":self.kernels,
        "split_kernels":self.split_kernels,
        "numberOfConvs":self.numberOfConvs,
        "activation":self.activation,
        "useResidualIdentityBlock":self.useResidualIdentityBlock,
        "residual_cardinality": self.residual_cardinality,
        "channelList":self.channelList,
        "useSpecNorm":self.useSpecNorm,
        "dropout_rate":self.dropout_rate,
        "useSelfAttention":self.useSelfAttention,
        "enableSkipConnectionsInput":self.enableSkipConnectionsInput,
        "padding": self.padding,
        "kernel_initializer":self.kernel_initializer,
        "gamma_initializer":self.gamma_initializer
        })
    return config

#Testcode
#layer = Decoder( number_of_levels = 5, filters = 64, limit_filters = 2048, useSelfAttention = True,useResidualConv2DBlock = False, upsampling="depth_to_space", kernels=3, split_kernels = False,  numberOfConvs = 2,activation = "leaky_relu",useResidualIdentityBlock = True,useSpecNorm=False, dropout_rate = 0.2)
#print(layer.get_config())
#DeepSaki.layers.helper.PlotLayer(layer,inputShape=(256,256,4))

