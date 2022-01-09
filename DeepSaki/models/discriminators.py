import tensorflow as tf
import DeepSaki.layers
import DeepSaki.initializer

class LayoutContentDiscriminator(tf.keras.Model):
  '''
  Discriminator/critic model with two outputs to enforce disentanglement.
  First, a layout output with (batch, height, width, 1) that focuses on the inputs layout by reducing its channel depth to 1. 
  Seccond, a content output that discriminates a feature vector with an spatial width of 1.
  Inspired by: http://arxiv.org/abs/2103.13389
  args:
    - filters (optional): defines the number of filters to which the input is exposed.
    - downsampling(optional): describes the downsampling method 
    - kernels (optional): size of the kernel for convolutional layers
    - first_kernel (optional): The first convolution can have a different kernel size, to e.g. increase the perceptive field, while the channel depth is still low.
    - split_kernels (optional): to decrease the number of parameters, a convolution with the kernel_size (kernel,kernel) can be splitted into two consecutive convolutions with the kernel_size (kernel,1) and (1,kernel) respectivly
    - numberOfConvs (optional): number of consecutive convolutional building blocks, i.e. Conv2DBlock.
    - activation (optional): string literal or tensorflow activation function object to obtain activation function
    - dropout_rate (optional): probability of the dropout layer. If the preceeding layer has more than one channel, spatial dropout is applied, otherwise standard dropout
    - useSpecNorm (optional): applies spectral normalization to convolutional and dense layers
    - use_bias (optional): determines whether convolutions and dense layers include a bias or not
    - padding (optional): padding type. Options are "none", "zero" or "reflection"
    - FullyConected (optional): determines whether 1x1 convolutions are replaced by linear layers, which gives the same result, but linear layers are faster. Option: "MLP" or "1x1_conv"
    - useSelfAttention (optional): Determines whether to apply self-attention after the encoder before branching.
    - kernel_initializer (optional): Initialization of the convolutions kernels.
    - gamma_initializer (optional): Initialization of the normalization layers.
  
  output call():
    - out1: content output
    - out2: layout output

  Example:
    >>> import DeepSaki
    >>> import tensorflow as tf
    >>> inputs = tf.keras.layers.Input(shape = (256,256,4))
    >>> model = tf.keras.Model(inputs=inputs, outputs=DeepSaki.models.LayoutContentDiscriminator().call(inputs))
    >>> model.summary()
    >>> tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True, show_dtype=True, to_file='Unet_discriminator_model.png')

  '''
  def __init__(self,
            filters = 64,
            downsampling = "conv_stride_2", 
            kernels = 3,
            first_kernel = 5,
            split_kernels = False,
            numberOfConvs = 2,
            activation = "leaky_relu",
            dropout_rate = 0.2,
            useSpecNorm=False,
            use_bias = True,
            padding = "none",
            FullyConected = "MLP",
            useSelfAttention = False,
            kernel_initializer = DeepSaki.initializer.HeAlphaUniform(),
            gamma_initializer =  DeepSaki.initializer.HeAlphaUniform()
            ):
    super(LayoutContentDiscriminator, self).__init__()
    self.encoder = DeepSaki.layers.Encoder(3, filters, 1024, False, downsampling, kernels, split_kernels, numberOfConvs,activation, first_kernel,False,channelList=[4*filters,4*filters,8*filters],useSpecNorm=useSpecNorm, padding = padding, use_bias = use_bias, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    if useSelfAttention:
        self.SA = DeepSaki.layers.ScalarGatedSelfAttention(useSpecNorm=useSpecNorm, intermediateChannel=None, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    else: 
      self.SA = None

    if FullyConected == "MLP":
      self.cont1 = DeepSaki.layers.DenseBlock(units = filters * 8, useSpecNorm = useSpecNorm, numberOfLayers = numberOfConvs, activation = activation, dropout_rate =dropout_rate, use_bias = use_bias, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
      self.cont2 = DeepSaki.layers.DenseBlock(units = filters * 8, useSpecNorm = useSpecNorm, numberOfLayers = numberOfConvs, activation = activation, dropout_rate =dropout_rate, use_bias = use_bias, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
      self.cont3 = DeepSaki.layers.DenseBlock(units = filters * 8, useSpecNorm = useSpecNorm, numberOfLayers = numberOfConvs, activation = activation, dropout_rate =0, final_activation=False, applyFinalNormalization = False, use_bias = use_bias, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    elif FullyConected == "1x1_conv": 
      self.cont1 = DeepSaki.layers.Conv2DBlock(filters=filters * 8, kernels = 1, activation = activation, split_kernels = split_kernels,numberOfConvs=numberOfConvs, useResidualConv2DBlock=False,dropout_rate=dropout_rate,useSpecNorm=useSpecNorm, padding=padding,use_bias = use_bias, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
      self.cont2 = DeepSaki.layers.Conv2DBlock(filters=filters * 8, kernels = 1, activation = activation, split_kernels = split_kernels,numberOfConvs=numberOfConvs, useResidualConv2DBlock=False,dropout_rate=dropout_rate,useSpecNorm=useSpecNorm, padding=padding,use_bias = use_bias, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
      self.cont3 = DeepSaki.layers.Conv2DBlock(filters=filters * 8, kernels = 1, activation = activation, split_kernels = split_kernels,numberOfConvs=numberOfConvs, useResidualConv2DBlock=False,dropout_rate=0,useSpecNorm=useSpecNorm, final_activation=False, applyFinalNormalization = False, padding=padding,use_bias = use_bias, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    else:
      raise Exception("FullyConected:{} is not defined".format(FullyConected))

    self.lay1 = DeepSaki.layers.Conv2DBlock(filters=1, kernels = kernels, activation = activation, split_kernels = split_kernels,numberOfConvs=numberOfConvs, useResidualConv2DBlock=False,dropout_rate=0,useSpecNorm=useSpecNorm, padding=padding, use_bias = use_bias, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    self.lay2 = DeepSaki.layers.Conv2DBlock(filters=1, kernels = kernels, activation = activation, split_kernels = split_kernels,numberOfConvs=numberOfConvs, useResidualConv2DBlock=False,dropout_rate=dropout_rate,useSpecNorm=useSpecNorm, padding=padding,use_bias = use_bias, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    self.lay3 = DeepSaki.layers.Conv2DBlock(filters=1, kernels = kernels, activation = activation, split_kernels = split_kernels,numberOfConvs=numberOfConvs, useResidualConv2DBlock=False,dropout_rate=dropout_rate,useSpecNorm=useSpecNorm, padding=padding,use_bias = use_bias, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    self.lay4 = DeepSaki.layers.Conv2DBlock(filters=1, kernels = kernels, activation = activation, split_kernels = split_kernels,numberOfConvs=numberOfConvs, useResidualConv2DBlock=False,dropout_rate=0,useSpecNorm=useSpecNorm,final_activation=False, applyFinalNormalization = False, padding=padding,use_bias = use_bias, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    #To enable mixed precission support for matplotlib and distributed training and to increase training stability
    self.linear_dtype = tf.keras.layers.Activation("linear", dtype = tf.float32)

  def call(self, inputs):
    x = inputs
    F = self.encoder(x)
    if self.SA is not None:
        F = self.SA(F) 

    batchsize, pool_height, pool_width, channels = tf.keras.backend.int_shape(F)
    F_cont = tf.keras.layers.AveragePooling2D(pool_size=(pool_height, pool_width))(F)
    out1 = self.cont1(F_cont)
    out1 = self.cont2(out1)
    out1 = self.cont3(out1)
    out1 = self.linear_dtype(out1)

    out2 = self.lay1(F)
    out2 = self.lay2(out2)
    out2 = self.lay3(out2)
    out2 = self.lay4(out2)
    out2 = self.linear_dtype(out2)

    return [out1, out2]

class PatchDiscriminator(tf.keras.Model):
  '''
  Discriminator/critic model with patched output.
  args:
    - filters (optional): defines the number of filters to which the input is exposed.
    - downsampling(optional): describes the downsampling method 
    - kernels (optional): size of the kernel for convolutional layers
    - first_kernel (optional): The first convolution can have a different kernel size, to e.g. increase the perceptive field, while the channel depth is still low.
    - split_kernels (optional): to decrease the number of parameters, a convolution with the kernel_size (kernel,kernel) can be splitted into two consecutive convolutions with the kernel_size (kernel,1) and (1,kernel) respectivly
    - numberOfConvs (optional): number of consecutive convolutional building blocks, i.e. Conv2DBlock.
    - activation (optional): string literal or tensorflow activation function object to obtain activation function
    - num_down_blocks (optional): Number of downsampling blocks in the encoder
    - dropout_rate (optional): probability of the dropout layer. If the preceeding layer has more than one channel, spatial dropout is applied, otherwise standard dropout
    - useSpecNorm (optional): applies spectral normalization to convolutional and dense layers
    - use_bias (optional): determines whether convolutions and dense layers include a bias or not
    - padding (optional): padding type. Options are "none", "zero" or "reflection"
    - useSelfAttention (optional): Determines whether to apply self-attention after the encoder before branching.
    - kernel_initializer (optional): Initialization of the convolutions kernels.
    - gamma_initializer (optional): Initialization of the normalization layers.
  
  output call():
    - out1: content output
    - out2: layout output

  Example:
    >>> import DeepSaki
    >>> import tensorflow as tf
    >>> inputs = tf.keras.layers.Input(shape = (256,256,4))
    >>> model = tf.keras.Model(inputs=inputs, outputs=DeepSaki.models.PatchDiscriminator().call(inputs))
    >>> model.summary()
    >>> tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True, show_dtype=True, to_file='PatchDiscriminator_model.png')

  '''
  def __init__(self,
            filters = 64,
            downsampling = "average_pooling", 
            kernels = 3,
            first_kernel = 5,
            split_kernels = False,
            numberOfConvs = 2,
            activation = "leaky_relu",
            num_down_blocks = 3,
            dropout_rate = 0.2,
            useSpecNorm= False,
            use_bias = True,
            useSelfAttention=False,
            padding = "none",
            kernel_initializer = DeepSaki.initializer.HeAlphaUniform(),
            gamma_initializer =  DeepSaki.initializer.HeAlphaUniform()
            ):
    super(PatchDiscriminator, self).__init__()

    self.encoder = DeepSaki.layers.Encoder(number_of_levels=num_down_blocks, filters=filters, limit_filters=512, useResidualConv2DBlock=False, downsampling=downsampling, kernels=kernels, split_kernels=split_kernels,numberOfConvs=numberOfConvs,activation=activation,first_kernel=first_kernel,useSpecNorm=useSpecNorm,useSelfAttention=useSelfAttention, use_bias = use_bias,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    self.conv1 = DeepSaki.layers.Conv2DBlock(filters = filters * (2**(num_down_blocks)), useResidualConv2DBlock = False, kernels = kernels, split_kernels = split_kernels, numberOfConvs = numberOfConvs, activation = activation,dropout_rate=dropout_rate,useSpecNorm=useSpecNorm, use_bias = use_bias,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    self.conv2 = DeepSaki.layers.Conv2DBlock(filters = 1,useResidualConv2DBlock = False, kernels = 5, split_kernels  = False, numberOfConvs = 1, activation = None,useSpecNorm=useSpecNorm, applyFinalNormalization = False, use_bias = use_bias,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    #To enable mixed precission support for matplotlib and distributed training and to increase training stability
    self.linear_dtype = tf.keras.layers.Activation("linear", dtype = tf.float32)

  def call(self, inputs):
    x = inputs
    x = self.encoder(x)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.linear_dtype(x)
    return x

  
class UNetDiscriminator(tf.keras.Model):
  '''
  U-Net based discriminator model with skip conections between encoder and decoder and two outputs; after the bottleneck and after the decoder.
  inspired by Schönfeld et. al. http://arxiv.org/abs/2002.12655 
  args:
  - number_of_levels (optional): number of down and apsampling levels of the model
  - upsampling (optional): describes the upsampling method used
  - downsampling (optional): describes the downsampling method 
- kernels (optional): size of the convolutions kernels
- first_kernel (optional): The first convolution can have a different kernel size, to e.g. increase the perceptive field, while the channel depth is still low.
- split_kernels (optional): to decrease the number of parameters, a convolution with the kernel_size (kernel,kernel) can be splitted into two consecutive convolutions with the kernel_size (kernel,1) and (1,kernel) respectivly
  - numberOfConvs (optional): number of consecutive convolutional building blocks, i.e. Conv2DBlock.
  - activation (optional): string literal or tensorflow activation function object to obtain activation function
  - useResidualConv2DBlock (optional): ads a residual connection in parallel to the Conv2DBlock
  - useResidualIdentityBlock (optional): Whether or not to use the ResidualIdentityBlock instead of the Conv2DBlock
  - residual_cardinality (optional): cardinality for the ResidualIdentityBlock
  - limit_filters (optional): limits the number of filters, which is doubled with every downsampling block
  - n_bottleneck_blocks (optional): Number of consecutive convolution blocks in the bottleneck
  - filters (optional): defines the number of filters to which the input is exposed.
  - dropout_rate (optional): probability of the dropout layer. If the preceeding layer has more than one channel, spatial dropout is applied, otherwise standard dropout
  - useSelfAttention (optional): Determines whether to apply self-attention in the decoder.
  - useSpecNorm (optional): applies spectral normalization to convolutional and dense layers
  - use_bias (optional): determines whether convolutions and dense layers include a bias or not
  - FullyConected (optional): determines whether 1x1 convolutions are replaced by linear layers, which gives the same result, but linear layers are faster. Option: "MLP" or "1x1_conv"
  - padding (optional): padding type. Options are "none", "zero" or "reflection"
  - kernel_initializer (optional): Initialization of the convolutions kernels.
  - gamma_initializer (optional): Initialization of the normalization layers.
  
  Input Shape:
    (batch, height, width, channel)

  Output Shape:
    out1: (batch, 1)
    out2: (batch, height, width, 1)

  Example:
    >>> import DeepSaki
    >>> import tensorflow as tf
    >>> inputs = tf.keras.layers.Input(shape = (256,256,4))
    >>> model = tf.keras.Model(inputs=inputs, outputs=DeepSaki.models.UNetDiscriminator(5).call(inputs))
    >>> model.summary()
    >>> tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True, show_dtype=True, to_file='Unet_discriminator_model.png')
  '''
  def __init__(self,
            number_of_levels,
            upsampling = "transpose_conv", 
            downsampling = "average_pooling",
            kernels = 3,
            first_kernel = 5,
            split_kernels = False,
            numberOfConvs = 2,
            activation = "leaky_relu", 
            useResidualConv2DBlock = False,
            useResidualIdentityBlock = False,
            residual_cardinality = 1,
            limit_filters = 512, 
            n_bottleneck_blocks = 1,  
            filters = 64, 
            dropout_rate = 0.2, 
            useSelfAttention=False,
            useSpecNorm=False,
            use_bias = True,
            FullyConected = "MLP",
            padding = "zero",
            kernel_initializer = DeepSaki.initializer.HeAlphaUniform(),
            gamma_initializer =  DeepSaki.initializer.HeAlphaUniform()
            ):
    super(UNetDiscriminator, self).__init__()
    ch = filters
    self.encoder = DeepSaki.layers.Encoder(number_of_levels=number_of_levels, filters=filters, limit_filters=limit_filters, useResidualConv2DBlock=useResidualConv2DBlock, downsampling=downsampling, kernels=kernels, split_kernels=split_kernels, numberOfConvs=numberOfConvs,activation=activation, first_kernel=first_kernel, useResidualIdentityBlock=useResidualIdentityBlock, channelList=[ch,2*ch,4*ch,8*ch,8*ch], useSpecNorm=useSpecNorm,useSelfAttention=useSelfAttention,outputSkips=True, use_bias = use_bias,residual_cardinality=residual_cardinality,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    self.bottleNeck = DeepSaki.layers.Bottleneck(useResidualIdentityBlock=useResidualIdentityBlock, n_bottleneck_blocks=n_bottleneck_blocks,useResidualConv2DBlock=useResidualConv2DBlock, kernels=kernels, split_kernels=split_kernels,numberOfConvs=numberOfConvs, activation = activation,dropout_rate=dropout_rate, channelList=[16*ch], useSpecNorm=useSpecNorm, use_bias = use_bias,residual_cardinality=residual_cardinality,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    self.decoder = DeepSaki.layers.Decoder(number_of_levels=number_of_levels, upsampling=upsampling, filters=filters, limit_filters=limit_filters, useResidualConv2DBlock=useResidualConv2DBlock, kernels=kernels, split_kernels=split_kernels,numberOfConvs=numberOfConvs,activation=activation,dropout_rate=dropout_rate, useResidualIdentityBlock=useResidualIdentityBlock, channelList=[8*ch,8*ch,4*ch,2*ch,ch], useSpecNorm=useSpecNorm, use_bias = use_bias,residual_cardinality=residual_cardinality,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer,enableSkipConnectionsInput=True)
    if FullyConected == "MLP":
      self.img_reconstruction = DeepSaki.layers.DenseBlock(units = 1, useSpecNorm = useSpecNorm, numberOfLayers = 1, activation = None, applyFinalNormalization = False, use_bias = use_bias, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    elif FullyConected == "1x1_conv": 
      self.img_reconstruction =DeepSaki.layers.Conv2DBlock(filters = 1, useResidualConv2DBlock = False, kernels = 1, split_kernels  = False, numberOfConvs = 1, activation = None,useSpecNorm=useSpecNorm, applyFinalNormalization = False, use_bias = use_bias,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer) 
    self.linear = DeepSaki.layers.DenseBlock(units = 1, useSpecNorm = useSpecNorm, numberOfLayers = 1, activation = None, applyFinalNormalization = False, use_bias = use_bias, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    #To enable mixed precission support for matplotlib and distributed training and to increase training stability
    self.linear_dtype = tf.keras.layers.Activation("linear", dtype = tf.float32)


  def call(self, inputs):
    x = inputs
    encoderOutput, skipConnections = self.encoder(x)
    bottleNeckOutput = self.bottleNeck(encoderOutput)
    decoderOutput = self.decoder([bottleNeckOutput,skipConnections])

    out1 = DeepSaki.layers.GlobalSumPooling2D()(bottleNeckOutput)  
    out1 = self.linear(out1)
    out1 = self.linear_dtype(out1)

    out2 = self.img_reconstruction(decoderOutput)
    out2 = self.linear_dtype(out2)
    return [out1, out2]
