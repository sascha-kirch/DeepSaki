import tensorflow as tf
import DeepSaki as dsk

class UNet(tf.keras.Model):
  '''
  U-Net model with skip conections between encoder and decoder. Input_shape = Output_shape
  args:
  - input_shape: Shape of the input data. E.g. (batch, height, width, channel)
  - number_of_levels (optional): number of down and apsampling levels of the model
  - upsampling (optional): describes the upsampling method used
  - downsampling (optional): describes the downsampling method
  - final_activation (optional): string literal or tensorflow activation function object to obtain activation function for the model's output activation
  - filters (optional): defines the number of filters to which the input is exposed.
  - kernels (optional): size of the convolutions kernels
  - first_kernel (optional): The first convolution can have a different kernel size, to e.g. increase the perceptive field, while the channel depth is still low.
  - useResidualIdentityBlock (optional): Whether or not to use the ResidualIdentityBlock instead of the Conv2DBlock
  - split_kernels (optional): to decrease the number of parameters, a convolution with the kernel_size (kernel,kernel) can be splitted into two consecutive convolutions with the kernel_size (kernel,1) and (1,kernel) respectivly
  - number_of_convs (optional): number of consecutive convolutional building blocks, i.e. Conv2DBlock.
  - activation (optional): string literal or tensorflow activation function object to obtain activation function
  - limit_filters (optional): limits the number of filters, which is doubled with every downsampling block
  - use_residual_Conv2DBlock (optional): ads a residual connection in parallel to the Conv2DBlock
  - useResidualIdentityBlock (optional): Whether or not to use the ResidualIdentityBlock instead of the Conv2DBlock
  - residual_cardinality (optional): cardinality for the ResidualIdentityBlock
  - n_bottleneck_blocks (optional): Number of consecutive convolution blocks in the bottleneck
  - dropout_rate (optional): probability of the dropout layer. If the preceeding layer has more than one channel, spatial dropout is applied, otherwise standard dropout
  - use_spec_norm (optional): applies spectral normalization to convolutional and dense layers
  - use_bias (optional): determines whether convolutions and dense layers include a bias or not
  - useSelfAttention (optional): Determines whether to apply self-attention in the decoder.
  - omit_skips (optional): defines how many layers should not output a skip connection output. Requires outputSkips to be True. E.g. if omit_skips = 2, the first two levels do not output a skip connection, it starts at level 3.
  - FullyConected (optional): determines whether 1x1 convolutions are replaced by linear layers, which gives the same result, but linear layers are faster. Option: "MLP" or "1x1_conv"
  - padding (optional): padding type. Options are "none", "zero" or "reflection"
  - kernel_initializer (optional): Initialization of the convolutions kernels.
  - gamma_initializer (optional): Initialization of the normalization layers.

  Input Shape:
    (batch, height, width, channel)

  Output Shape:
    (batch, height, width, channel)

  Example:
    >>> import DeepSaki as dsk
    >>> import tensorflow as tf
    >>> inputs = tf.keras.layers.Input(shape = (256,256,4))
    >>> model = tf.keras.Model(inputs=inputs, outputs=dsk.model.UNet((256,256,4),5).call(inputs))
    >>> model.summary()
    >>> tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True, show_dtype=True, to_file='Unet_model.png')
  '''
  def __init__(self,
            input_shape,
            number_of_levels = 3,
            upsampling = "transpose_conv",
            downsampling = "conv_stride_2",
            final_activation = "hard_sigmoid",
            filters = 64,
            kernels = 3,
            first_kernel = 5,
            split_kernels = False,
            number_of_convs = 2,
            activation = "leaky_relu",
            limit_filters = 512,
            use_residual_Conv2DBlock = False,
            useResidualIdentityBlock = False,
            residual_cardinality = 1,
            n_bottleneck_blocks = 1,
            dropout_rate = 0.2,
            use_spec_norm = False,
            use_bias = True,
            useSelfAttention=False,
            omit_skips = 0,
            FullyConected = "MLP",
            padding:dsk.layers.PaddingType=dsk.layers.PaddingType.ZERO,
            kernel_initializer = dsk.initializers.HeAlphaUniform(),
            gamma_initializer =  dsk.initializers.HeAlphaUniform()
            ):
    super(UNet, self).__init__()

    self.encoder = dsk.layers.Encoder(number_of_levels=number_of_levels, filters=filters, limit_filters=limit_filters, use_residual_Conv2DBlock=use_residual_Conv2DBlock, downsampling=downsampling, kernels=kernels, split_kernels=split_kernels, number_of_convs=number_of_convs,activation=activation, first_kernel=first_kernel,useResidualIdentityBlock=useResidualIdentityBlock,use_spec_norm=use_spec_norm, omit_skips=omit_skips, outputSkips=True, use_bias = use_bias, residual_cardinality=residual_cardinality,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    self.bottleNeck = dsk.layers.Bottleneck(useResidualIdentityBlock=useResidualIdentityBlock, n_bottleneck_blocks=n_bottleneck_blocks,use_residual_Conv2DBlock=use_residual_Conv2DBlock, kernels=kernels, split_kernels=split_kernels,number_of_convs=number_of_convs,activation = activation, dropout_rate=dropout_rate, use_spec_norm=use_spec_norm, use_bias = use_bias, residual_cardinality=residual_cardinality,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    self.decoder = dsk.layers.Decoder(number_of_levels=number_of_levels, upsampling=upsampling, filters=filters, limit_filters=limit_filters, use_residual_Conv2DBlock=use_residual_Conv2DBlock, kernels=kernels, split_kernels=split_kernels,number_of_convs=number_of_convs,activation=activation,dropout_rate=dropout_rate, useResidualIdentityBlock=useResidualIdentityBlock,use_spec_norm=use_spec_norm,useSelfAttention=useSelfAttention, use_bias = use_bias, residual_cardinality=residual_cardinality,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer,enableSkipConnectionsInput=True)
    if FullyConected == "MLP":
      self.img_reconstruction = dsk.layers.DenseBlock(units = input_shape[-1], use_spec_norm = use_spec_norm, numberOfLayers = 1, activation = final_activation, apply_final_normalization = False, use_bias = use_bias, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    elif FullyConected == "1x1_conv":
      self.img_reconstruction = dsk.layers.Conv2DBlock(filters = input_shape[-1],use_residual_Conv2DBlock = False, kernels = 1, split_kernels  = False, number_of_convs = 1, activation = final_activation, use_spec_norm=use_spec_norm, apply_final_normalization = False, use_bias = use_bias,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    #To enable mixed precission support for matplotlib and distributed training and to increase training stability
    self.linear_dtype = tf.keras.layers.Activation("linear", dtype = tf.float32)

  def call(self, inputs):
    x = inputs
    x, skipConnections = self.encoder(x)
    x = self.bottleNeck(x)
    x = self.decoder([x,skipConnections])
    x = self.img_reconstruction(x)
    x = self.linear_dtype(x)
    return x

class ResNet(tf.keras.Model):
  '''
  ResNet model in autoencoder architecture (encoder, bottleneck, decoder). Input_shape = Output_shape
  args:
  - input_shape: Shape of the input data. E.g. (batch, height, width, channel)
  - number_of_levels (optional): number of down and apsampling levels of the model
  - filters (optional): defines the number of filters to which the input is exposed.
  - split_kernels (optional): to decrease the number of parameters, a convolution with the kernel_size (kernel,kernel) can be splitted into two consecutive convolutions with the kernel_size (kernel,1) and (1,kernel) respectivly
  - kernels (optional): size of the convolutions kernels
  - first_kernel (optional): The first convolution can have a different kernel size, to e.g. increase the perceptive field, while the channel depth is still low.
  - number_of_convs (optional): number of consecutive convolutional building blocks, i.e. Conv2DBlock.
  - activation (optional): string literal or tensorflow activation function object to obtain activation function
  - final_activation (optional): string literal or tensorflow activation function object to obtain activation function for the model's output activation
  - use_residual_Conv2DBlock (optional): ads a residual connection in parallel to the Conv2DBlock
  - useResidualIdentityBlock (optional): Whether or not to use the ResidualIdentityBlock instead of the Conv2DBlock
  - residual_cardinality (optional): cardinality for the ResidualIdentityBlock
  - limit_filters (optional): limits the number of filters, which is doubled with every downsampling block
  - n_bottleneck_blocks (optional): Number of consecutive convolution blocks in the bottleneck
  - upsampling(optional): describes the upsampling method used
  - downsampling(optional): describes the downsampling method
  - dropout_rate (optional): probability of the dropout layer. If the preceeding layer has more than one channel, spatial dropout is applied, otherwise standard dropout
  - use_spec_norm (optional): applies spectral normalization to convolutional and dense layers
  - use_bias (optional): determines whether convolutions and dense layers include a bias or not
  - useSelfAttention (optional): Determines whether to apply self-attention in the decoder.
  - FullyConected (optional): determines whether 1x1 convolutions are replaced by linear layers, which gives the same result, but linear layers are faster. Option: "MLP" or "1x1_conv"
  - padding (optional): padding type. Options are "none", "zero" or "reflection"
  - kernel_initializer (optional): Initialization of the convolutions kernels.
  - gamma_initializer (optional): Initialization of the normalization layers.

  Input Shape:
    (batch, height, width, channel)

  Output Shape:
    (batch, height, width, channel)

  Example:
    >>> import DeepSaki as dsk
    >>> import tensorflow as tf
    >>> inputs = tf.keras.layers.Input(shape = (256,256,4))
    >>> model = tf.keras.Model(inputs=inputs, outputs=dsk.model.ResNet((256,256,4), 5,residual_cardinality=1).call(inputs))
    >>> model.summary()
    >>> tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True, show_dtype=True, to_file='ResNet_model.png')
  '''
  def __init__(self,
            input_shape,
            number_of_levels = 3,
            filters=64,
            split_kernels = False,
            kernels = 3,
            first_kernel = 5,
            number_of_convs = 2,
            activation = "leaky_relu",
            final_activation = "hard_sigmoid",
            use_residual_Conv2DBlock = False,
            useResidualIdentityBlock = True,
            residual_cardinality = 32,
            limit_filters = 512,
            n_bottleneck_blocks = 5,
            upsampling = "transpose_conv",
            downsampling = "average_pooling",
            dropout_rate = 0.2,
            use_spec_norm=False,
            use_bias = True,
            useSelfAttention= False,
            FullyConected = "MLP",
            padding:dsk.layers.PaddingType=dsk.layers.PaddingType.ZERO,
            kernel_initializer = dsk.initializers.HeAlphaUniform(),
            gamma_initializer =  dsk.initializers.HeAlphaUniform()
            ):
    super(ResNet, self).__init__()

    self.encoder = dsk.layers.Encoder(number_of_levels=number_of_levels, filters=filters, limit_filters=limit_filters, use_residual_Conv2DBlock=use_residual_Conv2DBlock, downsampling=downsampling, kernels=kernels, split_kernels=split_kernels, number_of_convs=number_of_convs,activation=activation, first_kernel=first_kernel,useResidualIdentityBlock=useResidualIdentityBlock,use_spec_norm=use_spec_norm, use_bias = use_bias, residual_cardinality=residual_cardinality,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    self.bottleNeck = dsk.layers.Bottleneck(useResidualIdentityBlock=useResidualIdentityBlock, n_bottleneck_blocks=n_bottleneck_blocks,use_residual_Conv2DBlock=use_residual_Conv2DBlock, kernels=kernels, split_kernels=split_kernels,number_of_convs=number_of_convs , activation = activation,dropout_rate=dropout_rate,use_spec_norm=use_spec_norm, use_bias = use_bias, residual_cardinality=residual_cardinality,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    self.decoder = dsk.layers.Decoder(number_of_levels=number_of_levels, upsampling=upsampling, filters=filters, limit_filters=limit_filters, use_residual_Conv2DBlock=use_residual_Conv2DBlock, kernels=kernels, split_kernels=split_kernels,number_of_convs=number_of_convs,activation=activation,dropout_rate=dropout_rate, useResidualIdentityBlock=useResidualIdentityBlock,use_spec_norm=use_spec_norm,useSelfAttention=useSelfAttention, use_bias = use_bias, residual_cardinality=residual_cardinality,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    if FullyConected == "MLP":
      self.img_reconstruction = dsk.layers.DenseBlock(units = input_shape[-1], use_spec_norm = use_spec_norm, numberOfLayers = 1, activation = final_activation, apply_final_normalization = False, use_bias = use_bias, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    elif FullyConected == "1x1_conv":
      self.img_reconstruction = dsk.layers.Conv2DBlock(filters = input_shape[-1],use_residual_Conv2DBlock = False, kernels = 1, split_kernels  = False, number_of_convs = 1, activation = final_activation, use_spec_norm=use_spec_norm, apply_final_normalization = False,use_bias = use_bias,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    #To enable mixed precission support for matplotlib and distributed training and to increase training stability
    self.linear_dtype = tf.keras.layers.Activation("linear", dtype = tf.float32)

  def call(self, inputs):
    x = inputs
    x = self.encoder(x)
    x = self.bottleNeck(x)
    x = self.decoder(x)
    x = self.img_reconstruction(x)
    x = self.linear_dtype(x)
    return x
