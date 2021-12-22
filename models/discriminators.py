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
    - filters (optional, default:64): defines the number of filters to which the input is exposed.
    - downsampling(optional, default: "conv_stride_2"): describes the downsampling method 
    - kernels (optional, default: 3): size of the kernel for convolutional layers
    - first_kernel (optional, default: 5): The first convolution can have a different kernel size, to e.g. increase the perceptive field, while the channel depth is still low.
    - split_kernels (optional, default: False): to decrease the number of parameters, a convolution with the kernel_size (kernel,kernel) can be splitted into two consecutive convolutions with the kernel_size (kernel,1) and (1,kernel) respectivly
    - numberOfConvs (optional, default: 2): number of consecutive convolutional building blocks, i.e. Conv2DBlock.
    - activation (optional, default: "leaky_relu"): string literal to obtain activation function
    - dropout_rate (optional, default: 0.2): probability of the dropout layer. If the preceeding layer has more than one channel, spatial dropout is applied, otherwise standard dropout
    - useSpecNorm (optional, default: False): applies spectral normalization to convolutional and dense layers
    - use_bias (optional, default: True): determines whether convolutions and dense layers include a bias or not
    - padding (optional, default: "none"): padding type. Options are "none", "zero" or "reflection"
    - FullyConected (optional, default: "MLP"): determines whether 1x1 convolutions are replaced by linear layers, which gives the same result, but linear layers are faster. Option: "MLP" or "1x1_conv"
    - useSelfAttention (optional, default: False): Determines whether to apply self-attention after the encoder before branching.
  
  output call():
    - out1: content output
    - out2: layout output
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

    out2 = self.lay1(F)
    out2 = self.lay2(out2)
    out2 = self.lay3(out2)
    out2 = self.lay4(out2)

    return [out1, out2]

#Testcode
#inputs = tf.keras.layers.Input(shape = (256,256,4))
#model = tf.keras.Model(inputs=inputs, outputs=LayoutContentDiscriminator((256,256,4)).call(inputs))
#model.summary()
#tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True, show_dtype=True, to_file='Unet_discriminator_model.png')