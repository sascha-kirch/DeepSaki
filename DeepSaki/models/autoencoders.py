import tensorflow as tf
from DeepSaki.initializers.he_alpha import HeAlphaUniform
from DeepSaki.layers.layer_helper import PaddingType
from DeepSaki.layers.sub_model_composites import Encoder, Decoder,Bottleneck
from DeepSaki.layers.layer_composites import Conv2DBlock, DenseBlock, ScalarGatedSelfAttention

from typing import Tuple, Optional

class UNet(tf.keras.Model):
  """U-Net based autoencoder model with skip conections between encoder and decoder. Input_shape = Output_shape.

  Architecture:
    ```mermaid
    flowchart LR
      i([Input])-->e1
      subgraph Encoder
      e1-->e2-->e3-->ex
      end
      subgraph Bottleneck
      ex-->b1-->bx
      end
      subgraph Decoder
      bx-->dx-->d3-->d2-->d1
      end
      d1-->o([Output])
      e1-->d1
      e2-->d2
      e3-->d3
      ex-->dx
    ```

  Example:
  ```python
  import DeepSaki as dsk
  import tensorflow as tf
  inputs = tf.keras.layers.Input(shape = (256,256,4))
  model = tf.keras.Model(inputs=inputs, outputs=dsk.model.UNet((256,256,4),5).call(inputs))
  model.summary()
  tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True, show_dtype=True, to_file='Unet_model.png')
  ```
  """

  def __init__(self,
            input_shape:Tuple[int,int,int,int],
            number_of_levels:int = 3,
            upsampling:str = "transpose_conv",
            downsampling:str = "conv_stride_2",
            final_activation:str = "hard_sigmoid",
            filters:int = 64,
            kernels:int = 3,
            first_kernel:int = 5,
            split_kernels:bool = False,
            number_of_convs:int = 2,
            activation:str = "leaky_relu",
            limit_filters:int = 512,
            use_residual_Conv2DBlock:bool = False,
            use_ResidualBlock:bool = False,
            residual_cardinality:int = 1,
            n_bottleneck_blocks:int = 1,
            dropout_rate:float = 0.2,
            use_spec_norm:bool = False,
            use_bias:bool = True,
            use_self_attention:bool=False,
            omit_skips:int = 0,
            fully_connected:str = "MLP",
            padding:PaddingType=PaddingType.ZERO,
            kernel_initializer: Optional[tf.keras.initializers.Initializer] = None,
            gamma_initializer: Optional[tf.keras.initializers.Initializer] =  None
            )->None:
    """Initialize the `UNet` object.

    Args:
        input_shape (Tuple[int,int,int,int]): Shape of the input data. E.g. `(batch, height, width, channel)`
        number_of_levels (int, optional): Number of down and upsampling levels of the model. Defaults to 3.
        upsampling (str, optional): Describes the upsampling method used. Defaults to "transpose_conv".
        downsampling (str, optional): Describes the downsampling method. Defaults to "conv_stride_2".
        final_activation (str, optional): String literal or tensorflow activation function object to obtain activation
            function for the model's output activation. Defaults to "hard_sigmoid".
        filters (int, optional): Number of filters for the initial encoder block. Defaults to 64.
        kernels (int, optional): Size of the convolutions kernels. Defaults to 3.
        first_kernel (int, optional): The first convolution can have a different kernel size, to e.g. increase the
            perceptive field, while the channel depth is still low. Defaults to 5.
        split_kernels (bool, optional): To decrease the number of parameters, a convolution with the kernel_size
            `(kernel,kernel)` can be splitted into two consecutive convolutions with the kernel_size `(kernel,1)` and
            `(1,kernel)` respectivly. Defaults to False.
        number_of_convs (int, optional): Number of consecutive convolutional building blocks, i.e. `Conv2DBlock`.
            Defaults to 2.
        activation (str, optional): String literal or tensorflow activation function object to obtain activation
            function. Defaults to "leaky_relu".
        limit_filters (int, optional): Limits the number of filters, which is doubled with every downsampling block.
            Defaults to 512.
        use_residual_Conv2DBlock (bool, optional): Adds a residual connection in parallel to the `Conv2DBlock`. Defaults
            to False.
        use_ResidualBlock (bool, optional): Whether or not to use the `ResidualBlock` instead of the
            `Conv2DBlock`. Defaults to False.
        residual_cardinality (int, optional): Cardinality for the `ResidualBlock`. Defaults to 1.
        n_bottleneck_blocks (int, optional): Number of consecutive convolution blocks in the bottleneck. Defaults to 1.
        dropout_rate (float, optional): Probability of the dropout layer. If the preceeding layer has more than one
            channel, spatial dropout is applied, otherwise standard dropout. Defaults to 0.2.
        use_spec_norm (bool, optional): Applies spectral normalization to convolutional and dense layers. Defaults to
            False.
        use_bias (bool, optional): Whether convolutions and dense layers include a bias or not. Defaults to True.
        use_self_attention (bool, optional): Determines whether to apply self-attention in the decoder. Defaults to
            False.
        omit_skips (int, optional): Defines how many layers should not output a skip connection output. Requires
            `output_skips` to be True. E.g. if `omit_skips = 2`, the first two levels do not output a skip connection,
            it starts at level 3. Defaults to 0.
        fully_connected (str, optional): Determines whether 1x1 convolutions are replaced by linear layers, which gives
            the same result, but linear layers are faster. Option: "MLP" or "1x1_conv". Defaults to "MLP".
        padding (PaddingType, optional): Padding type. Defaults to PaddingType.ZERO.
        kernel_initializer (tf.keras.initializers.Initializer, optional): Initialization of the convolutions kernels.
            Defaults to None.
        gamma_initializer (tf.keras.initializers.Initializer, optional): Initialization of the normalization layers.
            Defaults to None.
    """
    super(UNet, self).__init__()

    self.encoder = Encoder(number_of_levels=number_of_levels, filters=filters, limit_filters=limit_filters, use_residual_Conv2DBlock=use_residual_Conv2DBlock, downsampling=downsampling, kernels=kernels, split_kernels=split_kernels, number_of_convs=number_of_convs,activation=activation, first_kernel=first_kernel,use_ResidualBlock=use_ResidualBlock,use_spec_norm=use_spec_norm, omit_skips=omit_skips, output_skips=True, use_bias = use_bias, residual_cardinality=residual_cardinality,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    self.bottleNeck = Bottleneck(use_ResidualBlock=use_ResidualBlock, n_bottleneck_blocks=n_bottleneck_blocks,use_residual_Conv2DBlock=use_residual_Conv2DBlock, kernels=kernels, split_kernels=split_kernels,number_of_convs=number_of_convs,activation = activation, dropout_rate=dropout_rate, use_spec_norm=use_spec_norm, use_bias = use_bias, residual_cardinality=residual_cardinality,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    self.decoder = Decoder(number_of_levels=number_of_levels, upsampling=upsampling, filters=filters, limit_filters=limit_filters, use_residual_Conv2DBlock=use_residual_Conv2DBlock, kernels=kernels, split_kernels=split_kernels,number_of_convs=number_of_convs,activation=activation,dropout_rate=dropout_rate, use_ResidualBlock=use_ResidualBlock,use_spec_norm=use_spec_norm,use_self_attention=use_self_attention, use_bias = use_bias, residual_cardinality=residual_cardinality,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer,enable_skip_connections_input=True)
    if fully_connected == "MLP":
      self.img_reconstruction = DenseBlock(units = input_shape[-1], use_spec_norm = use_spec_norm, number_of_layers = 1, activation = final_activation, apply_final_normalization = False, use_bias = use_bias, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    elif fully_connected == "1x1_conv":
      self.img_reconstruction = Conv2DBlock(filters = input_shape[-1],use_residual_Conv2DBlock = False, kernels = 1, split_kernels  = False, number_of_convs = 1, activation = final_activation, use_spec_norm=use_spec_norm, apply_final_normalization = False, use_bias = use_bias,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    #To enable mixed precission support for matplotlib and distributed training and to increase training stability
    self.linear_dtype = tf.keras.layers.Activation("linear", dtype = tf.float32)

  def call(self, inputs:tf.Tensor)->tf.Tensor:
    """Calls the `UNet` model.

    Args:
        inputs (tf.Tensor): Tensor of shape (`batch`,`height`,`width`,`channel`).

    Returns:
        tf.Tensor: Tensor of shape (`batch`,`height`,`width`,`channel`)
    """
    x = inputs
    x, skip_connections = self.encoder(x)
    x = self.bottleNeck(x)
    x = self.decoder([x,skip_connections])
    x = self.img_reconstruction(x)
    return self.linear_dtype(x)

class ResNet(tf.keras.Model):
  """ResNet model in autoencoder architecture (encoder, bottleneck, decoder). Input_shape = Output_shape.

  Architecture:
    ```mermaid
    flowchart LR
      i([Input])-->e1
      subgraph Encoder
      e1-->e2-->e3-->ex
      end
      subgraph Bottleneck
      ex-->b1-->bx
      end
      subgraph Decoder
      bx-->dx-->d3-->d2-->d1
      end
      d1-->o([Output])
    ```

  Example:
  ```python
  import DeepSaki as dsk
  import tensorflow as tf
  inputs = tf.keras.layers.Input(shape = (256,256,4))
  model = tf.keras.Model(inputs=inputs, outputs=dsk.model.ResNet((256,256,4), 5,residual_cardinality=1).call(inputs))
  model.summary()
  tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True, show_dtype=True, to_file='ResNet_model.png')
  ```
  """

  def __init__(self,
            input_shape:Tuple[int,int,int,int],
            number_of_levels:int = 3,
            filters:int=64,
            split_kernels:bool = False,
            kernels:int = 3,
            first_kernel:int = 5,
            number_of_convs:int = 2,
            activation:str = "leaky_relu",
            final_activation:str = "hard_sigmoid",
            use_residual_Conv2DBlock:bool = False,
            use_ResidualBlock:bool = True,
            residual_cardinality:int = 32,
            limit_filters:int = 512,
            n_bottleneck_blocks:int = 5,
            upsampling:str = "transpose_conv",
            downsampling:str = "average_pooling",
            dropout_rate:float = 0.2,
            use_spec_norm:bool=False,
            use_bias:bool = True,
            use_self_attention:bool= False,
            fully_connected:str = "MLP",
            padding:PaddingType=PaddingType.ZERO,
            kernel_initializer:Optional[tf.keras.initializers.Initializer] = None,
            gamma_initializer:tf.keras.initializers.Initializer =  HeAlphaUniform()
            ):
    """Initialize the `ResNet` object.

    Args:
        input_shape (Tuple[int,int,int,int]): Shape of the input data. E.g. `(batch, height, width, channel)`
        number_of_levels (int, optional): Number of down and apsampling levels of the model. Defaults to 3.
        filters (int, optional): Number of filters for the initial encoder block. Defaults to 64.
        split_kernels (bool, optional): To decrease the number of parameters, a convolution with the kernel_size
            `(kernel,kernel)` can be splitted into two consecutive convolutions with the kernel_size `(kernel,1)` and
            `(1,kernel)` respectivly. Defaults to False.
        kernels (int, optional): Size of the convolutions kernels. Defaults to 3.
        first_kernel (int, optional): The first convolution can have a different kernel size, to e.g. increase the
            perceptive field, while the channel depth is still low. Defaults to 5.
        number_of_convs (int, optional): Number of consecutive convolutional building blocks, i.e. `Conv2DBlock`.
            Defaults to 2.
        activation (str, optional): String literal or tensorflow activation function object to obtain activation
            function. Defaults to "leaky_relu".
        final_activation (str, optional): String literal or tensorflow activation function object to obtain activation
            function for the model's output activation. Defaults to "hard_sigmoid".
        use_residual_Conv2DBlock (bool, optional): Ads a residual connection in parallel to the `Conv2DBlock`. Defaults
            to False.
        use_ResidualBlock (bool, optional): Whether or not to use the ResidualBlock instead of the
            `Conv2DBlock`. Defaults to False.
        residual_cardinality (int, optional): Cardinality for the ResidualBlock. Defaults to 1.
        limit_filters (int, optional): Limits the number of filters, which is doubled with every downsampling block.
            Defaults to 512.
        n_bottleneck_blocks (int, optional): Number of consecutive convolution blocks in the bottleneck. Defaults to 1.
        upsampling (str, optional): Describes the upsampling method used. Defaults to "transpose_conv".
        downsampling (str, optional): Describes the downsampling method. Defaults to "conv_stride_2".
        dropout_rate (float, optional): Probability of the dropout layer. If the preceeding layer has more than one
            channel, spatial dropout is applied, otherwise standard dropout. Defaults to 0.2.
        use_spec_norm (bool, optional): Applies spectral normalization to convolutional and dense layers. Defaults to
            False.
        use_bias (bool, optional): Whether convolutions and dense layers include a bias or not. Defaults to True.
        use_self_attention (bool, optional): Determines whether to apply self-attention in the decoder. Defaults to
            False.
        fully_connected (str, optional): Determines whether 1x1 convolutions are replaced by linear layers, which gives
            the same result, but linear layers are faster. Option: "MLP" or "1x1_conv". Defaults to "MLP".
        padding (PaddingType, optional): Padding type. Defaults to PaddingType.ZERO.
        kernel_initializer (tf.keras.initializers.Initializer, optional): Initialization of the convolutions kernels.
            Defaults to None.
        gamma_initializer (tf.keras.initializers.Initializer, optional): Initialization of the normalization layers.
            Defaults to None.
    """
    super(ResNet, self).__init__()

    self.encoder = Encoder(number_of_levels=number_of_levels, filters=filters, limit_filters=limit_filters, use_residual_Conv2DBlock=use_residual_Conv2DBlock, downsampling=downsampling, kernels=kernels, split_kernels=split_kernels, number_of_convs=number_of_convs,activation=activation, first_kernel=first_kernel,use_ResidualBlock=use_ResidualBlock,use_spec_norm=use_spec_norm, use_bias = use_bias, residual_cardinality=residual_cardinality,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    self.bottleNeck = Bottleneck(use_ResidualBlock=use_ResidualBlock, n_bottleneck_blocks=n_bottleneck_blocks,use_residual_Conv2DBlock=use_residual_Conv2DBlock, kernels=kernels, split_kernels=split_kernels,number_of_convs=number_of_convs , activation = activation,dropout_rate=dropout_rate,use_spec_norm=use_spec_norm, use_bias = use_bias, residual_cardinality=residual_cardinality,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    self.decoder = Decoder(number_of_levels=number_of_levels, upsampling=upsampling, filters=filters, limit_filters=limit_filters, use_residual_Conv2DBlock=use_residual_Conv2DBlock, kernels=kernels, split_kernels=split_kernels,number_of_convs=number_of_convs,activation=activation,dropout_rate=dropout_rate, use_ResidualBlock=use_ResidualBlock,use_spec_norm=use_spec_norm,use_self_attention=use_self_attention, use_bias = use_bias, residual_cardinality=residual_cardinality,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    if fully_connected == "MLP":
      self.img_reconstruction = DenseBlock(units = input_shape[-1], use_spec_norm = use_spec_norm, number_of_layers = 1, activation = final_activation, apply_final_normalization = False, use_bias = use_bias, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    elif fully_connected == "1x1_conv":
      self.img_reconstruction = Conv2DBlock(filters = input_shape[-1],use_residual_Conv2DBlock = False, kernels = 1, split_kernels  = False, number_of_convs = 1, activation = final_activation, use_spec_norm=use_spec_norm, apply_final_normalization = False,use_bias = use_bias,padding = padding, kernel_initializer = kernel_initializer, gamma_initializer = gamma_initializer)
    #To enable mixed precission support for matplotlib and distributed training and to increase training stability
    self.linear_dtype = tf.keras.layers.Activation("linear", dtype = tf.float32)

  def call(self, inputs:tf.Tensor)->tf.Tensor:
    """Calls the `ResNet` model.

    Args:
        inputs (tf.Tensor): Tensor of shape (`batch`,`height`,`width`,`channel`).

    Returns:
        tf.Tensor: Tensor of shape (`batch`,`height`,`width`,`channel`)
    """
    x = inputs
    x = self.encoder(x)
    x = self.bottleNeck(x)
    x = self.decoder(x)
    x = self.img_reconstruction(x)
    return self.linear_dtype(x)
