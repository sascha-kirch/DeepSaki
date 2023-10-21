from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import tensorflow as tf

from DeepSaki.initializers.he_alpha import HeAlphaUniform
from DeepSaki.layers.layer_composites import Conv2DBlock
from DeepSaki.layers.layer_composites import DownSampleBlock
from DeepSaki.layers.layer_composites import ResBlockDown
from DeepSaki.layers.layer_composites import ResBlockUp
from DeepSaki.layers.layer_composites import ResidualBlock
from DeepSaki.layers.layer_composites import ScalarGatedSelfAttention
from DeepSaki.layers.layer_composites import UpSampleBlock
from DeepSaki.layers.layer_helper import PaddingType
from DeepSaki.layers.layer_helper import dropout_func

class Encoder(tf.keras.layers.Layer):
    """Combines conv blocks with down sample blocks.

    The spatial width is halved with every level while the channel depth is doubled.
    Can be combined with `dsk.layers.Decoder` and `dsk.layers.Bottleneck` to form an auto encoder model.

    Tipp:
        Checkout the dsk.models api to find models using this layer.
    """

    def __init__(
        self,
        number_of_levels: int = 3,
        filters: int = 64,
        limit_filters: int = 1024,
        use_residual_Conv2DBlock: bool = False,
        downsampling: str = "conv_stride_2",
        kernels: int = 3,
        split_kernels: bool = False,
        number_of_convs: int = 2,
        activation: str = "leaky_relu",
        first_kernel: Optional[int] = None,
        use_ResidualBlock: bool = False,
        residual_cardinality: int = 1,
        channel_list: Optional[List[int]] = None,
        use_spec_norm: bool = False,
        use_bias: bool = True,
        dropout_rate: float = 0.0,
        use_self_attention: bool = False,
        omit_skips: int = 0,
        padding: PaddingType = PaddingType.ZERO,
        output_skips: bool = False,
        kernel_initializer: Optional[tf.keras.initializers.Initializer] = None,
        gamma_initializer: Optional[tf.keras.initializers.Initializer] = None,
    ) -> None:
        """Initializes the `Encoder` layer.

        Args:
            number_of_levels (int, optional): Number of downsampling levels of the model. Defaults to 3.
            filters (int, optional): Number of filters for the initial encoder block. Defaults to 64.
            limit_filters (int, optional): Limits the number of filters, which is doubled with every downsampling block.
                Defaults to 1024.
            use_residual_Conv2DBlock (bool, optional): Adds a residual connection in parallel to the `Conv2DBlock`.
                Defaults to False.
            downsampling (str, optional): Describes the downsampling method. Defaults to "conv_stride_2".
            kernels (int, optional): Size of the convolutions kernels. Defaults to 3.
            split_kernels (bool, optional): To decrease the number of parameters, a convolution with the kernel_size
                `(kernel,kernel)` can be splitted into two consecutive convolutions with the kernel_size `(kernel,1)` and
                `(1,kernel)` respectivly. Defaults to False.
            number_of_convs (int, optional): Number of consecutive convolutional building blocks, i.e. `Conv2DBlock`.
                Defaults to 2.
            activation (str, optional): String literal or tensorflow activation function object to obtain activation
                function. Defaults to "leaky_relu".
            first_kernel (Optional[int], optional): The first convolution can have a different kernel size, to e.g.
                increase the perceptive field, while the channel depth is still low. Defaults to None.
            use_ResidualBlock (bool, optional): Whether or not to use the `ResidualBlock` instead of the
                `Conv2DBlock`. Defaults to False.
            residual_cardinality (int, optional): Cardinality for the `ResidualBlock`. Defaults to 1.
            channel_list (Optional[List[int]], optional): alternativly to number_of_layers and filters, a list with the
                disired filters for each level can be provided. e.g. channel_list = [64, 128, 256] results in a 3-level
                Encoder with 64, 128, 256 filters for level 1, 2 and 3 respectivly. Defaults to None.
            use_spec_norm (bool, optional): Applies spectral normalization to convolutional and dense layers.
                Defaults to False.
            use_bias (bool, optional): Whether convolutions and dense layers include a bias or not. Defaults to True.
            dropout_rate (float, optional): Probability of the dropout layer. If the preceeding layer has more than one
                channel, spatial dropout is applied, otherwise standard dropout. Defaults to 0.0.
            use_self_attention (bool, optional): Determines whether to apply self-attention in the encoder. Defaults to False.
            omit_skips (int, optional): Defines how many layers should not output a skip connection output. Requires
                `output_skips` to be True. E.g. if `omit_skips = 2`, the first two levels do not output a skip connection,
                it starts at level 3. Defaults to 0.
            padding (PaddingType, optional): Padding type. Defaults to PaddingType.ZERO.
            output_skips (bool, optional): If true, ski connections are output. Could be used to attach an encoder to a
                decoder model. Defaults to False.
            kernel_initializer (tf.keras.initializers.Initializer, optional): Initialization of the convolutions kernels.
                Defaults to None.
            gamma_initializer (tf.keras.initializers.Initializer, optional): Initialization of the normalization layers.
                Defaults to None.
        """
        super(Encoder, self).__init__()
        self.number_of_levels = number_of_levels
        self.filters = filters
        self.limit_filters = limit_filters
        self.use_residual_Conv2DBlock = use_residual_Conv2DBlock
        self.downsampling = downsampling
        self.kernels = kernels
        self.split_kernels = split_kernels
        self.number_of_convs = number_of_convs
        self.activation = activation
        self.first_kernel = first_kernel
        self.use_ResidualBlock = use_ResidualBlock
        self.residual_cardinality = residual_cardinality
        self.channel_list = channel_list
        self.use_spec_norm = use_spec_norm
        self.dropout_rate = dropout_rate
        self.use_self_attention = use_self_attention
        self.omit_skips = omit_skips
        self.padding = padding
        self.output_skips = output_skips
        self.use_bias = use_bias
        self.kernel_initializer = HeAlphaUniform() if kernel_initializer is None else kernel_initializer
        self.gamma_initializer = HeAlphaUniform() if gamma_initializer is None else gamma_initializer

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer depending on the `input_shape` (output shape of the previous layer).

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor to this layer.
        """
        super(Encoder, self).build(input_shape)

        if self.channel_list is None:
            self.channel_list = [min(self.filters * 2**i, self.limit_filters) for i in range(self.number_of_levels)]
        else:
            self.number_of_levels = len(self.channel_list)

        self.encoderBlocks = []
        self.downSampleBlocks = []

        if self.use_self_attention:
            self.SA = ScalarGatedSelfAttention(
                use_spec_norm=self.use_spec_norm,
                intermediate_channel=None,
                kernel_initializer=self.kernel_initializer,
                gamma_initializer=self.gamma_initializer,
            )
        else:
            self.SA = None

        for i, ch in enumerate(self.channel_list):
            encoder_kernels = self.first_kernel if i == 0 and self.first_kernel else self.kernels

            if self.use_ResidualBlock:
                self.encoderBlocks.append(
                    ResidualBlock(
                        filters=ch,
                        activation=self.activation,
                        kernels=encoder_kernels,
                        number_of_blocks=self.number_of_convs,
                        use_spec_norm=self.use_spec_norm,
                        dropout_rate=self.dropout_rate,
                        use_bias=self.use_bias,
                        residual_cardinality=self.residual_cardinality,
                        padding=self.padding,
                        kernel_initializer=self.kernel_initializer,
                        gamma_initializer=self.gamma_initializer,
                    )
                )
                self.downSampleBlocks.append(
                    ResBlockDown(
                        activation=self.activation,
                        use_spec_norm=self.use_spec_norm,
                        use_bias=self.use_bias,
                        padding=self.padding,
                        kernel_initializer=self.kernel_initializer,
                        gamma_initializer=self.gamma_initializer,
                    )
                )
            else:
                self.encoderBlocks.append(
                    Conv2DBlock(
                        filters=ch,
                        use_residual_Conv2DBlock=self.use_residual_Conv2DBlock,
                        kernels=encoder_kernels,
                        split_kernels=self.split_kernels,
                        activation=self.activation,
                        number_of_convs=self.number_of_convs,
                        use_spec_norm=self.use_spec_norm,
                        dropout_rate=self.dropout_rate,
                        padding=self.padding,
                        use_bias=self.use_bias,
                        kernel_initializer=self.kernel_initializer,
                        gamma_initializer=self.gamma_initializer,
                    )
                )
                self.downSampleBlocks.append(
                    DownSampleBlock(
                        downsampling=self.downsampling,
                        activation=self.activation,
                        kernels=encoder_kernels,
                        use_spec_norm=self.use_spec_norm,
                        padding=self.padding,
                        use_bias=self.use_bias,
                        kernel_initializer=self.kernel_initializer,
                        gamma_initializer=self.gamma_initializer,
                    )
                )

    def call(self, inputs: tf.Tensor) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """Calls the `Encoder` layer.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch, height, width, channel)

        Raises:
            ValueError: If layer has not been built by calling build() on to layer.

        Returns:
            If `output_skips=False` only the final output of the Encoder is returned as a tensor of shape
                (`batch`, `height/2**number_of_levels`, `width/2**number_of_levels`,
                `min(channel*2**number_of_levels, limit_filters)`. If `output_skips=True` additionally returns a tensor
                of tensor (one for each level of the encoder) of shape (`batch`, `height/2**level`, `width/2**level`,
                `min(channel*2**level, limit_filters)`.
        """
        if not self.built:
            raise ValueError("This model has not yet been built.")

        x = inputs
        skips = []

        for level in range(self.number_of_levels):
            if level == 3 and self.SA is not None:
                x = self.SA(x)
            skip = self.encoderBlocks[level](x)
            x = self.downSampleBlocks[level](skip)
            if self.output_skips:
                if level >= self.omit_skips:  # omit the first skip connection
                    skips.append(skip)
                else:
                    skips.append(None)

        if self.output_skips:
            return x, skips
        return x

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(Encoder, self).get_config()
        config.update(
            {
                "number_of_levels": self.number_of_levels,
                "filters": self.filters,
                "limit_filters": self.limit_filters,
                "use_residual_Conv2DBlock": self.use_residual_Conv2DBlock,
                "downsampling": self.downsampling,
                "kernels": self.kernels,
                "split_kernels": self.split_kernels,
                "number_of_convs": self.number_of_convs,
                "activation": self.activation,
                "first_kernel": self.first_kernel,
                "use_ResidualBlock": self.use_ResidualBlock,
                "residual_cardinality": self.residual_cardinality,
                "channel_list": self.channel_list,
                "use_spec_norm": self.use_spec_norm,
                "use_bias": self.use_bias,
                "dropout_rate": self.dropout_rate,
                "use_self_attention": self.use_self_attention,
                "omit_skips": self.omit_skips,
                "padding": self.padding,
                "output_skips": self.output_skips,
                "kernel_initializer": self.kernel_initializer,
                "gamma_initializer": self.gamma_initializer,
            }
        )
        return config


class Bottleneck(tf.keras.layers.Layer):
    """Bottlenecks are sub-model blocks in auto-encoder-like models such as UNet or ResNet.

    It is composed of multiple convolution blocks which might have residuals.

    Can be combined with `dsk.layers.Encoder` and `dsk.layers.Decoder` to form an auto encoder model.

    Tipp:
        Checkout the dsk.models api to find models using this layer.

    """

    def __init__(
        self,
        n_bottleneck_blocks: int = 3,
        kernels: int = 3,
        split_kernels: bool = False,
        number_of_convs: int = 2,
        use_residual_Conv2DBlock: bool = True,
        use_ResidualBlock: bool = False,
        activation: str = "leaky_relu",
        dropout_rate: float = 0.2,
        channel_list: Optional[List[int]] = None,
        use_spec_norm: bool = False,
        use_bias: bool = True,
        residual_cardinality: int = 1,
        padding: PaddingType = PaddingType.ZERO,
        kernel_initializer: Optional[tf.keras.initializers.Initializer] = None,
        gamma_initializer: Optional[tf.keras.initializers.Initializer] = None,
    ) -> None:
        """Initializes the `Bottleneck` layer.

        Args:
            n_bottleneck_blocks (int, optional): Number of consecutive blocks. Defaults to 3.
            kernels (int, optional): Size of the convolutions kernels. Defaults to 3.
            split_kernels (bool, optional): To decrease the number of parameters, a convolution with the kernel_size
                `(kernel,kernel)` can be splitted into two consecutive convolutions with the kernel_size `(kernel,1)`
                and `(1,kernel)` respectivly. Defaults to False.
            number_of_convs (int, optional): : Number of consecutive conv layers within a basic building block.
                Defaults to 2.
            use_residual_Conv2DBlock (bool, optional): Adds a residual connection in parallel to the `Conv2DBlock`.
                Defaults to True.
            use_ResidualBlock (bool, optional): Whether or not to use the `ResidualBlock` instead of the
                `Conv2DBlock`. Defaults to False.
            activation (str, optional): String literal or tensorflow activation function object to obtain activation
                function. Defaults to "leaky_relu".
            dropout_rate (float, optional): Probability of the dropout layer. If the preceeding layer has more than one
                channel, spatial dropout is applied, otherwise standard dropout. Defaults to 0.2.
            channel_list (Optional[List[int]], optional): Alternativly to number_of_layers and filters, a list with the
                disired filters for each level can be provided. e.g. channel_list = [64, 128, 256] results in a 3-level
                Decoder with 64, 128, 256 filters for level 1, 2 and 3 respectivly. Defaults to None.
            use_spec_norm (bool, optional): Applies spectral normalization to convolutional and dense layers. Defaults
                to False.
            use_bias (bool, optional): Whether convolutions and dense layers include a bias or not. Defaults to True.
            residual_cardinality (int, optional): Cardinality for the `ResidualBlock`. Defaults to 1.
            padding (PaddingType, optional): Padding type. Defaults to PaddingType.ZERO.
            kernel_initializer (tf.keras.initializers.Initializer, optional): Initialization of the convolutions kernels.
                Defaults to None.
            gamma_initializer (tf.keras.initializers.Initializer, optional): Initialization of the normalization layers.
                Defaults to None.
        """
        super(Bottleneck, self).__init__()
        self.use_ResidualBlock = use_ResidualBlock
        self.n_bottleneck_blocks = n_bottleneck_blocks
        self.use_residual_Conv2DBlock = use_residual_Conv2DBlock
        self.kernels = kernels
        self.split_kernels = split_kernels
        self.number_of_convs = number_of_convs
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.channel_list = channel_list
        self.use_spec_norm = use_spec_norm
        self.use_bias = use_bias
        self.residual_cardinality = residual_cardinality
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.gamma_initializer = gamma_initializer

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer depending on the `input_shape` (output shape of the previous layer).

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor to this layer.
        """
        super(Bottleneck, self).build(input_shape)

        if self.channel_list is None:
            ch = input_shape[-1]
            self.channel_list = [ch for i in range(self.n_bottleneck_blocks)]

        self.layers = []
        for ch in self.channel_list:
            if self.use_ResidualBlock:
                self.layers.append(
                    ResidualBlock(
                        activation=self.activation,
                        filters=ch,
                        kernels=self.kernels,
                        number_of_blocks=self.number_of_convs,
                        use_spec_norm=self.use_spec_norm,
                        use_bias=self.use_bias,
                        residual_cardinality=self.residual_cardinality,
                        padding=self.padding,
                        kernel_initializer=self.kernel_initializer,
                        gamma_initializer=self.gamma_initializer,
                    )
                )
            else:
                self.layers.append(
                    Conv2DBlock(
                        filters=ch,
                        use_residual_Conv2DBlock=self.use_residual_Conv2DBlock,
                        kernels=self.kernels,
                        split_kernels=self.split_kernels,
                        number_of_convs=self.number_of_convs,
                        activation=self.activation,
                        use_spec_norm=self.use_spec_norm,
                        use_bias=self.use_bias,
                        padding=self.padding,
                        kernel_initializer=self.kernel_initializer,
                        gamma_initializer=self.gamma_initializer,
                    )
                )

        self.dropout = dropout_func(self.channel_list[-1], self.dropout_rate)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the `Bottleneck` layer.

        Args:
            inputs (tf.Tensor): Input tensor of shape (batch, height, width, channel)

        Raises:
            ValueError: If layer has not been built by calling build() on to layer.

        Returns:
            Tensor of shape (batch, height, width, channel)
        """
        if not self.built:
            raise ValueError("This model has not yet been built.")
        x = inputs

        for layer in self.layers:
            x = layer(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(Bottleneck, self).get_config()
        config.update(
            {
                "use_ResidualBlock": self.use_ResidualBlock,
                "n_bottleneck_blocks": self.n_bottleneck_blocks,
                "use_residual_Conv2DBlock": self.use_residual_Conv2DBlock,
                "kernels": self.kernels,
                "split_kernels": self.split_kernels,
                "number_of_convs": self.number_of_convs,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
                "use_spec_norm": self.use_spec_norm,
                "use_bias": self.use_bias,
                "channel_list": self.channel_list,
                "residual_cardinality": self.residual_cardinality,
                "padding": self.padding,
                "kernel_initializer": self.kernel_initializer,
                "gamma_initializer": self.gamma_initializer,
            }
        )
        return config


class Decoder(tf.keras.layers.Layer):
    """Combines conv blocks with up sample blocks.

    The spatial width is doubled with every level while the channel depth is halfed.
    Can be combined with `dsk.layers.Encoder` and `dsk.layers.Bottleneck` to form an auto encoder model.

    Tipp:
        Checkout the dsk.models api to find models using this layer.
    """

    def __init__(
        self,
        number_of_levels: int = 3,
        upsampling: str = "2D_upsample_and_conv",
        filters: int = 64,
        limit_filters: int = 1024,
        use_residual_Conv2DBlock: bool = False,
        kernels: int = 3,
        split_kernels: bool = False,
        number_of_convs: int = 2,
        activation: str = "leaky_relu",
        dropout_rate: float = 0.2,
        use_ResidualBlock: bool = False,
        residual_cardinality: int = 1,
        channel_list: Optional[List[int]] = None,
        use_spec_norm: bool = False,
        use_bias: bool = True,
        use_self_attention: bool = False,
        enable_skip_connections_input: bool = False,
        padding: PaddingType = PaddingType.ZERO,
        kernel_initializer: Optional[tf.keras.initializers.Initializer] = None,
        gamma_initializer: Optional[tf.keras.initializers.Initializer] = None,
    ) -> None:
        """Initializes the `Decoder` layer.

        Args:
            number_of_levels (int, optional): Number levels in the decoder. Effectivly the number of convolution and
                upsample pairs. Defaults to 3.
            upsampling (str, optional): Describes the upsampling method used. Defaults to "2D_upsample_and_conv".
            filters (int, optional): Base size of filters the is doubled with every level of the decoder.
                Defaults to 64.
            limit_filters (int, optional): Limits the number of filters, which is doubled with every downsampling block.
                Defaults to 512.
            use_residual_Conv2DBlock (bool, optional):Adds a residual connection in parallel to the `Conv2DBlock`.
                Defaults to False.
            kernels (int, optional): Size of the convolutions kernels. Defaults to 3.
            split_kernels (bool, optional): To decrease the number of parameters, a convolution with the kernel_size
                `(kernel,kernel)` can be splitted into two consecutive convolutions with the kernel_size `(kernel,1)` and
                `(1,kernel)` respectivly. Defaults to False.
            number_of_convs (int, optional): Number of consecutive convolutional building blocks, i.e. `Conv2DBlock`.
                Defaults to 2.
            activation (str, optional): String literal or tensorflow activation function object to obtain activation
                function. Defaults to "leaky_relu".
            dropout_rate (float, optional): Probability of the dropout layer. If the preceeding layer has more than one
                channel, spatial dropout is applied, otherwise standard dropout. Defaults to 0.2.
            use_ResidualBlock (bool, optional): Whether or not to use the `ResidualBlock` instead of the
                `Conv2DBlock`. Defaults to False.
            residual_cardinality (int, optional): Cardinality for the `ResidualBlock`. Defaults to 1.
            channel_list (Optional[List[int]], optional): Alternativly to number_of_layers and filters, a list with the
                disired filters for each level can be provided. e.g. channel_list = [64, 128, 256] results in a 3-level
                Decoder with 64, 128, 256 filters for level 1, 2 and 3 respectivly. Defaults to None.
            use_spec_norm (bool, optional): Applies spectral normalization to convolutional and dense layers. Defaults to False.
            use_bias (bool, optional): Whether convolutions and dense layers include a bias or not. Defaults to True.
            use_self_attention (bool, optional): Determines whether to apply self-attention in the decoder. Defaults to False.
            enable_skip_connections_input (bool, optional): Whether or not to input skip connections at each level. Defaults to False.
            padding (PaddingType, optional): Padding type. Defaults to PaddingType.ZERO.
            kernel_initializer (tf.keras.initializers.Initializer, optional): Initialization of the convolutions kernels.
                Defaults to None.
            gamma_initializer (tf.keras.initializers.Initializer, optional): Initialization of the normalization layers.
                Defaults to None.
        """
        super(Decoder, self).__init__()
        self.number_of_levels = number_of_levels
        self.filters = filters
        self.upsampling = upsampling
        self.limit_filters = limit_filters
        self.use_residual_Conv2DBlock = use_residual_Conv2DBlock
        self.kernels = kernels
        self.split_kernels = split_kernels
        self.number_of_convs = number_of_convs
        self.activation = activation
        self.use_ResidualBlock = use_ResidualBlock
        self.channel_list = channel_list
        self.use_spec_norm = use_spec_norm
        self.use_bias = use_bias
        self.dropout_rate = dropout_rate
        self.use_self_attention = use_self_attention
        self.enable_skip_connections_input = enable_skip_connections_input
        self.residual_cardinality = residual_cardinality
        self.padding = padding

        self.kernel_initializer = HeAlphaUniform() if kernel_initializer is None else kernel_initializer
        self.gamma_initializer = HeAlphaUniform() if gamma_initializer is None else gamma_initializer

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer depending on the `input_shape` (output shape of the previous layer).

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor to this layer.
        """
        super(Decoder, self).build(input_shape)

        if self.channel_list is None:
            self.channel_list = [
                min(self.filters * 2**i, self.limit_filters) for i in reversed(range(self.number_of_levels))
            ]
        else:
            self.number_of_levels = len(self.channel_list)

        self.decoderBlocks = []
        self.upSampleBlocks = []

        if self.use_self_attention:
            self.SA = ScalarGatedSelfAttention(
                use_spec_norm=self.use_spec_norm,
                intermediate_channel=None,
                kernel_initializer=self.kernel_initializer,
                gamma_initializer=self.gamma_initializer,
            )
        else:
            self.SA = None

        for i, ch in enumerate(self.channel_list):
            dropout_rate = self.dropout_rate if i < int(self.number_of_levels / 2) else 0

            if self.use_ResidualBlock:
                self.decoderBlocks.append(
                    ResidualBlock(
                        filters=ch,
                        activation=self.activation,
                        kernels=self.kernels,
                        number_of_blocks=self.number_of_convs,
                        use_spec_norm=self.use_spec_norm,
                        dropout_rate=dropout_rate,
                        use_bias=self.use_bias,
                        residual_cardinality=self.residual_cardinality,
                        padding=self.padding,
                        kernel_initializer=self.kernel_initializer,
                        gamma_initializer=self.gamma_initializer,
                    )
                )
                self.upSampleBlocks.append(
                    ResBlockUp(
                        activation=self.activation,
                        use_spec_norm=self.use_spec_norm,
                        use_bias=self.use_bias,
                        padding=self.padding,
                        kernel_initializer=self.kernel_initializer,
                        gamma_initializer=self.gamma_initializer,
                    )
                )
            else:
                self.decoderBlocks.append(
                    Conv2DBlock(
                        filters=ch,
                        use_residual_Conv2DBlock=self.use_residual_Conv2DBlock,
                        kernels=self.kernels,
                        split_kernels=self.split_kernels,
                        activation=self.activation,
                        number_of_convs=self.number_of_convs,
                        dropout_rate=dropout_rate,
                        use_spec_norm=self.use_spec_norm,
                        use_bias=self.use_bias,
                        padding=self.padding,
                        kernel_initializer=self.kernel_initializer,
                        gamma_initializer=self.gamma_initializer,
                    )
                )
                self.upSampleBlocks.append(
                    UpSampleBlock(
                        kernels=self.kernels,
                        upsampling=self.upsampling,
                        activation=self.activation,
                        use_spec_norm=self.use_spec_norm,
                        use_bias=self.use_bias,
                        padding=self.padding,
                        kernel_initializer=self.kernel_initializer,
                        gamma_initializer=self.gamma_initializer,
                    )
                )

    def call(self, inputs: Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]) -> tf.Tensor:
        """Calls the `Decoder` layer.

        Args:
            inputs (Union[tf.Tensor , Tuple[tf.Tensor, tf.Tensor]]): If `enable_skip_connections_input=False` only
                inputs a tensor of shape (batch, height, width, min(channel*2**level, limit_filters)). If
                `enable_skip_connections_input=True`, additonally at every level of the decoder, skip connections from
                an encoder can be inserted.

        Raises:
            ValueError: If layer has not been built by calling build() on to layer.

        Returns:
            tf.Tensor: Tensor of shape (`batch`, `height*2**number_of_levels`, `width*2**number_of_levels`,`filters`).
        """
        if not self.built:
            raise ValueError("This model has not yet been built.")
        skip_connections = None
        if self.enable_skip_connections_input:
            x, skip_connections = inputs
        else:
            x = inputs

        for level in range(self.number_of_levels):
            if level == 3 and self.SA is not None:
                x = self.SA(x)
            x = self.upSampleBlocks[level](x)
            if skip_connections is not None:
                x = tf.keras.layers.concatenate([x, skip_connections[self.number_of_levels - (level + 1)]])
            x = self.decoderBlocks[level](x)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(Decoder, self).get_config()
        config.update(
            {
                "number_of_levels": self.number_of_levels,
                "filters": self.filters,
                "limit_filters": self.limit_filters,
                "use_residual_Conv2DBlock": self.use_residual_Conv2DBlock,
                "upsampling": self.upsampling,
                "kernels": self.kernels,
                "split_kernels": self.split_kernels,
                "number_of_convs": self.number_of_convs,
                "activation": self.activation,
                "use_ResidualBlock": self.use_ResidualBlock,
                "residual_cardinality": self.residual_cardinality,
                "channel_list": self.channel_list,
                "use_spec_norm": self.use_spec_norm,
                "dropout_rate": self.dropout_rate,
                "use_self_attention": self.use_self_attention,
                "enable_skip_connections_input": self.enable_skip_connections_input,
                "padding": self.padding,
                "kernel_initializer": self.kernel_initializer,
                "gamma_initializer": self.gamma_initializer,
            }
        )
        return config
