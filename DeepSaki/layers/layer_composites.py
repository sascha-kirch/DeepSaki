from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import tensorflow as tf
import tensorflow_addons as tfa

from DeepSaki.initializers.he_alpha import HeAlphaUniform
from DeepSaki.layers.layer_helper import PaddingType,pad_func, dropout_func


class Conv2DSplitted(tf.keras.layers.Layer):
    """
    To decrease the number of parameters, a convolution with the kernel_size (kernel,kernel) can be splitted into two consecutive convolutions with the kernel_size (kernel,1) and (1,kernel) respectivly
    args:
      - filters: number of filters in the output feature map
      - kernels: size of the convolutions kernels, which will be translated to (kernels, 1) & (1,kernels) for the first and seccond convolution respectivly
      - use_spec_norm (optional, default: False): applies spectral normalization to convolutional  layers
      - strides (optional, default: (1,1)): stride of the filter
      - use_bias (optional, default: True): determines whether convolutions layers include a bias or not
      - kernel_initializer (optional, default: HeAlphaUniform()): Initialization of the convolutions kernels.
    """

    def __init__(
        self,
        filters: int,
        kernels: int,
        use_spec_norm: bool = False,
        strides: Tuple[int, int] = (1, 1),
        use_bias: bool = True,
        kernel_initializer: tf.keras.initializers.Initializer = HeAlphaUniform(),
    ) -> None:
        super(Conv2DSplitted, self).__init__()
        self.filters = filters
        self.kernels = kernels
        self.use_spec_norm = use_spec_norm
        self.strides = strides
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer

        self.conv1 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(1, kernels),
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
            strides=strides,
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=(kernels, 1),
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
            strides=strides,
        )

        if use_spec_norm:
            self.conv1 = tfa.layers.SpectralNormalization(self.conv1)
            self.conv2 = tfa.layers.SpectralNormalization(self.conv2)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.conv1(inputs)
        return self.conv2(x)

    def get_config(self) -> Dict[str, Any]:
        config = super(Conv2DSplitted, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "kernels": self.kernels,
                "use_spec_norm": self.use_spec_norm,
                "strides": self.strides,
                "use_bias": self.use_bias,
                "kernel_initializer": self.kernel_initializer,
            }
        )
        return config


class Conv2DBlock(tf.keras.layers.Layer):
    """
    Wraps a two-dimensional convolution into a more complex building block
    args:
      - filters: number of filters in the output feature map
      - kernels: size of the convolutions kernels
      - use_residual_Conv2DBlock (optional, default: False):
      - split_kernels (optional, default: False): to decrease the number of parameters, a convolution with the kernel_size (kernel,kernel) can be splitted into two consecutive convolutions with the kernel_size (kernel,1) and (1,kernel) respectivly
      - number_of_convs (optional, default: 1): number of consecutive convolutional building blocks, i.e. Conv2DBlock.
      - activation (optional, default: "leaky_relu"): string literal to obtain activation function
      - dropout_rate (optional, default: 0): probability of the dropout layer. If the preceeding layer has more than one channel, spatial dropout is applied, otherwise standard dropout
      - final_activation (optional, default: True): whether or not to activate the output of this layer
      - use_spec_norm (optional, default: False): applies spectral normalization to convolutional and dense layers
      - strides (optional, default: (1,1)): stride of the filter
      - padding (optional, default: "zero"): padding type. Options are "none", "zero" or "reflection"
      - apply_final_normalization (optional, default: True): Whether or not to place a normalization on the layer's output
      - use_bias (optional, default: True): determines whether convolutions and dense layers include a bias or not
      - kernel_initializer (optional, default: HeAlphaUniform()): Initialization of the convolutions kernels.
      - gamma_initializer (optional, default: HeAlphaUniform()): Initialization of the normalization layers.
    """

    def __init__(
        self,
        filters: int,
        kernels: int,
        use_residual_Conv2DBlock: bool = False,
        split_kernels: bool = False,
        number_of_convs: int = 1,
        activation: str = "leaky_relu",
        dropout_rate: float = 0.0,
        final_activation: bool = True,
        use_spec_norm: bool = False,
        strides: Tuple[int, int] = (1, 1),
        padding: PaddingType = PaddingType.ZERO,
        apply_final_normalization: bool = True,
        use_bias: bool = True,
        kernel_initializer: tf.keras.initializers.Initializer = HeAlphaUniform(),
        gamma_initializer: tf.keras.initializers.Initializer = HeAlphaUniform(),
    ):
        super(Conv2DBlock, self).__init__()
        self.filters = filters
        self.use_residual_Conv2DBlock = use_residual_Conv2DBlock
        self.kernels = kernels
        self.split_kernels = split_kernels
        self.number_of_convs = number_of_convs
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.final_activation = final_activation
        self.use_spec_norm = use_spec_norm
        self.strides = strides
        self.padding = padding
        self.apply_final_normalization = apply_final_normalization
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.gamma_initializer = gamma_initializer

        self.pad = int((kernels - 1) / 2)  # assumes odd kernel size, which is typical!

        if split_kernels:
            self.convs = [
                Conv2DSplitted(
                    filters=filters, kernels=kernels, use_bias=use_bias, strides=strides, use_spec_norm=use_spec_norm
                )
                for _ in range(number_of_convs)
            ]
        else:
            if use_spec_norm:
                self.convs = [
                    tfa.layers.SpectralNormalization(
                        tf.keras.layers.Conv2D(
                            filters=filters,
                            kernel_size=(kernels, kernels),
                            kernel_initializer=kernel_initializer,
                            use_bias=use_bias,
                            strides=strides,
                        )
                    )
                    for _ in range(number_of_convs)
                ]
            else:
                self.convs = [
                    tf.keras.layers.Conv2D(
                        filters=filters,
                        kernel_size=(kernels, kernels),
                        kernel_initializer=kernel_initializer,
                        use_bias=use_bias,
                        strides=strides,
                    )
                    for _ in range(number_of_convs)
                ]

        if apply_final_normalization:
            num_instancenorm_blocks = number_of_convs
        else:
            num_instancenorm_blocks = number_of_convs - 1
        self.IN_blocks = [
            tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)
            for _ in range(num_instancenorm_blocks)
        ]
        self.dropout = dropout_func(filters, dropout_rate)

    def build(self, input_shape: tf.TensorShape) -> None:
        super(Conv2DBlock, self).build(input_shape)
        # print("Model built with shape: {}".format(input_shape))
        self.residualConv = None
        if input_shape[-1] != self.filters and self.use_residual_Conv2DBlock:
            # split kernels for kernel_size = 1 not required
            self.residualConv = tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=1,
                kernel_initializer=self.kernel_initializer,
                use_bias=self.use_bias,
                strides=self.strides,
            )
            if self.use_spec_norm:
                self.residualConv = tfa.layers.SpectralNormalization(self.residualConv)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if not self.built:
            raise ValueError("This model has not yet been built.")
        x = inputs

        for block in range(self.number_of_convs):
            residual = x

            if self.pad != 0 and self.padding != "none":
                x = pad_func(pad_values=(self.pad, self.pad), padding_type=self.padding)(x)
            x = self.convs[block](x)

            if self.use_residual_Conv2DBlock:
                if (
                    block == 0 and self.residualConv is not None
                ):  # after the first conf, the channel depth matches between input and output
                    residual = self.residualConv(residual)
                x = tf.keras.layers.Add()([x, residual])

            if block != (self.number_of_convs - 1) or self.apply_final_normalization:
                x = self.IN_blocks[block](x)

            if block != (self.number_of_convs - 1) or self.final_activation:
                x = tf.keras.layers.Activation(self.activation)(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        config = super(Conv2DBlock, self).get_config()
        config.update(
            {
                "filters": self.filters,
                "use_residual_Conv2DBlock": self.use_residual_Conv2DBlock,
                "kernels": self.kernels,
                "split_kernels": self.split_kernels,
                "number_of_convs": self.number_of_convs,
                "activation": self.activation,
                "dropout_rate": self.dropout_rate,
                "final_activation": self.final_activation,
                "use_spec_norm": self.use_spec_norm,
                "strides": self.strides,
                "padding": self.padding,
                "apply_final_normalization": self.apply_final_normalization,
                "use_bias": self.use_bias,
                "pad": self.pad,
                "kernel_initializer": self.kernel_initializer,
                "gamma_initializer": self.gamma_initializer,
            }
        )
        return config


# testcode
# layer = Conv2DBlock(filters = 128, use_residual_Conv2DBlock = True, kernels = 3, split_kernels  = True, use_spec_norm = True, number_of_convs = 3, activation = "relu", dropout_rate =0.1, final_activation = False, apply_final_normalization = True)
# print(layer.get_config())
# DeepSaki.layers.plot_layer(layer,input_shape=(256,256,64))


class DenseBlock(tf.keras.layers.Layer):
    """
    Wraps a dense layer into a more complex building block
    args:
      - units: number of units of each dense block
      - numberOfLayers (optional, default: 1): number of consecutive convolutional building blocks, i.e. Conv2DBlock.
      - activation (optional, default: "leaky_relu"): string literal to obtain activation function
      - dropout_rate (optional, default: 0): probability of the dropout layer. If the preceeding layer has more than one channel, spatial dropout is applied, otherwise standard dropout
      - final_activation (optional, default: True): whether or not to activate the output of this layer
      - use_spec_norm (optional, default: False): applies spectral normalization to convolutional and dense layers
      - apply_final_normalization (optional, default: True): Whether or not to place a normalization on the layer's output
      - use_bias (optional, default: True): determines whether convolutions and dense layers include a bias or not
      - kernel_initializer (optional, default: HeAlphaUniform()): Initialization of the convolutions kernels.
      - gamma_initializer (optional, default: HeAlphaUniform()): Initialization of the normalization layers.
    """

    def __init__(
        self,
        units: int,
        numberOfLayers: int = 1,
        activation: str = "leaky_relu",
        dropout_rate: float = 0.0,
        final_activation: bool = True,
        use_spec_norm: bool = False,
        apply_final_normalization: bool = True,
        use_bias: bool = True,
        kernel_initializer: tf.keras.initializers.Initializer = HeAlphaUniform(),
        gamma_initializer: tf.keras.initializers.Initializer = HeAlphaUniform(),
    ) -> None:
        super(DenseBlock, self).__init__()
        self.units = units
        self.numberOfLayers = numberOfLayers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.final_activation = final_activation
        self.use_spec_norm = use_spec_norm
        self.apply_final_normalization = apply_final_normalization
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.gamma_initializer = gamma_initializer

        if use_spec_norm:
            self.DenseBlocks = [
                tfa.layers.SpectralNormalization(
                    tf.keras.layers.Dense(units=units, use_bias=use_bias, kernel_initializer=kernel_initializer)
                )
                for _ in range(numberOfLayers)
            ]
        else:
            self.DenseBlocks = [
                tf.keras.layers.Dense(units=units, use_bias=use_bias, kernel_initializer=kernel_initializer)
                for _ in range(numberOfLayers)
            ]

        if apply_final_normalization:
            num_instancenorm_blocks = numberOfLayers
        else:
            num_instancenorm_blocks = numberOfLayers - 1
        self.IN_blocks = [
            tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)
            for _ in range(num_instancenorm_blocks)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs

        for block in range(self.numberOfLayers):
            x = self.DenseBlocks[block](x)

            if block != (self.numberOfLayers - 1) or self.apply_final_normalization:
                x = self.IN_blocks[block](x)

            if block != (self.numberOfLayers - 1) or self.final_activation:
                x = tf.keras.layers.Activation(self.activation)(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)
        return x

    def get_config(self) -> Dict[str, Any]:
        config = super(DenseBlock, self).get_config()
        config.update(
            {
                "units": self.units,
                "numberOfLayers": self.numberOfLayers,
                "dropout_rate": self.dropout_rate,
                "activation": self.activation,
                "final_activation": self.final_activation,
                "use_spec_norm": self.use_spec_norm,
                "apply_final_normalization": self.apply_final_normalization,
                "use_bias": self.use_bias,
                "kernel_initializer": self.kernel_initializer,
                "gamma_initializer": self.gamma_initializer,
            }
        )
        return config


# testcode
# layer = DenseBlock(units = 512, numberOfLayers = 3, activation = "leaky_relu", apply_final_normalization=False)
# print(layer.get_config())
# DeepSaki.layers.plot_layer(layer,input_shape=(256,256,64))


class DownSampleBlock(tf.keras.layers.Layer):
    """
    Spatial down-sampling for grid-like data
    args:
      - downsampling (optional, default: "average_pooling"):
      - activation (optional, default: "leaky_relu"): string literal to obtain activation function
      - kernels (optional, default: 3): size of the convolution's kernels when using downsampling = "conv_stride_2"
      - use_spec_norm (optional, default: False): applies spectral normalization to convolutional and dense layers
      - padding (optional, default: "zero"): padding type. Options are "none", "zero" or "reflection"
      - use_bias (optional, default: True): determines whether convolutions and dense layers include a bias or not
      - kernel_initializer (optional, default: HeAlphaUniform()): Initialization of the convolutions kernels.
      - gamma_initializer (optional, default: HeAlphaUniform()): Initialization of the normalization layers.
    """

    def __init__(
        self,
        downsampling: str = "average_pooling",
        activation: str = "leaky_relu",
        kernels: int = 3,
        use_spec_norm: bool = False,
        padding: PaddingType = PaddingType.ZERO,
        use_bias: bool = True,
        kernel_initializer: tf.keras.initializers.Initializer = HeAlphaUniform(),
        gamma_initializer: tf.keras.initializers.Initializer = HeAlphaUniform(),
    ) -> None:
        super(DownSampleBlock, self).__init__()
        self.kernels = kernels
        self.downsampling = downsampling
        self.activation = activation
        self.use_spec_norm = use_spec_norm
        self.padding = padding
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.gamma_initializer = gamma_initializer

    def build(self, input_shape: tf.TensorShape) -> None:
        super(DownSampleBlock, self).build(input_shape)

        self.layers = []
        if self.downsampling == "conv_stride_2":
            self.layers.append(
                Conv2DBlock(
                    input_shape[-1],
                    self.kernels,
                    activation=self.activation,
                    strides=(2, 2),
                    use_spec_norm=self.use_spec_norm,
                    padding=self.padding,
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    gamma_initializer=self.gamma_initializer,
                )
            )
        elif self.downsampling == "max_pooling":
            # Only spatial downsampling, increase in features is done by the conv2D_block specified later!
            self.layers.append(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        elif self.downsampling == "average_pooling":
            self.layers.append(tf.keras.layers.AveragePooling2D(pool_size=(2, 2)))
        elif self.downsampling == "space_to_depth":
            pass
        else:
            raise Exception("Undefined downsampling provided")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        if self.downsampling == "space_to_depth":
            x = tf.nn.space_to_depth(x, block_size=2)
        else:
            for layer in self.layers:
                x = layer(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        config = super(DownSampleBlock, self).get_config()
        config.update(
            {
                "kernels": self.kernels,
                "downsampling": self.downsampling,
                "activation": self.activation,
                "use_spec_norm": self.use_spec_norm,
                "padding": self.padding,
                "use_bias": self.use_bias,
                "kernel_initializer": self.kernel_initializer,
                "gamma_initializer": self.gamma_initializer,
            }
        )
        return config


# layer = DownSampleBlock(numOfChannels = 3, kernels = 3, downsampling = "conv_stride_2", activation = "leaky_relu", use_spec_norm = True)
# print(layer.get_config())
# DeepSaki.layers.plot_layer(layer,input_shape=(256,256,64))


class UpSampleBlock(tf.keras.layers.Layer):
    """
    Spatial up-sampling for grid-like data
    args:
      - upsampling (optional, default: "2D_upsample_and_conv"):
      - activation (optional, default: "leaky_relu"): string literal to obtain activation function
      - kernels (optional, default: 3): size of the convolution's kernels when using upsampling = "2D_upsample_and_conv" or "transpose_conv"
      - split_kernels (optional, default: False): to decrease the number of parameters, a convolution with the kernel_size (kernel,kernel) can be splitted into two consecutive convolutions with the kernel_size (kernel,1) and (1,kernel) respectivly. applies to upsampling = "2D_upsample_and_conv"
      - use_spec_norm (optional, default: False): applies spectral normalization to convolutional and dense layers
      - padding (optional, default: "zero"): padding type. Options are "none", "zero" or "reflection"
      - use_bias (optional, default: True): determines whether convolutions and dense layers include a bias or not
      - kernel_initializer (optional, default: HeAlphaUniform()): Initialization of the convolutions kernels.
      - gamma_initializer (optional, default: HeAlphaUniform()): Initialization of the normalization layers.
    """

    def __init__(
        self,
        upsampling: str = "2D_upsample_and_conv",
        activation: str = "leaky_relu",
        kernels: int = 3,
        split_kernels: bool = False,
        use_spec_norm: bool = False,
        use_bias: bool = True,
        padding: PaddingType = PaddingType.ZERO,
        kernel_initializer: tf.keras.initializers.Initializer = HeAlphaUniform(),
        gamma_initializer: tf.keras.initializers.Initializer = HeAlphaUniform(),
    ) -> None:
        super(UpSampleBlock, self).__init__()
        self.kernels = kernels
        self.split_kernels = split_kernels
        self.activation = activation
        self.use_spec_norm = use_spec_norm
        self.upsampling = upsampling
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.gamma_initializer = gamma_initializer
        self.padding = padding

    def build(self, input_shape: tf.TensorShape) -> None:
        super(UpSampleBlock, self).build(input_shape)
        self.layers = []

        if self.upsampling == "2D_upsample_and_conv":
            self.layers.append(tf.keras.layers.UpSampling2D(interpolation="bilinear"))
            self.layers.append(
                Conv2DBlock(
                    filters=input_shape[-1],
                    use_residual_Conv2DBlock=False,
                    kernels=1,
                    split_kernels=self.split_kernels,
                    number_of_convs=1,
                    activation=self.activation,
                    use_spec_norm=self.use_spec_norm,
                    use_bias=self.use_bias,
                    padding=self.padding,
                    kernel_initializer=self.kernel_initializer,
                    gamma_initializer=self.gamma_initializer,
                )
            )
        elif self.upsampling == "transpose_conv":
            self.layers.append(
                tf.keras.layers.Conv2DTranspose(
                    input_shape[-1],
                    kernel_size=(self.kernels, self.kernels),
                    strides=(2, 2),
                    kernel_initializer=self.kernel_initializer,
                    padding="same",
                    use_bias=self.use_bias,
                )
            )
            self.layers.append(tfa.layers.InstanceNormalization(gamma_initializer=self.gamma_initializer))
            self.layers.append(tf.keras.layers.Activation(self.activation))
        elif self.upsampling == "depth_to_space":
            self.layers.append(
                Conv2DBlock(
                    filters=4 * input_shape[-1],
                    use_residual_Conv2DBlock=False,
                    kernels=1,
                    split_kernels=False,
                    number_of_convs=1,
                    activation=self.activation,
                    use_spec_norm=self.use_spec_norm,
                    use_bias=self.use_bias,
                    padding=self.padding,
                    kernel_initializer=self.kernel_initializer,
                    gamma_initializer=self.gamma_initializer,
                )
            )
        else:
            raise Exception("Undefined upsampling provided")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        for layer in self.layers:
            x = layer(x)
        if self.upsampling == "depth_to_space":
            x = tf.nn.depth_to_space(x, block_size=2)
        return x

    def get_config(self) -> Dict[str, Any]:
        config = super(UpSampleBlock, self).get_config()
        config.update(
            {
                "kernels": self.kernels,
                "split_kernels": self.split_kernels,
                "activation": self.activation,
                "use_spec_norm": self.use_spec_norm,
                "upsampling": self.upsampling,
                "use_bias": self.use_bias,
                "padding": self.padding,
                "kernel_initializer": self.kernel_initializer,
                "gamma_initializer": self.gamma_initializer,
            }
        )
        return config


# Testcode
# layer = UpSampleBlock(kernels = 3, upsampling = "transpose_conv", activation = "leaky_relu", use_spec_norm = True, split_kernels = False)
# print(layer.get_config())
# DeepSaki.layers.plot_layer(layer,input_shape=(256,256,64))


class ResidualIdentityBlock(tf.keras.layers.Layer):
    """
    Residual identity block with configurable cardinality
    args:
      - filters: number of filters in the output feature map
      - kernels: size of the convolutions kernels
      - numberOfBlocks (optional, default: 1): number of consecutive convolutional building blocks.
      - activation (optional, default: "leaky_relu"): string literal to obtain activation function
      - dropout_rate (optional, default: 0): probability of the dropout layer. If the preceeding layer has more than one channel, spatial dropout is applied, otherwise standard dropout
      - use_spec_norm (optional, default: False): applies spectral normalization to convolutional and dense layers
      - residual_cardinality (optional, default: 1): number of parallel convolution blocks
      - padding (optional, default: "zero"): padding type. Options are "none", "zero" or "reflection"
      - use_bias (optional, default: True): determines whether convolutions and dense layers include a bias or not
      - kernel_initializer (optional, default: HeAlphaUniform()): Initialization of the convolutions kernels.
      - gamma_initializer (optional, default: HeAlphaUniform()): Initialization of the normalization layers.
    """

    def __init__(
        self,
        filters: int,
        kernels: int,
        activation: str = "leaky_relu",
        numberOfBlocks: int = 1,
        use_spec_norm: bool = False,
        residual_cardinality: int = 1,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        padding: PaddingType = PaddingType.ZERO,
        kernel_initializer: tf.keras.initializers.Initializer = HeAlphaUniform(),
        gamma_initializer: tf.keras.initializers.Initializer = HeAlphaUniform(),
    ) -> None:
        super(ResidualIdentityBlock, self).__init__()
        self.activation = activation
        self.filters = filters
        self.kernels = kernels
        self.numberOfBlocks = numberOfBlocks
        self.use_spec_norm = use_spec_norm
        self.residual_cardinality = residual_cardinality
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.gamma_initializer = gamma_initializer

        self.pad = int((kernels - 1) / 2)  # assumes odd kernel size, which is typical!

        if residual_cardinality > 1:
            self.intermediateFilters = int(max(filters / 32, filters / 16, filters / 8, filters / 4, filters / 2, 1))
        else:
            self.intermediateFilters = int(max(filters / 4, filters / 2, 1))

        # for each block, add several con
        self.blocks = []
        for i in range(numberOfBlocks):
            cardinals = []
            for _ in range(residual_cardinality):
                cardinals.append(
                    [
                        Conv2DBlock(
                            filters=self.intermediateFilters,
                            use_residual_Conv2DBlock=False,
                            kernels=1,
                            split_kernels=False,
                            number_of_convs=1,
                            activation=activation,
                            use_spec_norm=use_spec_norm,
                            use_bias=use_bias,
                            padding=padding,
                            kernel_initializer=kernel_initializer,
                            gamma_initializer=gamma_initializer,
                        ),
                        Conv2DBlock(
                            filters=self.intermediateFilters,
                            use_residual_Conv2DBlock=False,
                            kernels=kernels,
                            split_kernels=False,
                            number_of_convs=1,
                            activation=activation,
                            padding=PaddingType.NONE,
                            use_spec_norm=use_spec_norm,
                            use_bias=use_bias,
                            kernel_initializer=kernel_initializer,
                            gamma_initializer=gamma_initializer,
                        ),
                        Conv2DBlock(
                            filters=filters,
                            use_residual_Conv2DBlock=False,
                            kernels=1,
                            split_kernels=False,
                            number_of_convs=1,
                            activation=activation,
                            use_spec_norm=use_spec_norm,
                            use_bias=use_bias,
                            padding=padding,
                            kernel_initializer=kernel_initializer,
                            gamma_initializer=gamma_initializer,
                        ),
                    ]
                )
            self.blocks.append(cardinals)

        self.dropout = dropout_func(filters, dropout_rate)

    def build(self, input_shape: tf.TensorShape) -> None:
        super(ResidualIdentityBlock, self).build(input_shape)
        self.conv0 = None
        if input_shape[-1] != self.filters:
            self.conv0 = Conv2DBlock(
                filters=self.filters,
                use_residual_Conv2DBlock=False,
                kernels=1,
                split_kernels=False,
                number_of_convs=1,
                activation=self.activation,
                use_spec_norm=self.use_spec_norm,
                use_bias=self.use_bias,
                padding=self.padding,
                kernel_initializer=self.kernel_initializer,
                gamma_initializer=self.gamma_initializer,
            )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if not self.built:
            raise ValueError("This model has not yet been built.")
        x = inputs

        if self.conv0 is not None:
            x = self.conv0(x)

        for block in range(self.numberOfBlocks):
            residual = x

            if self.pad != 0 and self.padding != "none":
                x = pad_func(pad_values=(self.pad, self.pad), padding_type=self.padding)(x)

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

    def get_config(self) -> Dict[str, Any]:
        config = super(ResidualIdentityBlock, self).get_config()
        config.update(
            {
                "activation": self.activation,
                "filters": self.filters,
                "kernels": self.kernels,
                "numberOfBlocks": self.numberOfBlocks,
                "use_spec_norm": self.use_spec_norm,
                "residual_cardinality": self.residual_cardinality,
                "dropout_rate": self.dropout_rate,
                "use_bias": self.use_bias,
                "padding": self.padding,
                "kernel_initializer": self.kernel_initializer,
                "gamma_initializer": self.gamma_initializer,
            }
        )
        return config


# Testcode
# layer = ResidualIdentityBlock(filters =64, activation = "leaky_relu", kernels = 3, numberOfBlocks=2,use_spec_norm = False, residual_cardinality =5, dropout_rate=0.2)
# print(layer.get_config())
# DeepSaki.layers.plot_layer(layer,input_shape=(256,256,64))


class ResBlockDown(tf.keras.layers.Layer):
    """
    Spatial down-sampling with residual connection for grid-like data
    args:
      - activation (optional, default: "leaky_relu"): string literal to obtain activation function
      - use_spec_norm (optional, default: False): applies spectral normalization to convolutional and dense layers
      - use_bias (optional, default: True): determines whether convolutions and dense layers include a bias or not
      - padding (optional, default: "zero"): padding type. Options are "none", "zero" or "reflection"
      - kernel_initializer (optional, default: HeAlphaUniform()): Initialization of the convolutions kernels.
      - gamma_initializer (optional, default: HeAlphaUniform()): Initialization of the normalization layers.
    """

    def __init__(
        self,
        activation: str = "leaky_relu",
        use_spec_norm: bool = False,
        use_bias: bool = True,
        padding: PaddingType = PaddingType.ZERO,
        kernel_initializer: tf.keras.initializers.Initializer = HeAlphaUniform(),
        gamma_initializer: tf.keras.initializers.Initializer = HeAlphaUniform(),
    ) -> None:
        super(ResBlockDown, self).__init__()
        self.activation = activation
        self.use_spec_norm = use_spec_norm
        self.use_bias = use_bias
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.gamma_initializer = gamma_initializer

    def build(self, input_shape: tf.TensorShape) -> None:
        super(ResBlockDown, self).build(input_shape)

        self.convRes = Conv2DBlock(
            filters=input_shape[-1],
            use_residual_Conv2DBlock=False,
            kernels=1,
            split_kernels=False,
            number_of_convs=1,
            activation=self.activation,
            use_spec_norm=self.use_spec_norm,
            use_bias=self.use_bias,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            gamma_initializer=self.gamma_initializer,
        )
        self.conv1 = Conv2DBlock(
            filters=input_shape[-1],
            use_residual_Conv2DBlock=False,
            kernels=3,
            split_kernels=False,
            number_of_convs=1,
            activation=self.activation,
            use_spec_norm=self.use_spec_norm,
            use_bias=self.use_bias,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            gamma_initializer=self.gamma_initializer,
        )
        self.conv2 = Conv2DBlock(
            filters=input_shape[-1],
            use_residual_Conv2DBlock=False,
            kernels=3,
            split_kernels=False,
            number_of_convs=1,
            activation=self.activation,
            use_spec_norm=self.use_spec_norm,
            use_bias=self.use_bias,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            gamma_initializer=self.gamma_initializer,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        path1 = inputs
        path2 = inputs

        path1 = self.convRes(path1)
        path1 = tf.keras.layers.AveragePooling2D()(path1)

        path2 = self.conv1(path2)
        path2 = self.conv2(path2)
        path2 = tf.keras.layers.AveragePooling2D()(path2)

        return tf.keras.layers.Add()([path1, path2])

    def get_config(self) -> Dict[str, Any]:
        config = super(ResBlockDown, self).get_config()
        config.update(
            {
                "activation": self.activation,
                "use_spec_norm": self.use_spec_norm,
                "use_bias": self.use_bias,
                "padding": self.padding,
                "kernel_initializer": self.kernel_initializer,
                "gamma_initializer": self.gamma_initializer,
            }
        )
        return config


# Testcode
# layer = ResBlockDown( activation = "leaky_relu", use_spec_norm = True)
# print(layer.get_config())
# DeepSaki.layers.plot_layer(layer,input_shape=(256,256,64))


class ResBlockUp(tf.keras.layers.Layer):
    """
    Spatial down-sampling with residual connection for grid-like data
    args:
      - activation (optional, default: "leaky_relu"): string literal to obtain activation function
      - use_spec_norm (optional, default: False): applies spectral normalization to convolutional and dense layers
      - padding (optional, default: "zero"): padding type. Options are "none", "zero" or "reflection"
      - use_bias (optional, default: True): determines whether convolutions and dense layers include a bias or not
      - kernel_initializer (optional, default: HeAlphaUniform()): Initialization of the convolutions kernels.
      - gamma_initializer (optional, default: HeAlphaUniform()): Initialization of the normalization layers.
    """

    def __init__(
        self,
        activation: str = "leaky_relu",
        use_spec_norm: bool = False,
        use_bias: bool = True,
        padding: PaddingType = PaddingType.ZERO,
        kernel_initializer: tf.keras.initializers.Initializer = HeAlphaUniform(),
        gamma_initializer: tf.keras.initializers.Initializer = HeAlphaUniform(),
    ) -> None:
        super(ResBlockUp, self).__init__()
        self.activation = activation
        self.use_spec_norm = use_spec_norm
        self.use_bias = use_bias
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.gamma_initializer = gamma_initializer

    def build(self, input_shape: tf.TensorShape) -> None:
        super(ResBlockUp, self).build(input_shape)
        self.convRes = Conv2DBlock(
            filters=input_shape[-1],
            use_residual_Conv2DBlock=False,
            kernels=1,
            split_kernels=False,
            number_of_convs=1,
            activation=self.activation,
            use_spec_norm=self.use_spec_norm,
            use_bias=self.use_bias,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            gamma_initializer=self.gamma_initializer,
        )
        self.conv1 = Conv2DBlock(
            filters=input_shape[-1],
            use_residual_Conv2DBlock=False,
            kernels=3,
            split_kernels=False,
            number_of_convs=1,
            activation=self.activation,
            use_spec_norm=self.use_spec_norm,
            use_bias=self.use_bias,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            gamma_initializer=self.gamma_initializer,
        )
        self.conv2 = Conv2DBlock(
            filters=input_shape[-1],
            use_residual_Conv2DBlock=False,
            kernels=3,
            split_kernels=False,
            number_of_convs=1,
            activation=self.activation,
            use_spec_norm=self.use_spec_norm,
            use_bias=self.use_bias,
            padding=self.padding,
            kernel_initializer=self.kernel_initializer,
            gamma_initializer=self.gamma_initializer,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        path1 = inputs
        path2 = inputs

        path1 = tf.keras.layers.UpSampling2D()(path1)
        path1 = self.convRes(path1)

        path2 = tf.keras.layers.UpSampling2D()(path2)
        path2 = self.conv1(path2)
        path2 = self.conv2(path2)

        return tf.keras.layers.Add()([path1, path2])

    def get_config(self) -> Dict[str, Any]:
        config = super(ResBlockUp, self).get_config()
        config.update(
            {
                "activation": self.activation,
                "use_spec_norm": self.use_spec_norm,
                "use_bias": self.use_bias,
                "padding": self.padding,
                "kernel_initializer": self.kernel_initializer,
                "gamma_initializer": self.gamma_initializer,
            }
        )
        return config


# Testcode
# layer = ResBlockUp(activation = "leaky_relu", use_spec_norm = True)
# print(layer.get_config())
# DeepSaki.layers.plot_layer(layer,input_shape=(256,256,64))


@tf.keras.utils.register_keras_serializable(package="Custom", name="scale")
class ScaleLayer(tf.keras.layers.Layer):
    """
    trainable scalar that can act as trainable gate
    args:
      - initializer (optional, default: tf.keras.initializers.Ones()): initializes the scalar weight
    """

    def __init__(self, initializer: tf.keras.initializers.Initializer = tf.keras.initializers.Ones()) -> None:
        super(ScaleLayer, self).__init__()
        self.initializer = initializer
        self.scale = self.add_weight(
            shape=[1], initializer=initializer, constraint=DeepSaki.constraints.NonNegative(), trainable=True
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs * self.scale

    def get_config(self) -> Dict[str, Any]:
        config = super(ScaleLayer, self).get_config()
        config.update({"scale": self.scale, "initializer": self.initializer})
        return config


class ScalarGatedSelfAttention(tf.keras.layers.Layer):
    """
    Scaled dot-product self attention that is gated by a learnable scalar.
    args:
    - use_spec_norm (optional, default: False): applies spectral normalization to convolutional and dense layers
    - intermediateChannel (optional, default: None): Integer that determines the intermediate channels within the self-attention model. If None, intermediate channels = inputChannels/8
    - kernel_initializer (optional, default: HeAlphaUniform()): Initialization of the convolutions kernels.
    - gamma_initializer (optional, default: HeAlphaUniform()): Initialization of the normalization layers.
    """

    def __init__(
        self,
        use_spec_norm: bool = False,
        intermediate_channel: Optional[bool] = None,
        kernel_initializer: tf.keras.initializers.Initializer = HeAlphaUniform(),
        gamma_initializer: tf.keras.initializers.Initializer = HeAlphaUniform(),
    ) -> None:
        super(ScalarGatedSelfAttention, self).__init__()
        self.use_spec_norm = use_spec_norm
        self.intermediateChannel = intermediate_channel
        self.kernel_initializer = kernel_initializer
        self.gamma_initializer = gamma_initializer

    def build(self, input_shape: tf.TensorShape) -> None:
        super(ScalarGatedSelfAttention, self).build(input_shape)
        batchSize, height, width, numChannel = input_shape
        if self.intermediateChannel is None:
            self.intermediateChannel = numChannel // 8

        self.w_f = DenseBlock(
            units=self.intermediateChannel,
            use_spec_norm=self.use_spec_norm,
            numberOfLayers=1,
            activation=None,
            apply_final_normalization=False,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            gamma_initializer=self.gamma_initializer,
        )
        self.w_g = DenseBlock(
            units=self.intermediateChannel,
            use_spec_norm=self.use_spec_norm,
            numberOfLayers=1,
            activation=None,
            apply_final_normalization=False,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            gamma_initializer=self.gamma_initializer,
        )
        self.w_h = DenseBlock(
            units=self.intermediateChannel,
            use_spec_norm=self.use_spec_norm,
            numberOfLayers=1,
            activation=None,
            apply_final_normalization=False,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            gamma_initializer=self.gamma_initializer,
        )
        self.w_fgh = DenseBlock(
            units=numChannel,
            use_spec_norm=self.use_spec_norm,
            numberOfLayers=1,
            activation=None,
            apply_final_normalization=False,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            gamma_initializer=self.gamma_initializer,
        )

        self.LN_f = tf.keras.layers.LayerNormalization(gamma_initializer=self.gamma_initializer)
        self.LN_g = tf.keras.layers.LayerNormalization(gamma_initializer=self.gamma_initializer)
        self.LN_h = tf.keras.layers.LayerNormalization(gamma_initializer=self.gamma_initializer)
        self.LN_fgh = tf.keras.layers.LayerNormalization(gamma_initializer=self.gamma_initializer)
        self.scale = ScaleLayer()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if not self.built:
            raise ValueError("This model has not yet been built.")

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

        return tf.keras.layers.Add()([f_g_h, inputs])

    def get_config(self) -> Dict[str, Any]:
        config = super(ScalarGatedSelfAttention, self).get_config()
        config.update(
            {
                "use_spec_norm": self.use_spec_norm,
                "intermediateChannel": self.intermediateChannel,
                "gamma_initializer": self.gamma_initializer,
                "kernel_initializer": self.kernel_initializer,
            }
        )
        return config


# Testcode
# layer =ScalarGatedSelfAttention(use_spec_norm = True,intermediateChannel = None)
# print(layer.get_config())
# DeepSaki.layers.plot_layer(layer,input_shape=(32,32,512))
