from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import tensorflow as tf
import tensorflow_addons as tfa

from DeepSaki.constraints import NonNegative
from DeepSaki.initializers.he_alpha import HeAlphaUniform
from DeepSaki.layers.layer_helper import PaddingType
from DeepSaki.layers.layer_helper import dropout_func
from DeepSaki.layers.layer_helper import pad_func

class Conv2DSplitted(tf.keras.layers.Layer):
    """Convolution layer where a single convolution is splitted into two consecutive convolutions.

    To decrease the number of parameters, a convolution with the kernel_size (kernel,kernel) can be splitted into two
    consecutive convolutions with the kernel_size (kernel,1) and (1,kernel) respectivly.
    """

    def __init__(
        self,
        filters: int = 3,
        kernels: int = 3,
        use_spec_norm: bool = False,
        strides: Tuple[int, int] = (1, 1),
        use_bias: bool = True,
        kernel_initializer: Optional[tf.keras.initializers.Initializer] = None,
    ) -> None:
        """Initialize the `Conv2DSplitted` object.

        Args:
            filters (int, optional): Number of filters in the output feature map. Defaults to 3.
            kernels (int, optional): Size of the convolutions' kernels, which will be translated to (kernels, 1) and
                (1,kernels) for the first and seccond convolution respectivly. Defaults to 3.
            use_spec_norm (bool, optional): Applies spectral normalization to convolutional and dense layers. Defaults
                to False.
            strides (Tuple[int, int], optional): Strides of the convolution layers. Defaults to (1, 1).
            use_bias (bool, optional): Whether or not to use bias weights. Defaults to True.
            kernel_initializer (tf.keras.initializers.Initializer, optional): Initialization of the convolutions
                kernels. Defaults to None.
        """
        super(Conv2DSplitted, self).__init__()
        self.filters = filters
        self.kernels = kernels
        self.use_spec_norm = use_spec_norm
        self.strides = strides
        self.use_bias = use_bias
        self.kernel_initializer = HeAlphaUniform() if kernel_initializer is None else kernel_initializer

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
        """Calls the `Conv2DSplitted` layer.

        Args:
            inputs (tf.Tensor): Tensor of shape `(batch, height, width, channel)`.

        Returns:
            Convoluted tensor of shape `(batch, H, W, filters)`, where `H` and `W` depend on the padding type used in
                the convolution layers.
        """
        x = self.conv1(inputs)
        return self.conv2(x)

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
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
    """Wraps a two-dimensional convolution into a more complex building block."""

    def __init__(
        self,
        filters: int = 3,
        kernels: int = 3,
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
        kernel_initializer: Optional[tf.keras.initializers.Initializer] = None,
        gamma_initializer: Optional[tf.keras.initializers.Initializer] = None,
    ) -> None:
        """Initializes the `Conv2DBlock` layer.

        Args:
            filters (int, optional): Number of individual filters. Defaults to 3.
            kernels (int, optional): Size of the convolutions kernels. Defaults to 3.
            use_residual_Conv2DBlock (bool, optional): Adds a residual connection in parallel to the `Conv2DBlock`.
                Defaults to False.
            split_kernels (bool, optional): To decrease the number of parameters, a convolution with the kernel_size
                `(kernel,kernel)` can be splitted into two consecutive convolutions with the kernel_size `(kernel,1)`
                and `(1,kernel)` respectivly. Defaults to False.
            number_of_convs (int, optional): Number of consecutive convolutional building blocks, i.e. `Conv2DBlock`.
                Defaults to 1.
            activation (str, optional): String literal or tensorflow activation function object to obtain activation
                function. Defaults to "leaky_relu".
            dropout_rate (float, optional): Probability of the dropout layer. If the preceeding layer has more than one
                channel, spatial dropout is applied, otherwise standard dropout. Defaults to 0.0.
            final_activation (bool, optional): Whether or not to activate the output of this layer. Defaults to True.
            use_spec_norm (bool, optional): Applies spectral normalization to convolutional and dense layers.
                Defaults to False.
            strides (Tuple[int, int], optional): Stride of the filter. Defaults to (1, 1).
            padding (PaddingType, optional): Padding type. Defaults to PaddingType.ZERO.
            apply_final_normalization (bool, optional): Whether or not to place a normalization on the layer's output.
                Defaults to True.
            use_bias (bool, optional): Whether convolutions and dense layers include a bias or not. Defaults to True.
            kernel_initializer (tf.keras.initializers.Initializer, optional): Initialization of the convolutions kernels.
                Defaults to None.
            gamma_initializer (tf.keras.initializers.Initializer, optional): Initialization of the normalization layers.
                Defaults to None.
        """
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
        self.kernel_initializer = HeAlphaUniform() if kernel_initializer is None else kernel_initializer
        self.gamma_initializer = HeAlphaUniform() if gamma_initializer is None else gamma_initializer

        self.pad = int((kernels - 1) / 2)  # assumes odd kernel size, which is typical!

        if split_kernels:
            self.convs = [
                Conv2DSplitted(
                    filters=filters, kernels=kernels, use_bias=use_bias, strides=strides, use_spec_norm=use_spec_norm
                )
                for _ in range(number_of_convs)
            ]
        elif use_spec_norm:
            self.convs = [
                tfa.layers.SpectralNormalization(
                    tf.keras.layers.Conv2D(
                        filters=filters,
                        kernel_size=(kernels, kernels),
                        kernel_initializer=self.kernel_initializer,
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
                    kernel_initializer=self.kernel_initializer,
                    use_bias=use_bias,
                    strides=strides,
                )
                for _ in range(number_of_convs)
            ]

        num_instancenorm_blocks = number_of_convs if apply_final_normalization else number_of_convs - 1

        self.IN_blocks = [
            tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)
            for _ in range(num_instancenorm_blocks)
        ]
        self.dropout = dropout_func(filters, dropout_rate)

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer depending on the `input_shape` (output shape of the previous layer).

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor to this layer.
        """
        super(Conv2DBlock, self).build(input_shape)
        self.residual_conv = None
        if input_shape[-1] != self.filters and self.use_residual_Conv2DBlock:
            # split kernels for kernel_size = 1 not required
            self.residual_conv = tf.keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=1,
                kernel_initializer=self.kernel_initializer,
                use_bias=self.use_bias,
                strides=self.strides,
            )
            if self.use_spec_norm:
                self.residual_conv = tfa.layers.SpectralNormalization(self.residual_conv)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the `Conv2DBlock` layer.

        Args:
            inputs (tf.Tensor): Tensor of shape (batch, height, width, channel)

        Raises:
            ValueError: If layer has not been built by calling build() on to layer.

        Returns:
            Tensor of shape (batch, `H`, `W`, `filters`). The values for `H`, `W`, `C` depend on the stride as well on
            the padding. If padding is applied in the stride is (1,1), the output shape matches the input shape. If for
            example the stride is(2,2) the output shape would be (batch, `height/(2*number_of_convs)`,
            `width/(2*number_of_convs)`, `filters`).
        """
        if not self.built:
            raise ValueError("This model has not yet been built.")
        x = inputs

        for block in range(self.number_of_convs):
            residual = x

            if self.pad != 0 and self.padding != PaddingType.NONE:
                x = pad_func(pad_values=(self.pad, self.pad), padding_type=self.padding)(x)
            x = self.convs[block](x)

            if self.use_residual_Conv2DBlock:
                if (
                    block == 0 and self.residual_conv is not None
                ):  # after the first conf, the channel depth matches between input and output
                    residual = self.residual_conv(residual)
                x = tf.keras.layers.Add()([x, residual])

            if block != (self.number_of_convs - 1) or self.apply_final_normalization:
                x = self.IN_blocks[block](x)

            if block != (self.number_of_convs - 1) or self.final_activation:
                x = tf.keras.layers.Activation(self.activation)(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
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
                "kernel_initializer": self.kernel_initializer,
                "gamma_initializer": self.gamma_initializer,
            }
        )
        return config


class DenseBlock(tf.keras.layers.Layer):
    """Wraps a dense layer into a more complex building block."""

    def __init__(
        self,
        units: int,
        number_of_layers: int = 1,
        activation: str = "leaky_relu",
        dropout_rate: float = 0.0,
        final_activation: bool = True,
        use_spec_norm: bool = False,
        apply_final_normalization: bool = True,
        use_bias: bool = True,
        kernel_initializer: Optional[tf.keras.initializers.Initializer] = None,
        gamma_initializer: Optional[tf.keras.initializers.Initializer] = None,
    ) -> None:
        """Initializes the `DenseBlock` layer.

        Args:
            units (int): Number of units of each dense block
            number_of_layers (int, optional): Number of consecutive subblocks. Defaults to 1.
            activation (str, optional): String literal to obtain activation function. Defaults to "leaky_relu".
            dropout_rate (float, optional): Probability of the dropout layer dropping weights. If the preceeding layer
                has more than one channel, spatial dropout is applied, otherwise standard dropout. Defaults to 0.0.
            final_activation (bool, optional): Whether or not to activate the output of this layer. Defaults to True.
            use_spec_norm (bool, optional): Applies spectral normalization to dense layers. Defaults to False.
            apply_final_normalization (bool, optional): Whether or not to apply normalization at the layer's output.
                Defaults to True.
            use_bias (bool, optional): Whether dense layers include bias weights. Defaults to True.
            kernel_initializer (tf.keras.initializers.Initializer, optional): Initialization of the convolutions kernels.
                Defaults to None.
            gamma_initializer (tf.keras.initializers.Initializer, optional): Initialization of the normalization layers.
                Defaults to None.
        """
        super(DenseBlock, self).__init__()
        self.units = units
        self.number_of_layers = number_of_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.final_activation = final_activation
        self.use_spec_norm = use_spec_norm
        self.apply_final_normalization = apply_final_normalization
        self.use_bias = use_bias
        self.kernel_initializer = HeAlphaUniform() if kernel_initializer is None else kernel_initializer
        self.gamma_initializer = HeAlphaUniform() if gamma_initializer is None else gamma_initializer

        if use_spec_norm:
            self.DenseBlocks = [
                tfa.layers.SpectralNormalization(
                    tf.keras.layers.Dense(units=units, use_bias=use_bias, kernel_initializer=self.kernel_initializer)
                )
                for _ in range(number_of_layers)
            ]
        else:
            self.DenseBlocks = [
                tf.keras.layers.Dense(units=units, use_bias=use_bias, kernel_initializer=self.kernel_initializer)
                for _ in range(number_of_layers)
            ]

        num_instancenorm_blocks = number_of_layers if apply_final_normalization else number_of_layers - 1
        self.IN_blocks = [
            tfa.layers.InstanceNormalization(gamma_initializer=self.gamma_initializer)
            for _ in range(num_instancenorm_blocks)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs

        for block in range(self.number_of_layers):
            x = self.DenseBlocks[block](x)

            if block != (self.number_of_layers - 1) or self.apply_final_normalization:
                x = self.IN_blocks[block](x)

            if block != (self.number_of_layers - 1) or self.final_activation:
                x = tf.keras.layers.Activation(self.activation)(x)

        if self.dropout_rate > 0:
            x = self.dropout(x)
        return x

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(DenseBlock, self).get_config()
        config.update(
            {
                "units": self.units,
                "number_of_layers": self.number_of_layers,
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


class DownSampleBlock(tf.keras.layers.Layer):
    """Spatial down-sampling for grid-like data using `DeepSaki.layers.ConvBlock2D()`."""

    def __init__(
        self,
        downsampling: str = "average_pooling",
        activation: str = "leaky_relu",
        kernels: int = 3,
        use_spec_norm: bool = False,
        padding: PaddingType = PaddingType.ZERO,
        use_bias: bool = True,
        kernel_initializer: Optional[tf.keras.initializers.Initializer] = None,
        gamma_initializer: Optional[tf.keras.initializers.Initializer] = None,
    ) -> None:
        """Initializes the `DownSampleBlock` layer.

        Args:
            downsampling (str, optional): Downsampling method used. Defaults to "average_pooling".
            activation (str, optional): String literal or tensorflow activation function object to obtain activation
                function. Defaults to "leaky_relu".
            kernels (int, optional): Size of the convolutions kernels. Defaults to 3.
            use_spec_norm (bool, optional): Applies spectral normalization to convolutional and dense layers. Defaults
                to False.
            padding (PaddingType, optional): Padding type. Defaults to PaddingType.ZERO.
            use_bias (bool, optional): Whether convolutions and dense layers include a bias or not. Defaults to True.
            kernel_initializer (tf.keras.initializers.Initializer, optional): Initialization of the convolutions kernels.
                Defaults to None.
            gamma_initializer (tf.keras.initializers.Initializer, optional): Initialization of the normalization layers.
                Defaults to None.
        """
        super(DownSampleBlock, self).__init__()
        self.kernels = kernels
        self.downsampling = downsampling
        self.activation = activation
        self.use_spec_norm = use_spec_norm
        self.padding = padding
        self.use_bias = use_bias
        self.kernel_initializer = HeAlphaUniform() if kernel_initializer is None else kernel_initializer
        self.gamma_initializer = HeAlphaUniform() if gamma_initializer is None else gamma_initializer

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer depending on the `input_shape` (output shape of the previous layer).

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor to this layer.
        """
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
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
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


class UpSampleBlock(tf.keras.layers.Layer):
    """Spatial up-sampling for grid-like data using `DeepSaki.layers.ConvBlock2D()`."""

    def __init__(
        self,
        upsampling: str = "2D_upsample_and_conv",
        activation: str = "leaky_relu",
        kernels: int = 3,
        use_spec_norm: bool = False,
        use_bias: bool = True,
        padding: PaddingType = PaddingType.ZERO,
        kernel_initializer: Optional[tf.keras.initializers.Initializer] = None,
        gamma_initializer: Optional[tf.keras.initializers.Initializer] = None,
    ) -> None:
        """Initializes the `UpSampleBlock` layer.

        Args:
            upsampling (str, optional): _description_. Defaults to "2D_upsample_and_conv".
            activation (str, optional): String literal or tensorflow activation function object to obtain activation
                function. Defaults to "leaky_relu".
            kernels (int, optional): Size of the convolutions kernels. Defaults to 3.
            use_spec_norm (bool, optional): Applies spectral normalization to convolutional and dense layers. Defaults
                to False.
            use_bias (bool, optional): Whether convolutions and dense layers include a bias or not. Defaults to True.
            padding (PaddingType, optional): Padding type. Defaults to PaddingType.ZERO.
            kernel_initializer (tf.keras.initializers.Initializer, optional): Initialization of the convolutions kernels.
                Defaults to None.
            gamma_initializer (tf.keras.initializers.Initializer, optional): Initialization of the normalization layers.
                Defaults to None.
        """
        super(UpSampleBlock, self).__init__()
        self.kernels = kernels
        self.activation = activation
        self.use_spec_norm = use_spec_norm
        self.upsampling = upsampling
        self.use_bias = use_bias
        self.kernel_initializer = HeAlphaUniform() if kernel_initializer is None else kernel_initializer
        self.gamma_initializer = HeAlphaUniform() if gamma_initializer is None else gamma_initializer
        self.padding = padding

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer depending on the `input_shape` (output shape of the previous layer).

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor to this layer.
        """
        super(UpSampleBlock, self).build(input_shape)
        self.layers = []

        if self.upsampling == "2D_upsample_and_conv":
            self.layers.append(tf.keras.layers.UpSampling2D(interpolation="bilinear"))
            self.layers.append(
                Conv2DBlock(
                    filters=input_shape[-1],
                    use_residual_Conv2DBlock=False,
                    kernels=1,
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
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(UpSampleBlock, self).get_config()
        config.update(
            {
                "kernels": self.kernels,
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


class ResidualBlock(tf.keras.layers.Layer):
    """Residual block with configurable cardinality."""

    def __init__(
        self,
        filters: int = 3,
        kernels: int = 3,
        activation: str = "leaky_relu",
        number_of_blocks: int = 1,
        use_spec_norm: bool = False,
        residual_cardinality: int = 1,
        dropout_rate: float = 0.0,
        use_bias: bool = True,
        padding: PaddingType = PaddingType.ZERO,
        kernel_initializer: Optional[tf.keras.initializers.Initializer] = None,
        gamma_initializer: Optional[tf.keras.initializers.Initializer] = None,
    ) -> None:
        """Initializes the `ResidualBlock` layer.

        Args:
            filters (int, optional): Number of individual filters. Defaults to 3.
            kernels (int, optional): Size of the convolutions kernels. Defaults to 3.
            activation (str, optional): String literal or tensorflow activation function object to obtain activation
                function. Defaults to "leaky_relu".
            number_of_blocks (int, optional): Number of residual subblocks. Defaults to 1.
            use_spec_norm (bool, optional): Applies spectral normalization to convolutional and dense layers. Defaults
                to False.
            residual_cardinality (int, optional): Number of parallel paths. Defaults to 1.
            dropout_rate (float, optional): Probability of the dropout layer. If the preceeding layer has more than one
                channel, spatial dropout is applied, otherwise standard dropout. Defaults to 0.0.
            use_bias (bool, optional): Whether convolutions and dense layers include a bias or not. Defaults to True.
            padding (PaddingType, optional): Padding type. Defaults to PaddingType.ZERO.
            kernel_initializer (tf.keras.initializers.Initializer, optional): Initialization of the convolutions kernels.
                Defaults to None.
            gamma_initializer (tf.keras.initializers.Initializer, optional): Initialization of the normalization layers.
                Defaults to None.
        """
        super(ResidualBlock, self).__init__()
        self.activation = activation
        self.filters = filters
        self.kernels = kernels
        self.number_of_blocks = number_of_blocks
        self.use_spec_norm = use_spec_norm
        self.residual_cardinality = residual_cardinality
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.padding = padding
        self.kernel_initializer = HeAlphaUniform() if kernel_initializer is None else kernel_initializer
        self.gamma_initializer = HeAlphaUniform() if gamma_initializer is None else gamma_initializer

        self.pad = int((kernels - 1) / 2)  # assumes odd kernel size, which is typical!

        if residual_cardinality > 1:
            self.intermediateFilters = int(max(filters / 32, filters / 16, filters / 8, filters / 4, filters / 2, 1))
        else:
            self.intermediateFilters = int(max(filters / 4, filters / 2, 1))

        # for each block, add several con
        self.blocks = []
        for _ in range(number_of_blocks):
            cardinals = [
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
                        kernel_initializer=self.kernel_initializer,
                        gamma_initializer=self.gamma_initializer,
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
                        kernel_initializer=self.kernel_initializer,
                        gamma_initializer=self.gamma_initializer,
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
                        kernel_initializer=self.kernel_initializer,
                        gamma_initializer=self.gamma_initializer,
                    ),
                ]
                for _ in range(residual_cardinality)
            ]

            self.blocks.append(cardinals)

        self.dropout = dropout_func(filters, dropout_rate)

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer depending on the `input_shape` (output shape of the previous layer).

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor to this layer.
        """
        super(ResidualBlock, self).build(input_shape)
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

        for block in range(self.number_of_blocks):
            residual = x

            if self.pad != 0 and self.padding != PaddingType.NONE:
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
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(ResidualBlock, self).get_config()
        config.update(
            {
                "activation": self.activation,
                "filters": self.filters,
                "kernels": self.kernels,
                "number_of_blocks": self.number_of_blocks,
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


class ResBlockDown(tf.keras.layers.Layer):
    """Spatial down-sampling with residual connection for grid-like data using `DeepSaki.layers.ConvBlock2D()`.

    Architecture:
        ```mermaid
        flowchart LR
            i([input_tensor])--> c1 & c2
            subgraph ResBlockDown
                subgraph Block[2x]
                c1[dsk.layers.Conv2DBlock]
                end
                c1-->ap1[AveragePooling2D]
                c2[dsk.layers.Conv2DBlock]-->ap2[AveragePooling2D]
                ap1 & ap2-->add((+))
            end
            add-->o([output_tensor])
        ```

    """

    def __init__(
        self,
        activation: str = "leaky_relu",
        use_spec_norm: bool = False,
        use_bias: bool = True,
        padding: PaddingType = PaddingType.ZERO,
        kernel_initializer: Optional[tf.keras.initializers.Initializer] = None,
        gamma_initializer: Optional[tf.keras.initializers.Initializer] = None,
    ) -> None:
        """Initializes the `ResBlockDown` layer.

        Args:
            activation (str, optional): String literal or tensorflow activation function object to obtain activation
                function. Defaults to "leaky_relu".
            use_spec_norm (bool, optional): Applies spectral normalization to convolutional and dense layers. Defaults
                to False.
            use_bias (bool, optional): Whether convolutions and dense layers include a bias or not. Defaults to True.
            padding (PaddingType, optional): Padding type. Defaults to PaddingType.ZERO.
            kernel_initializer (tf.keras.initializers.Initializer, optional): Initialization of the convolutions kernels.
                Defaults to None.
            gamma_initializer (tf.keras.initializers.Initializer, optional): Initialization of the normalization layers.
                Defaults to None.
        """
        super(ResBlockDown, self).__init__()
        self.activation = activation
        self.use_spec_norm = use_spec_norm
        self.use_bias = use_bias
        self.padding = padding
        self.kernel_initializer = HeAlphaUniform() if kernel_initializer is None else kernel_initializer
        self.gamma_initializer = HeAlphaUniform() if gamma_initializer is None else gamma_initializer

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer depending on the `input_shape` (output shape of the previous layer).

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor to this layer.
        """
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
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
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


class ResBlockUp(tf.keras.layers.Layer):
    """Spatial up-sampling with residual connection for grid-like data using `DeepSaki.layers.ConvBlock2D()`.

    Architecture:
        ```mermaid
        flowchart LR
            i([input_tensor])-->up1 & up2
            subgraph ResBlockUp
                up1[UpSample2D]-->c1
                subgraph Block[2x]
                c1[dsk.layers.Conv2DBlock]
                end
                up2[UpSample2D]-->c2[dsk.layers.Conv2DBlock]
                c1 & c2-->add((+))
            end
            add-->o([output_tensor])
        ```

    """

    def __init__(
        self,
        activation: str = "leaky_relu",
        use_spec_norm: bool = False,
        use_bias: bool = True,
        padding: PaddingType = PaddingType.ZERO,
        kernel_initializer: Optional[tf.keras.initializers.Initializer] = None,
        gamma_initializer: Optional[tf.keras.initializers.Initializer] = None,
    ) -> None:
        """Initializes the `ResBlockUp` layer.

        Args:
            activation (str, optional): String literal or tensorflow activation function object to obtain activation
                function. Defaults to "leaky_relu".
            use_spec_norm (bool, optional): Applies spectral normalization to convolutional and dense layers. Defaults
                to False.
            use_bias (bool, optional): Whether convolutions and dense layers include a bias or not. Defaults to True.
            padding (PaddingType, optional): Padding type. Defaults to PaddingType.ZERO.
            kernel_initializer (tf.keras.initializers.Initializer, optional): Initialization of the convolutions kernels.
                Defaults to None.
            gamma_initializer (tf.keras.initializers.Initializer, optional): Initialization of the normalization layers.
                Defaults to None.
        """
        super(ResBlockUp, self).__init__()
        self.activation = activation
        self.use_spec_norm = use_spec_norm
        self.use_bias = use_bias
        self.padding = padding
        self.kernel_initializer = HeAlphaUniform() if kernel_initializer is None else kernel_initializer
        self.gamma_initializer = HeAlphaUniform() if gamma_initializer is None else gamma_initializer

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer depending on the `input_shape` (output shape of the previous layer).

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor to this layer.
        """
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
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
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


@tf.keras.utils.register_keras_serializable(package="Custom", name="scale")
class ScaleLayer(tf.keras.layers.Layer):
    """Trainable non-negative scalar that might be used as a trainable gate.

    It is a single learnable weight that is multiplied by all weights of the input tensor through broadcasting.
    """

    def __init__(self, initializer: Optional[tf.keras.initializers.Initializer] = None) -> None:
        """Initializes the `ScaleLayer` layer.

        Args:
            initializer (tf.keras.initializers.Initializer, optional): Initializer used to initialize the scalar weight.
                Defaults to None..
        """
        super(ScaleLayer, self).__init__()
        self.initializer = tf.keras.initializers.Ones() if initializer is None else initializer
        self.scale = self.add_weight(shape=[1], initializer=initializer, constraint=NonNegative(), trainable=True)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Calls the `ScaleLayer` layer.

        Args:
            inputs (tf.Tensor): Tensor of shape (batch, ...).

        Returns:
            Non-negative scaled version of the input. Tensor of shape (batch, ...).
        """
        return inputs * self.scale

    def get_config(self) -> Dict[str, Any]:
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(ScaleLayer, self).get_config()
        config.update({"scale": self.scale, "initializer": self.initializer})
        return config


class ScalarGatedSelfAttention(tf.keras.layers.Layer):
    """Scaled dot-product self attention that is gated by a learnable scalar.

    Info:
        Implementation as used in the [VoloGAN paper](https://arxiv.org/abs/2207.09204).

    Architecture:
        ```mermaid
        flowchart LR
            i([input_tensor])-->f & g & h
            subgraph ScalarGatedSelfAttention
                f[dsk.layers.DenseBlock] --> t[Transpose]
                g[dsk.layers.DenseBlock] & t --> m1[Multiply] --> s[SoftMax]
                h[dsk.layers.DenseBlock] & s --> m2[Multiply] --> v[DenseBlock] --> sc[dsk.layers.ScaleLayer]
            end
            sc -->o([output_tensor])
        ```
    """

    def __init__(
        self,
        use_spec_norm: bool = False,
        intermediate_channel: Optional[bool] = None,
        kernel_initializer: Optional[tf.keras.initializers.Initializer] = None,
        gamma_initializer: Optional[tf.keras.initializers.Initializer] = None,
    ) -> None:
        """Initializes the `ScalarGatedSelfAttention` layer.

        Args:
            use_spec_norm (bool, optional): If true, applies spectral normalization to convolutional and dense layers.
                Defaults to False.
            intermediate_channel (Optional[bool], optional): Intermediate channels for the self attention mechanism.
                Defaults to None.
            kernel_initializer (tf.keras.initializers.Initializer, optional): Initialization of the kernel weights.
                Defaults to None.
            gamma_initializer (tf.keras.initializers.Initializer, optional): Initialization of the normalization layers.
                Defaults to None.
        """
        super(ScalarGatedSelfAttention, self).__init__()
        self.use_spec_norm = use_spec_norm
        self.intermediate_channel = intermediate_channel
        self.kernel_initializer = HeAlphaUniform() if kernel_initializer is None else kernel_initializer
        self.gamma_initializer = HeAlphaUniform() if gamma_initializer is None else gamma_initializer

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build layer depending on the `input_shape` (output shape of the previous layer).

        Args:
            input_shape (tf.TensorShape): Shape of the input tensor to this layer.
        """
        super(ScalarGatedSelfAttention, self).build(input_shape)
        batch_size, height, width, num_channel = input_shape
        if self.intermediate_channel is None:
            self.intermediate_channel = num_channel // 8

        self.w_f = DenseBlock(
            units=self.intermediate_channel,
            use_spec_norm=self.use_spec_norm,
            number_of_layers=1,
            activation=None,
            apply_final_normalization=False,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            gamma_initializer=self.gamma_initializer,
        )
        self.w_g = DenseBlock(
            units=self.intermediate_channel,
            use_spec_norm=self.use_spec_norm,
            number_of_layers=1,
            activation=None,
            apply_final_normalization=False,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            gamma_initializer=self.gamma_initializer,
        )
        self.w_h = DenseBlock(
            units=self.intermediate_channel,
            use_spec_norm=self.use_spec_norm,
            number_of_layers=1,
            activation=None,
            apply_final_normalization=False,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            gamma_initializer=self.gamma_initializer,
        )
        self.w_fgh = DenseBlock(
            units=num_channel,
            use_spec_norm=self.use_spec_norm,
            number_of_layers=1,
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
        """Serialization of the object.

        Returns:
            Dictionary with the class' variable names as keys.
        """
        config = super(ScalarGatedSelfAttention, self).get_config()
        config.update(
            {
                "use_spec_norm": self.use_spec_norm,
                "intermediate_channel": self.intermediate_channel,
                "gamma_initializer": self.gamma_initializer,
                "kernel_initializer": self.kernel_initializer,
            }
        )
        return config
