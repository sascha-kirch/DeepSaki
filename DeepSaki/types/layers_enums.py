from enum import Enum
from enum import auto


class MultiplicationType(Enum):
    """`Enum` used to define how two matrices shall be multiplied.

    Attributes:
        ELEMENT_WISE: Indicates to apply an element-wise multiplication of 2 tensors.
        MATRIX_PRODUCT: Indicates to apply a matrix-product between 2 tensors.

    **Example:**
    ```python hl_lines="3"
    import DeepSaki as ds
    layer = ds.layers.FourierConvolution2D(
                multiplication_type = ds.types.MultiplicationType.MATRIX_PRODUCT
            )
    ```
    """

    ELEMENT_WISE = auto()
    MATRIX_PRODUCT = auto()


class FrequencyFilter(Enum):
    """`Enum` used to define valid filters for `rFFT2DFilter`.

    Attributes:
        LOW_PASS: Indicates that low frequency components shall be kept and high frequency components shall be
            filtered.
        HIGH_PASS: Indicates that high frequency components shall be kept and low frequency components shall be
            filtered.

    **Example:**
    ```python hl_lines="3"
    import DeepSaki as ds
    layer = ds.layers.rFFT2DFilter(
                filter_type = ds.types.FrequencyFilter.LOW_PASS
            )
    ```
    """

    LOW_PASS = auto()
    HIGH_PASS = auto()


class PaddingType(Enum):
    """`Enum` used to define different types of padding opperations.

    Attributes:
        ZERO: Indicates to apply a zero padding operations.
        REFLECTION: Indicates to apply a reflection padding operation.

    **Example:**
    ```python hl_lines="3"
    import DeepSaki as ds
    layer = ds.models.LayoutContentDiscriminator(
                padding = ds.types.PaddingType.ZERO
            )
    ```
    """

    NONE = auto()
    ZERO = auto()
    REFLECTION = auto()


class InitializerFunc(Enum):
    """`Enum` used to define different types of initializer functions.

    Attributes:
        RANDOM_NORMAL: Corresponds to a random normal initializer function.
        RANDOM_UNIFORM: Corresponds to a random uniform initializer function.
        GLOROT_NORMAL: Corresponds to a Glorot normal initializer function.
        GLOROT_UNIFORM: Corresponds to a Glorot uniform initializer function.
        HE_NORMAL: Corresponds to a He normal initializer function.
        HE_UNIFORM: Corresponds to a He uniform initializer function.
        HE_ALPHA_NORMAL: Corresponds to a He Alpha normal initializer function.
        HE_ALPHA_UNIFORM: Corresponds to a He Alpha Uniform initializer function.

    **Example:**
    ```python hl_lines="3"
    import DeepSaki as ds
    layer = ds.layers.get_initializer(
                initializer = ds.types.InitializerFunc.HE_ALPHA_NORMAL
            )
    ```
    """

    NONE = auto()
    RANDOM_NORMAL = auto()
    RANDOM_UNIFORM = auto()
    GLOROT_NORMAL = auto()
    GLOROT_UNIFORM = auto()
    HE_NORMAL = auto()
    HE_UNIFORM = auto()
    HE_ALPHA_NORMAL = auto()
    HE_ALPHA_UNIFORM = auto()


class UpSampleType(Enum):
    """`Enum` used to define different types of up sampling functions.

    Attributes:
        RESAMPLE_AND_CONV: Use a resampling technique like interpolation followed by a convolution.
        TRANSPOSE_CONV: Use a transpose convolution.
        DEPTH_TO_SPACE: Use a depth to space operation after a convolution the adjust the filters.

    **Example:**
    ```python hl_lines="3"
    import DeepSaki as ds
    layer = ds.layers.Decoder(
                upsampling = ds.types.UpSampleType.TRANSPOSE_CONV
            )
    ```
    """

    RESAMPLE_AND_CONV = auto()
    TRANSPOSE_CONV = auto()
    DEPTH_TO_SPACE = auto()


class DownSampleType(Enum):
    """`Enum` used to define different types of down sampling functions.

    Attributes:
        AVG_POOLING: Use average pooling.
        CONV_STRIDE_2: Use a convolution with a stride of (2,2).
        MAX_POOLING: Use max pooling.
        SPACE_TO_DEPTH: Use a space to depth operation followed by a convolution the adjust the filters.

    **Example:**
    ```python hl_lines="3"
    import DeepSaki as ds
    layer = ds.layers.Encoder(
                downsampling = ds.types.DownSampleType.SPACE_TO_DEPTH
            )
    ```
    """

    AVG_POOLING = auto()
    CONV_STRIDE_2 = auto()
    MAX_POOLING = auto()
    SPACE_TO_DEPTH = auto()


class LinearLayerType(Enum):
    """`Enum` used to define different types of linear functions.

    Attributes:
        MLP: Uses an MLP.
        CONV_1x1: Use a convolution with a kernel of (1,1).

    **Example:**
    ```python hl_lines="3"
    import DeepSaki as ds
    layer = ds.models.UNet(
                linear_layer_type = ds.types.LinearLayerType.MLP
            )
    ```
    """

    MLP = auto()
    CONV_1x1 = auto()
