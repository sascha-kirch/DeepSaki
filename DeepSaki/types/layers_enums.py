from enum import Enum
from enum import auto

class MultiplicationType(Enum):
    """`Enum` used to define how two matrices shall be multiplied.

    Attributes:
        ELEMENT_WISE: Indicates to apply an element-wise multiplication of 2 tensors.
        MATRIX_PRODUCT: Indicates to apply a matrix-product between 2 tensors.
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
    """

    LOW_PASS = auto()
    HIGH_PASS = auto()

class PaddingType(Enum):
    """`Enum` used to define different types of padding opperations.

    Attributes:
        ZERO: Indicates to apply a zero padding operations.
        REFLECTION: Indicates to apply a reflection padding operation.
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
    RESAMPLE_AND_CONV=auto()
    TRANSPOSE_CONV=auto()
    DEPTH_TO_SPACE=auto()

class DownSampleType(Enum):
    AVG_POOLING = auto()
    CONV_STRIDE_2 = auto()
    MAX_POOLING = auto()
    SPACE_TO_DEPTH = auto()

class LinearLayerType(Enum):
    MLP=auto()
    CONV_1x1= auto()
