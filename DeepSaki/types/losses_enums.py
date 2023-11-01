from enum import Enum
from enum import auto

class LossCalcType(Enum):
    """`Enum` to define the loss calculation type used for image based losses derived from `ImageBasedLoss`.

    Attributes:
        PER_IMAGE: Loss is calculated over the entire image.
        PER_CHANNEL: Loss is calculated per channel.
    """

    PER_IMAGE = auto()
    PER_CHANNEL = auto()


class LossType(Enum):
    """`Enum` to define the loss type used for image based losses derived from `ImageBasedLoss`.

    Attributes:
        MAE: Loss is a form of mean absolute error.
        MSE: Loss is a form of mean squared error.
    """

    MAE = auto()
    MSE = auto()
