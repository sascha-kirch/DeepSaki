from enum import Enum
from enum import auto

class CurrentOptimizer(Enum):
    """`Enum` used to define how two matrices shall be multiplied.

    Attributes:
        SGD: Indicates to switch to the SGD optimizer.
        ADAM: Indicates to switch to the ADAM optimizer.
        NADAM: Indicates to switch to the NADAM optimizer.
    """
    SGD = auto()
    ADAM = auto()
    NADAM = auto()
