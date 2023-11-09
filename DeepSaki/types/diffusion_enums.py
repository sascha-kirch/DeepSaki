from enum import Enum
from enum import auto

class ScheduleType(Enum):
    LINEAR = auto()
    SIGMOID = auto()
    COSINE = auto()


class variance_type(Enum):
    LEARNED_RANGE = auto()
    LEARNED = auto()
    LOWER_BOUND = auto()
    UPPER_BOUND = auto()
