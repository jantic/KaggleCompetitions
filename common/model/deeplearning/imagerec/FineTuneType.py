from enum import Enum


class FineTuneType(Enum):
    NONE = 0
    OUTPUT_ONLY = 1
    LAST_FULLY_CONNECTED_TO_OUTPUT = 2
    ALL_FULLY_CONNECTED_TO_OUTPUT = 3
    LAST_CONV_TO_OUTPUT = 4
