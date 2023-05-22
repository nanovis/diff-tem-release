from enum import Enum


class BinaryFileFormat(Enum):
    MRC_INT = "mrc-int"
    MRC = "mrc"
    RAW = "raw"

    @staticmethod
    def options():
        return ["mrc", "mrc-int", "raw"]
