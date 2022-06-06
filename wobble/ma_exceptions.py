"""
This file contains custom-made exceptions suitable for use with the ModalAnalysis class
"""

__author__ = "Oisín Morrison, Leonhard Driever"
__credits__ = [
    "Oisín Morrison <oisin.morrison@epfl.ch>",
    "Leonhard Driever <leonhard.driever@epfl.ch>"
]


class FileError(Exception):
    pass


class BoundaryConditionError(Exception):
    pass


class AxisError(Exception):
    pass


class ModeError(Exception):
    pass


class SolutionError(Exception):
    pass
