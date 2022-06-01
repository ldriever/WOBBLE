"""
This file contains custom-made exceptions suitable for use with the ModalAnalysis class
"""


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
