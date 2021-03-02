"""
Provides a couple of ensurance functions which raise an exception when invalid data is passed.
"""

from common.base import Column, Coordinates


def valid_column(col: Column) -> None:
    """
    Ensures that a column index has a valid value.

    :param col: Column index
    """
    if not(0 <= col <= 6):
        raise Exception(f"Invalid column: {col}")


def valid_row(row: Column) -> None:
    """
    Ensures that a row index has a valid value.

    :param row: Row index
    """
    if not(0 <= row <= 5):
        raise Exception(f"Invalid row: {row}")


def valid_coordinates(coords: Coordinates):
    """
    Ensures that a coordinate pair is valid.

    :param coords: The coordinates object
    """
    valid_column(coords.column)
    valid_row(coords.row)