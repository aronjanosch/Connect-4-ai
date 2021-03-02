from typing import NamedTuple

Column = int
Row = int

# alias
PlayerAction = Column

ROW_NONE = Row(-1)


class Coordinates(NamedTuple):
    """
    A simple coordinate tuple.
    """

    # The row
    row: Row

    # The column
    column: Column


class SavedState:
    pass