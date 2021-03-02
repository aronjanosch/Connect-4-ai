from enum import Enum
from typing import List, Optional, NamedTuple
import copy

import numpy as np

from common.players import Player, PLAYER_NONE, PLAYER_1, other_player, PLAYER_2
from common import ensure
from common.base import Column, Row, Coordinates, ROW_NONE


class BoardShape(NamedTuple):
    """
    A tuple of (rows, columns)
    """

    # number of rows
    rows: int

    # number of columns
    columns: int


class GameState(Enum):
    """
    Simple game state Enum.
    """

    # The game is ongoing, nobody has won yet and move can be made
    ONGOING = 1

    # The game has ended in a draw
    DRAW = 2

    # The gams has ended because a player has connected four stones.
    WON = 3


class Board(object):
    """
    A class representing a board state as well as a bunch of useful utility functions.
    """

    # The initial board state
    init_board: np.ndarray

    # The current board state
    current_board: np.ndarray

    # The current player
    player: Player

    # The board shape
    shape: BoardShape

    def __init__(self, board_data: Optional[np.ndarray] = None, player: Optional[Player] = None):
        """
        Create a new board

        :param board_data: optionally takes a board state from a numpy array
        :param player: The current player, Player 1 if not set
        """
        self.shape = BoardShape(6, 7)
        self.init_board = np.zeros(self.shape, dtype=np.int8)
        self.init_board[:] = PLAYER_NONE

        self.player = player if player is not None else PLAYER_1

        if board_data is not None and board_data.shape == self.shape:
            self.current_board = board_data.copy()
        else:
            self.current_board = self.init_board.copy()

    def set(self, coords: Coordinates, player: Player, overwrite: bool = False) -> None:
        """
        Sets a given stone on the Board

        :param coords: The coordinates to set
        :param player: The player to set the stone for
        :param overwrite: Specifies if a present stone should be overwritten
        """
        ensure.valid_coordinates(coords)

        if self.get(coords) != PLAYER_NONE and not overwrite:
            raise Exception(f"Not overwriting player at [{coords.row}, {coords.column}]")

        self.current_board[coords.row, coords.column] = player

    def get(self, coords: Coordinates) -> Player:
        """
        Gets the player at the coordinates specified.

        :param coords: The coordinates to check
        :return: The player on this board at coords
        """

        ensure.valid_coordinates(coords)
        return self.current_board[coords.row, coords.column]

    def __getitem__(self, item: Coordinates) -> Player:
        """
        Indexer for coordinates. Returns the player at the given coordinates.

        :param item: The coordinates to check
        :return: The player at the given coordinates
        """
        return self.get(item)

    def __setitem__(self, item: Coordinates, value: Player) -> None:
        """
        Sets the player at the given coordinates, potentially overwriting.

        :param item: The coordinates
        :param value: The player to set
        """
        return self.set(item, value, overwrite=True)

    def next_free_row(self, col: Column) -> Row:
        """
        Gets the index of the next free row in a given column

        :param col: The column to check
        :return: The next free row index in the given column of ROW_NONE if none is available
        """

        ensure.valid_column(col)

        column_values = self.current_board[:, col]
        for i in range(len(column_values)):
            if column_values[i] == PLAYER_NONE:
                return i

        return ROW_NONE

    def actions(self) -> List[Column]:
        """
        Gets all playable moves (or actions, columns) on the current board.

        :return: A list of playable columns
        """
        return [c for c in range(self.shape.columns) if self.can_drop(Column(c))]

    def can_drop(self, col: Column) -> bool:
        """
        Checks if a stone can be dropped at the specified column

        :param col: Column
        :return: True if a stone can be dropped at the given position
        """
        ensure.valid_column(col)

        return self.next_free_row(col) != ROW_NONE

    def is_final(self):
        """
        Checks if the board is final, meaning that no stone can be dropped anymore

        :return: True if the board is in a final state.
        """
        return not self.is_playable()

    def is_playable(self):
        """
        Checks if the board is still playable, meaning that a stone can be dropped on at least one column

        :return: True if the board is playable
        """
        for c in range(self.shape.columns):
            if self.can_drop(Column(c)):
                return True

        return False

    def get_state(self) -> GameState:
        """
        Gets the game state of this board.

        :return: The game state
        """
        if self.is_playable():
            return GameState.ONGOING
        elif self.check_winner(self.player) or self.check_winner(other_player(self.player)):
            return GameState.WON
        else:
            return GameState.DRAW

    def drop_piece_copy(self, col: Column, player: Optional[Player] = None) -> 'Board':
        """
        Creates a copy of this board and drops a stone on the copy. This is equivalent to:

        >>> b.copy().drop_piece(col, player)

        :param col: The column to drop the stone at
        :param player: The player to drop the stone as. If not set, use current player.
        :return: The copied board after the stone has been dropped
        """
        b = self.copy()
        b.drop_piece(col, player)
        return b

    def drop_piece(self, col: Column, player: Optional[Player] = None) -> None:
        """
        Drops a stone at the specified column and toggles the current player.

        :param col: The column to drop the stone at
        :param player: The player to drop the stone as. If not set, use current player.
        """
        ensure.valid_column(col)

        if player is None:
            player = self.player

        next_free = self.next_free_row(col)
        if next_free == ROW_NONE:
            raise Exception(f"Cannot drop on column {col}")
        else:
            coords = Coordinates(row=next_free, column=col)
            self.set(coords, player)

            self.player = other_player(self.player)


    def check_winner(self, player: Optional[Player] = None) -> bool:
        """
        Checks if the board is won by the specified player.

        :param player: The player to check a win for.
        :return: True if the player has won
        """
        n = 4

        if player is None:
            player = other_player(self.player)

        rows, cols = self.shape
        rows_edge = rows - n + 1
        cols_edge = cols - n + 1

        for i in range(rows):
            for j in range(cols_edge):
                if np.all(self.current_board[i, j:j + n] == player):
                    return True

        for i in range(rows_edge):
            for j in range(cols):
                if np.all(self.current_board[i:i + n, j] == player):
                    return True

        for i in range(rows_edge):
            for j in range(cols_edge):
                block = self.current_board[i:i + n, j:j + n]
                if np.all(np.diag(block) == player):
                    return True
                if np.all(np.diag(block[::-1, :]) == player):
                    return True

        return False

    def encode(self) -> np.array:
        """
        Encodes the board using one-hot encoding (only 0 or 1).

        :return: The board encoded in one-hot encoding as a numpy array.
        """
        b = self.current_board
        encoded = np.zeros([6, 7, 3]).astype(int)
        encoding_dict = {
            PLAYER_1: 0,
            PLAYER_2: 1
        }

        for row in range(self.shape.rows):
            for column in range(self.shape.columns):
                p = b[row, column]
                if p != PLAYER_NONE:
                    enc_translate = encoding_dict[p]
                    encoded[row, column, enc_translate] = 1

        if self.player == PLAYER_1:
            encoded[:, :, 2] = PLAYER_1  # set player to move

        return encoded

    def mirrored_copy(self) -> 'Board':
        """
        Creates and returns a (vertically) mirrored copy of this board.

        :return: Returns the mirrored board as a copy.
        """
        b = self.copy()
        b.current_board = np.flip(self.current_board, 1)
        return b

    def copy(self) -> 'Board':
        """
        Creates a deep copy of this board object.

        :return: A deep copy of this board.
        """
        return copy.deepcopy(self)

    def pretty_print(self) -> str:
        """
        Returns `board` converted to a human readable string representation,
        to be used when playing or printing diagnostics to the console (stdout). The piece in
        board[0, 0] should appear in the lower-left. Here's an example output:
        |==============|
        |              |
        |              |
        |    X X       |
        |    O X X     |
        |  O X O O     |
        |  O O X X     |
        |==============|
        |0 1 2 3 4 5 6 |
        """

        bar = f"|{'=' * self.shape.columns * 2}|"
        r = f"{bar}\n"

        for row in reversed(range(self.shape.rows)):
            r_row = ""
            for column in range(self.shape.columns):
                p = self.get(Coordinates(row, column))
                r_row += f"{'X' if p == PLAYER_1 else ('O' if p == PLAYER_2 else ' ')} "

            r += f"|{r_row}|\n"

        r += f"{bar}\n"

        r_footer = ""
        for i in range(self.shape.columns):
            r_footer += f"{i} "

        r += f"|{r_footer}|\n"
        return r

    def __str__(self):
        """
        Converts this board to a string representation using pretty_print().

        :return: A string representation of this board
        """
        return self.pretty_print()

    def __hash__(self):
        r = hash(self.current_board.data.tobytes())
        return r
