from typing import Union, Generator, Callable, Set, List, Iterable

import numpy as np

from common.base import PlayerAction
from common.players import PLAYER_NONE, Player


class Vector2:
    """
    A vector of two components x and y
    """

    x: PlayerAction
    y: int

    def __init__(self, x: PlayerAction, y: int):
        self.x = x
        self.y = y

    def __add__(self, other: 'Vector2'):
        """
        Simple vector addition

        :param other: the other vector
        :return: this vector + other vector
        """
        return self.__class__(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector2'):
        """
        Simple vector subtraction

        :param other: the other vector
        :return: this vector - other vector
        """
        return self.__class__(self.x - other.x, self.y - other.y)

    def __mul__(self, other: Union[int, 'Vector2']):
        """
        Simple vector multiplication

        :param other: the factor, either a scalar or another Vector2
        :return:
        """
        if type(other) is int:
            return self.__class__(self.x * other, self.y * other)
        elif type(other) is Vector2:
            return self.__class__(self.x * other.x, self.y * other.y)

    def __neg__(self):
        """
        Simple vector negation

        :return: This vector with its components negated.
        """
        return self.__class__(-self.x, -self.y)


class Position(Vector2):
    """
    This class behaves the same as a Vector2, with some more features
    """

    x: PlayerAction
    y: int

    def __init__(self, x: PlayerAction, y: int):
        super().__init__(x, y)

    @property
    def column(self):
        """
        Alias for x

        :return: The x coordinate
        """
        return self.x

    def __int__(self) -> int:
        """
        Converts this position to a scalar

        :return: This position expressed as a scalar
        """
        return int(self.y * 7 + self.column)

    def __hash__(self) -> int:
        """
        Hashes this position.

        :return: The hash value
        """
        return int(self)

    def __eq__(self, other):
        """
        Checks for equality with another Position.

        :param other: The other Position
        :return: Returns a boolean representing if the two are equal
        """
        return self.__hash__() == other.__hash__()

    def __lt__(self, other: 'Position') -> bool:
        """
        Checks if this position is lower than other.
        This ordering is not intuitively an "order", but it compares the ordinals of the positions,
        which is useful for normalizing.

        :param other: Other position
        :return: True if this position is lower than other, otherwise False.
        """
        return int(self) < int(other)

    def __str__(self):
        """
        Converts this position to a string representation in the form ColumnRow, e.g. a1 for
        x=0, y=1.

        :return:
        """
        col_names = ["a", "b", "c", "d", "e", "f", "g"]
        if self.is_valid():
            return f"{col_names[self.column]}{self.y + 1}"
        else:
            return f"Invalid [column={self.column}, y={self.y}]"

    def __repr__(self):
        return self.__str__()

    def is_valid(self):
        """
        Checks if a Position has valid coordinates.

        :return: True if both coordinates are valid
        """
        return 0 <= self.column <= 6 and 0 <= self.y <= 5

    def is_lonely(self, board: np.ndarray):
        """
        Checks if a stone is lonely on a board, meaning it is not connected to another stone
        of the same player.

        :param board: The board to check against
        :return: True if the stone on this Position on the Board is lonely
        """
        player = self.get_player(board)
        for nb in self.get_neighbours():
            if nb.get_player(board) == player:
                return False

        return True

    def is_empty(self, board: np.ndarray):
        """
        Checks if there is no stone at this Position on the board.

        :param board: The board to check against
        :return: True if the board is empty at this Position.
        """
        return self.get_player(board) == PLAYER_NONE

    def get_player(self, board: np.ndarray) -> Player:
        """
        Gets the player on the board at this position.

        :param board: The board to check against
        :return: The player at this Position on the board
        """
        return board[self.y, self.column]

    def can_be_played(self, board: np.ndarray) -> bool:
        """
        Checks if the board can immediately be played at this Position.

        :param board: The board to check against
        :return: True if a stone can be placed at this Position in the immediate turn.
        """
        if not self.is_valid() or not self.is_empty(board):
            return False

        # check if there is a stone below this position
        below = Position(self.column, self.y - 1)
        if below.is_valid():
            return below.get_player(board) != PLAYER_NONE
        else:
            return True

    def get_neighbours(self) -> Generator['Position', None, None]:
        """
        Enumerates the neighboring Positions of this one.

        :return: A generator for all neighbors
        """
        for dc in range(-1, 2):
            for dy in range(-1, 2):
                if dc != 0 or dy != 0:
                    p = self + Vector2(dc, dy)
                    if p.is_valid():
                        yield p


class Beam:
    """
    A beam is an abstraction for a line on a board in any direction.
    It is used to find connected stones in a row / column / diagonally.
    """
    def __init__(self, start: Position, direction: Vector2, length: int):
        """
        Creates a beam

        :param start: Starting position of this beam
        :param direction: A direction vector
        :param length: The length of this beam
        """
        self.start = start
        self.direction = direction
        self.length = length

        # a zero length beam does not make sense
        if length < 1:
            raise Exception("Beam must not have a length below 1.")

    def __contains__(self, item: Position) -> bool:
        """
        Checks if a position is included in this beam

        :param item: The position to check
        :return: True if this beam contains the Position
        """
        pos_set = set(self.get_positions())
        return item in pos_set

    def __hash__(self):
        """
        Calculates a (normalized) hash for this beam

        :return: The hash value
        """
        return hash((self.normalized_start(), self.normalized_end()))

    def __eq__(self, other: 'Beam') -> bool:
        """
        Checks if this beam is equal to another.

        :param other: The other beam
        :return: True if the two beams are equal
        """
        my_end = self.end()
        other_end = other.end()
        return (self.start == other.start and my_end == other_end) or (
                    self.start == other_end and my_end == other.start)

    def __str__(self):
        """
        Converts this beam to a string representation.

        :return: The string representation
        """
        return f"{self.start} -> {self.end()} (L={self.length})"

    def __repr__(self):
        return str(self)

    def normalized_start(self) -> Position:
        """
        Gets the normalized start position of this beam

        :return: The normalized start position
        """
        end = self.end()
        if self.start < end:
            return self.start
        else:
            return end

    def normalized_end(self) -> Position:
        """
        Gets the normalized end position of this beam

        :return: The normalized end position
        """
        end = self.end()
        if self.start < end:
            return end
        else:
            return self.start

    def same_orientation(self, other: 'Beam') -> bool:
        """
        Checks if this beam has the same orientation as the other (horizontally / vertically / diagonally).

        :param other: The other beam
        :return: True if the orientations are the same
        """
        return self.direction == other.direction or self.direction == -other.direction

    def get_positions(self) -> List[Position]:
        """
        Gets a position list of all positions contained in this beam.

        :return: The positon list
        """
        pos_list = []
        for i in range(self.length):
            v = self.direction * i
            pos_list.append(self.start + v)

        return pos_list

    def is_valid(self) -> bool:
        """
        Checks if this beam has a valid starting and ending point.

        :return: True if start and end point are valid.
        """
        return self.start.is_valid() and self.end().is_valid()

    def end(self) -> Position:
        """
        Gets the end position of the beam

        :return: The end position
        """
        return self.start + (self.direction * (self.length - 1))

    def neighbour_left(self) -> Position:
        """
        Calculates the left adjacent neighbor position to this Beam

        :return: The left neighbor
        """
        return self.start - self.direction

    def neighbour_left_free(self, board: np.ndarray) -> bool:
        """
        Checks if the left neighbor of this beam is free on the specified Board.

        :param board: The board to check against
        :return: True if the left neighbor position is free
        """
        nb = self.neighbour_left()
        return nb.is_valid() and nb.get_player(board) == PLAYER_NONE

    def neighbour_right_free(self, board: np.ndarray) -> bool:
        """
        Checks if the right neighbor of this beam is free on the specified Board.

        :param board: The board to check against
        :return: True if the right neighbor position is free
        """
        nb = self.neighbour_right()
        return nb.is_valid() and nb.get_player(board) == PLAYER_NONE

    def neighbours_left_playable_count(self, board: np.ndarray) -> int:
        """
        Gets the number of playable Positions left to this Beam on the specified Board.

        :param board: The board to check against
        :return: The number of playable positions left to this Beam
        """
        cnt = 0
        nb = self.neighbour_left()

        while nb.can_be_played(board):
            cnt += 1
            nb = nb - self.direction

        return cnt

    def neighbours_right_playable_count(self, board: np.ndarray):
        """
        Gets the number of playable Positions right to this Beam on the specified Board.

        :param board: The board to check against
        :return: The number of playable positions right to this Beam
        """
        cnt = 0
        nb = self.neighbour_right()

        while nb.can_be_played(board):
            cnt += 1
            nb = nb + self.direction

        return cnt

    def neighbour_right(self) -> Position:
        """
        Calculates the right adjacent neighbor position to this Beam

        :return: The right neighbor
        """
        return self.end() + self.direction

    def extend_left(self) -> 'Beam':
        """
        Extends this beam to the left by one.

        :return: A new beam, extended by one to the left
        """
        return Beam(self.neighbour_left(), self.direction, self.length + 1)

    def extend_right(self) -> 'Beam':
        """
        Extends this beam to the right by one.

        :return: A new beam, extended by one to the right
        """
        return Beam(self.start, self.direction, self.length + 1)

    def extend_while_player(self, board: np.ndarray, player: Player) -> 'Beam':
        """
        Extends this beam on both ends while the neighboring positions are conquered by the specified player
        on the specified board.

        :param board: The board to check against
        :param player: The player to check for
        :return: A new beam which is this beam extended by the maximum stones conquered by player.
        """
        return self.extend_while(lambda p: p.get_player(board) == player)

    def extend_while_blank(self, board: np.ndarray) -> 'Beam':
        """
        Extends this beam on both ends while the neighboring positions are empty on the specified board.

        :param board: The board to check against
        :return: A new beam which is this beam extended by the maximum empty positions.
        """
        return self.extend_while(lambda p: p.get_player(board) == PLAYER_NONE)

    def extend_while_player_or_blank(self, board: np.ndarray, player: Player) -> 'Beam':
        """
        Extends this beam on both ends while the neighboring positions are conquered by the specified player
        or empty on the specified board.

        :param board: The board to check against
        :param player: The player to check for
        :return: A new beam which is this beam extended by the maximum stones conquered by player or empty.
        """
        return self.extend_while(lambda p: p.get_player(board) in (player, PLAYER_NONE))

    def extend_while(self, condition: Callable[[Position], bool]):
        """
        Extends this beam on both ends while a condition is met for the respective position.

        :param condition: A predicate which is the condition to be met
        :return: A new beam which is this beam extended on both sides as long as the condition was met.
        """
        b = self

        while b.neighbour_left().is_valid() and condition(b.neighbour_left()):
            b = b.extend_left()
        while b.neighbour_right().is_valid() and condition(b.neighbour_right()):
            b = b.extend_right()

        return b

    def extend_to_four_with_gap(self, board: np.ndarray, player: Player, max_gap_size: int = 1):
        """
        Tries to extend this beam while the stones don't change, consuming at maximum max_gap_size empty Positions.

        :param board: The board to extend on
        :param player: The player to extend for
        :param max_gap_size: The maximum empty Positions to conquer
        :return: A new beam, extended accordingly.
        """
        def consume_gap() -> bool:
            nonlocal max_gap_size
            if max_gap_size > 0:
                max_gap_size -= 1
                return True
            else:
                return False

        return self.extend_while(
            lambda p: p.get_player(board) == player or (p.is_empty(board) and consume_gap())
        )


class PositionCollection:
    """
    A position collection is just a simple set of Position objects together with a couple utility functions.
    """
    positions: Set[Position]

    def __init__(self, board: np.ndarray, player: Player, positions: Union[List[Position], Set[Position]]):
        """
        Creates a position collection.

        :param board: The board to locate the positions at
        :param player: The player to be considered
        :param positions: A list of positions
        """
        self.board = board
        self.player = player
        self.positions = set(positions)

    def __iter__(self) -> Iterable[Position]:
        """
        Iterates over all positions in this collection

        :return: An iterable over the positions in this collection
        """
        return iter(self.positions)

    def get_beams(self) -> List[Beam]:
        """
        Gets a list of all beams found for this position collection.

        :return: List of beams which can be created for this position collection
        """
        beams = set()

        # All directions available
        directions = [
            Vector2(0, 1),
            Vector2(0, -1),
            Vector2(1, 0),
            Vector2(-1, 0),
            Vector2(1, 1),
            Vector2(-1, -1),
            Vector2(1, -1),
            Vector2(-1, 1)
        ]

        # iterate over the positions, finding beams for each position in every possible direction
        for p in self.positions:
            for d in directions:
                b = Beam(p, d, 1).extend_while_player(self.board, self.player)
                beams.add(b)

        return list(beams)


def get_player_positions(
        board: np.ndarray, player: Player
) -> PositionCollection:
    """
    Finda all player positions on a given board

    :param board: The board to check against
    :param player: The player to find positions for
    :return: The position collection for the given player on the given board
    """
    visited = []
    for y in range(6):
        for c in range(7):
            p = Position(PlayerAction(c), y)
            if p.get_player(board) == player:
                visited.append(p)

    return PositionCollection(board, player, visited)
