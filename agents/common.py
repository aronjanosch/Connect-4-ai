from enum import Enum
from typing import Callable, Tuple, List, Set, Union, Iterable, Generator
from typing import Optional

import numpy as np

from common.players import PLAYER_NONE

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

PlayerAction = np.int8  # The column to be played


class GameState(Enum):
    IS_WIN = 1
    IS_WON = IS_WIN
    IS_LOST = 2
    IS_DRAW = -1
    STILL_PLAYING = 0


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).
    """
    return np.asarray([[NO_PLAYER] * 7] * 6, dtype=BoardPiece)


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
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

    bar = f"|{'=' * board.shape[1] * 2}|"
    r = f"{bar}\n"

    for row in reversed(range(board.shape[0])):
        r_row = ""
        for column in range(board.shape[1]):
            p = board[row][column]
            r_row += f"{'X' if p == PLAYER1 else ('O' if p == PLAYER2 else ' ')} "

        r += f"|{r_row}|\n"

    r += f"{bar}\n"

    r_footer = ""
    for i in range(board.shape[1]):
        r_footer += f"{i} "

    r += f"|{r_footer}|\n"
    return r


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.

    :param pp_board: The string representation
    :return: The parser board state
    """

    rows = pp_board.split("\n")
    data_rows = list(reversed(rows[1:7]))

    board = initialize_game_state()
    for i in range(board.shape[0]):
        row = data_rows[i]
        values = [
            row[1 + 2 * j] for j in range(board.shape[1])
        ]
        board_pieces = [
            PLAYER1 if v == "X" else (PLAYER2 if v == "O" else NO_PLAYER) for v in values
        ]
        board[i] = board_pieces

    return board


def apply_player_action(
        board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """
    if copy:
        board = np.copy(board)

    next_row = -1

    for i in range(board.shape[0]):
        if board[i, action] == NO_PLAYER:
            next_row = i
            break

    if next_row != -1:
        board[next_row, action] = player

    return board


def connected_four(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """

    n_rows = board.shape[0]
    n_columns = board.shape[1]
    size = max(n_rows, n_columns)

    def is_valid(pos):
        return bool(0 <= pos[0] < n_rows and 0 <= pos[1] < n_columns)

    line_vectors = [
        np.array([1, 0]),
        np.array([0, 1]),
        np.array([1, -1]),
        np.array([1, 1])
    ]

    rows_to_check = range(n_rows)
    columns_to_check = range(n_columns)

    if last_action is not None:
        columns_to_check = [last_action]

    for r in rows_to_check:
        for c in columns_to_check:
            pos = np.array([r, c])
            lines = [[pos + i * lv for i in range(-size + 1, size)] for lv in line_vectors]
            lines_valid = [[l for l in lvs if is_valid(l)] for lvs in lines]
            for line in lines_valid:
                accu = 0
                path = []
                for p in line:
                    if board[p[0], p[1]] == player:
                        accu += 1
                        path.append(p)
                        if accu >= 4:
                            print(f"Found 4 in a row for player {player} on path {path}")
                            return True
                    else:
                        accu = 0
                        path.clear()

    return False


def check_end_state(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """

    def is_board_full():
        for row in board:
            for b in row:
                if b == NO_PLAYER:
                    return False

        return True

    has_won = connected_four(board, player, last_action=last_action)
    enemy_won = connected_four(board, player, last_action=last_action)
    if has_won:
        return GameState.IS_WIN
    elif enemy_won:
        return GameState.IS_LOST
    else:
        if is_board_full():
            return GameState.IS_DRAW
        else:
            return GameState.STILL_PLAYING



