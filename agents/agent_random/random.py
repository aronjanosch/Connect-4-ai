import random as rnd
from typing import Optional, Tuple

from common.base import SavedState, PlayerAction
from common.board import Board
from common.players import Player


def generate_move_random(
    board: Board, player: Player, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Generates a random move for a given board state.
    :param board: The board to generate the move for
    :param player: The player to generate a move for
    :param saved_state: The saved state (not used here)
    :return: The randomly generated move and the passed saved state
    """

    print(f"I'm the random agent. Playing as {board.player}")

    available_columns = board.actions()

    action = PlayerAction(rnd.choice(available_columns))
    print(f"Choosing to play {action}!")

    return action, saved_state
