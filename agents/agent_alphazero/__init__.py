from typing import Optional, Tuple

from common.base import SavedState, PlayerAction
from common.board import Board, PLAYER_NONE
from common.players import Player
from .monte_carlo import generate_move as generate_move_mc
from .neural_net import Connect4Network, load_connect4network

# the neural network stored in the background
nn: Optional[Connect4Network] = None


def generate_move(board: Board, player: Player, saved_state: Optional[SavedState]) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Genrates move using trained model and MCTS
    @param board: connect4board
    @param saved_state:
    @return: PlayerAction
    """
    from agents.agent_alphazero.alpha_zero_args import AlphaZeroArgs
    global nn

    # translate the board
    b = board.copy()
    b.current_board[b.current_board == 0] = PLAYER_NONE

    alpha_zero_args = AlphaZeroArgs()

    if nn is None:
        nn = load_connect4network(alpha_zero_args, alpha_zero_args.playing_iteration)

    print(f"I'm AlphaZero playing as Player {b.player}")

    return PlayerAction(generate_move_mc(nn, b, alpha_zero_args)), saved_state
