import random

import numpy as np
from typing import NamedTuple, Optional, Tuple, Dict

from common.base import Column, PlayerAction, SavedState
from common.players import Player
from common.board import Board, GameState

from .heuristic_1 import calculate_heuristic

# A move order that prefers playing in the middle
moveOrder = [3, 2, 4, 1, 5, 0, 6]


class MiniMaxResult(NamedTuple):
    """
    A result container used by mini_max
    """

    # The best move
    best_move: Optional[Column]

    # The calculated heuristic for best_move
    value: float


class MiniMaxCache2(SavedState):
    """
    Caching class used in MiniMax. Stores MiniMaxResults for a given board with a given maximizing player.
    """

    # the caching dictionary we will use (hash -> stored result)
    cache: Dict[int, MiniMaxResult]

    def __init__(self):
        # initialize an empty cache
        self.cache = {}

    def calc_hash(self, board: Board, max_player: Player):
        """
        Calculates the hash value of a given board state combined with the given maximizing player.
        Mirrored boards with the same maximizing player will have the same hash value.
        :param board: The board state
        :param max_player: The maximizing player
        :return: Returns the calculated hash
        """
        return min(hash((board, max_player)), hash((board.mirrored_copy(), max_player)))

    def put(self, board: Board, max_player: Player, result: MiniMaxResult) -> None:
        """
        Put a result to the cache, potentially overwriting an older one.
        :param board: The board state
        :param max_player: the maximizing player
        :param result: The result to store for (board, max_player)
        """
        hs = self.calc_hash(board, max_player)
        self.cache[hs] = result

    def get(self, board: Board, max_player: Player) -> Optional[MiniMaxResult]:
        """
        Get a previously stored result from the cache, if one is stored.
        :param board: The board state
        :param max_player: The maximizing player
        :return: The stored result, if any
        """
        hs = self.calc_hash(board, max_player)
        if hs in self.cache:
            return self.cache[hs]
        else:
            return None


def mini_max(board: Board, alpha: float, beta: float, depth: int, max_player: Player, cache: MiniMaxCache2) \
        -> MiniMaxResult:
    """
    Execute the MiniMax algorithm with alpha-beta-pruning on a board.
    :param board: The board
    :param alpha: Lower boundary
    :param beta: Upper boundary
    :param depth: Maximum depth
    :param max_player: The maximizing player ID
    :param cache: The MiniMax cache to use
    :return: Returns the best possible result after traversing the tree with the given depth.
    """

    # first, let's see if we already cached this value
    cached_value = cache.get(board, max_player)
    if cached_value is not None:
        return cached_value

    # If we hit max depth or the board is in a final state, calculate and return the heuristic.
    # We cannot decide about the best move here, so leave it empty.
    if depth == 0 or board.get_state() != GameState.ONGOING:
        return MiniMaxResult(None, calculate_heuristic(board, max_player).value)

    # collect a list of available moves from the board and order them according to moveOrder
    # which orders moves based on their distance to the middle column
    available_moves = board.actions()
    available_moves = [move for move in moveOrder if move in available_moves]

    # check if we are maximizing right now
    is_maximizing = (board.player == max_player)

    # store min and max value, initialize with boundaries
    max_value, min_value = alpha, beta
    # initialize the best move randomly
    best_move: Column = random.choice(available_moves)

    # enumerate all available moves. Recurse for each while respecting is_maximizing and adjusting
    # min_value and max_value accordingly.
    for move in available_moves:
        # create a board with the new state
        child_board = board.drop_piece_copy(move)

        if is_maximizing:
            min_result = mini_max(child_board, max_value, beta, depth - 1, max_player, cache)
            if min_result == np.inf:
                # we cannot perform better than +inf.
                return MiniMaxResult(move, np.inf)
            elif min_result.value > max_value:
                max_value = min_result.value
                best_move = move
                if max_value >= beta:
                    # beta pruning
                    break
        else:
            max_result = mini_max(child_board, alpha, min_value, depth - 1, max_player, cache)
            if max_result.value == -np.inf:
                # we cannot perform better than -inf.
                return MiniMaxResult(move, -np.inf)
            elif max_result.value < min_value:
                min_value = max_result.value
                best_move = move
                if min_value <= alpha:
                    # alpha pruning
                    break

    # return the best move and according heuristic
    return MiniMaxResult(best_move, max_value if is_maximizing else min_value)


def generate_move_minimax(
        board: Board, player: Player, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Generates a move for player on the current board.
    Optionally consumes and updates a saved state.
    :param board: The board state
    :param player: The player to generate a move for
    :param saved_state: (optional) Saved state
    :return: Returns the calculated move and the new saved state
    """

    # first, check if a winning move is possible. This can fix a problem where MiniMax
    # would not finish the game since all moves will have
    for move in board.actions():
        nb = board.drop_piece_copy(move)
        if nb.check_winner(player):
            return move, saved_state

    # retrieve our MiniMaxCache from saved state if possible, otherwise create the cache object
    cache = saved_state if type(saved_state) == MiniMaxCache2 else MiniMaxCache2()

    # calculate the MiniMax result
    minimax_result = mini_max(
        board,
        alpha=-np.inf,
        beta=np.inf,
        depth=3,
        max_player=player,
        cache=cache
    )

    print(f"generate_move_minimax() returned move {minimax_result.best_move}, h={minimax_result.value}")
    return minimax_result.best_move, cache

