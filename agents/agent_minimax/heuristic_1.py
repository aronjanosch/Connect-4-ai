"""
This heuristic is based on the following paper: https://www.researchgate.net/publication/331552609_Research_on_Different_Heuristics_for_Minimax_Algorithm_Insight_from_Connect-4_Game
The basic idea is to separate a board state into four different feature levels, each handling a number of
connected stones and assigning scores with different orders of magnitude to them. These feature values are then
added together to form the heuristic value.
"""

import numpy as np
from typing import Optional, Tuple

from common.base import SavedState
from common.board import Board
from common.players import Player, other_player

from . import heuristic_helper
from .base import Connect4Heuristics


def feature1(board: Board, player: Player) -> float:
    """
    Feature 1 checks for four connected stones (win).

    :param board: The board to check against
    :param player: The player to check for
    :return: The heuristic for feature 1
    """
    if board.check_winner(player):
        return np.inf
    else:
        return float(0)


def feature2(board: Board, player: Player) -> float:
    """
    Feature 2 checks for three connected stones (either consecutive or with one gap in between).

    :param board: The board to check against
    :param player: The player to check for
    :return: The heuristic for feature 2
    """
    positions = heuristic_helper.get_player_positions(board.current_board, player)
    beams = positions.get_beams()
    beams3 = [b for b in beams if b.length == 3]
    beams3_gap = [
        b for b in [b.extend_to_four_with_gap(board.current_board, player) for b in beams if b.length == 2] if b.length == 4
    ]

    feature_value = float(0)

    # Feature 2 - A move can only be made on one of the immediately adjacent columns.
    for b in beams3:
        nb_left = b.neighbour_left()
        nb_right = b.neighbour_right()

        if nb_left.can_be_played(board.current_board) and nb_right.can_be_played(board.current_board):
            # The player’s win is unstoppable. Therefore, this feature will be given Infinity.
            return float(2000000)
        elif nb_left.can_be_played(board.current_board) or nb_right.can_be_played(board.current_board):
            # The red player is probably going to win at (d, 1) in the next move
            # or be stopped by the opponent. Therefore, this feature will be given a lower
            # score, 900,000.
            feature_value += float(900000)
        else:
            # It has no promising future. Therefore, this feature will be given 0.
            pass

    for _ in beams3_gap:
        # The win can be stopped, which is just like the situation above. Therefore, we also give it 900,000.
        feature_value += float(900000)

    return feature_value


def feature3(board: Board, player: Player) -> float:
    """
    Feature 3 checks for two connected stones and the free space next to these.

    :param board: The board to check against
    :param player: The player to check for
    :return: The heuristic for feature 2
    """
    positions = heuristic_helper.get_player_positions(board.current_board, player)
    beams = positions.get_beams()
    beams2 = [b for b in beams if b.length == 2]

    points_table = {
        2: 10000,
        3: 20000,
        4: 30000,
        5: 40000
    }

    feature_value = float(0)

    for b in beams2:
        if b.neighbour_left_free(board.current_board) and b.neighbour_left_free(board.current_board):
            # A move can be made on either immediately adjacent columns
            feature_value += float(50000)
        else:
            cnt_playable_left = b.neighbours_left_playable_count(board.current_board)
            cnt_playable_right = b.neighbours_right_playable_count(board.current_board)

            cnt_playable = max(cnt_playable_left, cnt_playable_right)

            if cnt_playable in points_table:
                feature_value += points_table[cnt_playable]

    return feature_value


def feature4(board: Board, player: Player) -> float:
    """
    Feature 4 checks for sole stones which are not connected to any stones of the same player.

    :param board: The board to check against
    :param player: The player to check for
    :return: The heuristic for feature 4
    """
    positions = heuristic_helper.get_player_positions(board.current_board, player)

    feature_value = float(0)

    column_points = {
        0: 40,
        1: 70,
        2: 120,
        3: 200,
        4: 120,
        5: 70,
        6: 40
    }

    for p in positions:
        if p.is_lonely(board.current_board):
            feature_value += column_points[p.column]

    return feature_value


def calculate_heuristic_internal(
    board: Board,
    player: Player,
    saved_state: Optional[SavedState] = None
) -> Connect4Heuristics:
    """
    Actually calculates the heuristic by adding the features together.

    :param board: The board to calculate the heuristic for
    :param player: The player to calculate the heuristic for
    :param saved_state: The saved state (not used here).
    :return: The total heuristic value for the Board and the specified Player.
    """

    f1 = feature1(board, player)
    f2 = feature2(board, player)
    f3 = feature3(board, player)
    f4 = feature4(board, player)

    total_heuristic = f1 + f2 + f3 + f4

    return Connect4Heuristics(total_heuristic, player)


def calculate_heuristic(
    board: Board, max_player: Player, saved_state: Optional[SavedState] = None
) -> Connect4Heuristics:
    """
    Calculates the heuristic for a given board, adjusted to the max_player.
    This function calculates the heuristic for both the maximizing and the minimizing player and substracts these
    value from another.

    :param board: The board to calculate the heuristic for
    :param max_player: The maximizing player
    :param saved_state: The saved state (not used here).
    :return: The calculated heuristic.
    """

    h_max_player = calculate_heuristic_internal(board, max_player, saved_state=saved_state)
    h_min_player = calculate_heuristic_internal(board, other_player(max_player), saved_state=saved_state)

    return Connect4Heuristics(h_max_player.value - h_min_player.value, max_player)
