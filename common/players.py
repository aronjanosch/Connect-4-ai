import numpy as np

# No player.
PLAYER_NONE: int = 0

# Player 1
PLAYER_1: int = 1

# Player 2
PLAYER_2: int = 2

# Player type
Player = int


def other_player(player: Player):
    """
    Returns the opponent of player of PLAYER_NONE if player is PLAYER_NONE

    :param player: The player to find the opponent for
    :return: The opponent player
    """
    return PLAYER_2 if player == PLAYER_1 else PLAYER_1
