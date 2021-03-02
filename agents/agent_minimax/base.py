from common.players import Player


class Connect4Heuristics:
    """
    A heuristic value class containing the heuristic value as well as the player this heuristic has been calculated for.
    """

    # The heuristic value
    value: float

    # The player this heuristic was calculated for
    perspective: Player

    def __init__(self, value: float, perspective: Player):
        """
        Initialize a heuristic object

        :param value: The heuristic value
        :param perspective: The player whose heuristic value was calculated
        """
        self.value = value
        self.perspective = perspective
