from agents import common
from agents.common import PlayerAction, PLAYER1, PLAYER2, NO_PLAYER


class TestCommon:
    def test_connected_four_horizontal(self):
        c4_yes = common.initialize_game_state()
        common.apply_player_action(c4_yes, PlayerAction(0), common.PLAYER1)
        common.apply_player_action(c4_yes, PlayerAction(1), common.PLAYER1)
        common.apply_player_action(c4_yes, PlayerAction(2), common.PLAYER1)
        common.apply_player_action(c4_yes, PlayerAction(3), common.PLAYER1)

        c4_no = common.initialize_game_state()
        common.apply_player_action(c4_no, PlayerAction(0), common.PLAYER1)
        common.apply_player_action(c4_no, PlayerAction(1), common.PLAYER1)
        common.apply_player_action(c4_no, PlayerAction(2), common.PLAYER2)
        common.apply_player_action(c4_no, PlayerAction(3), common.PLAYER1)

        assert common.connected_four(c4_yes, PLAYER1) == True
        assert common.connected_four(c4_yes, PLAYER1, PlayerAction(3)) == True
        assert common.connected_four(c4_no, PLAYER1) == False
        assert common.connected_four(c4_no, PLAYER1, PlayerAction(3)) == False

    def test_connected_four_vertical(self):
        c4_yes = common.initialize_game_state()
        common.apply_player_action(c4_yes, PlayerAction(0), common.PLAYER1)
        common.apply_player_action(c4_yes, PlayerAction(0), common.PLAYER1)
        common.apply_player_action(c4_yes, PlayerAction(0), common.PLAYER1)
        common.apply_player_action(c4_yes, PlayerAction(0), common.PLAYER1)

        c4_no = common.initialize_game_state()
        common.apply_player_action(c4_no, PlayerAction(0), common.PLAYER1)
        common.apply_player_action(c4_no, PlayerAction(0), common.PLAYER1)
        common.apply_player_action(c4_no, PlayerAction(0), common.PLAYER2)
        common.apply_player_action(c4_no, PlayerAction(0), common.PLAYER1)

        assert common.connected_four(c4_yes, PLAYER1) == True
        assert common.connected_four(c4_yes, PLAYER1, PlayerAction(0)) == True
        assert common.connected_four(c4_no, PLAYER1) == False
        assert common.connected_four(c4_no, PLAYER1, PlayerAction(0)) == False

    # def test_connected_four_diagonal(self):
    #     c4_yes = common.initialize_game_state()
    #     common.apply_player_action(c4_yes, PlayerAction(0), common.PLAYER1)
    #     common.apply_player_action(c4_yes, PlayerAction(1), common.PLAYER2)
    #     common.apply_player_action(c4_yes, PlayerAction(1), common.PLAYER1)
    #     common.apply_player_action(c4_yes, PlayerAction(2), common.PLAYER2)
    #     common.apply_player_action(c4_yes, PlayerAction(3), common.PLAYER1)
    #     common.apply_player_action(c4_yes, PlayerAction(3), common.PLAYER2)
    #     common.apply_player_action(c4_yes, PlayerAction(2), common.PLAYER1)
    #     common.apply_player_action(c4_yes, PlayerAction(0), common.PLAYER2)
    #     common.apply_player_action(c4_yes, PlayerAction(0), common.PLAYER1)
    #     common.apply_player_action(c4_yes, PlayerAction(0), common.PLAYER2)
    #
    #     c4_no = common.initialize_game_state()
    #     common.apply_player_action(c4_no, PlayerAction(0), common.PLAYER1)
    #     common.apply_player_action(c4_no, PlayerAction(0), common.PLAYER1)
    #     common.apply_player_action(c4_no, PlayerAction(0), common.PLAYER2)
    #     common.apply_player_action(c4_no, PlayerAction(0), common.PLAYER1)
    #
    #     assert common.connected_four(c4_yes, PLAYER1) == True
    #     assert common.connected_four(c4_yes, PLAYER1, PlayerAction(0)) == True
    #     assert common.connected_four(c4_no, PLAYER1) == False
    #     assert common.connected_four(c4_no, PLAYER1, PlayerAction(0)) == False
