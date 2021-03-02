import unittest
import numpy as np

from common.board import Board
from common.players import PLAYER_1, PLAYER_2

from . import heuristic_1


class TestHeuristic(unittest.TestCase):
    def test_feature1(self):
        # |==============|
        # |              |
        # |              |
        # |              |
        # |              |
        # |O O O         |
        # |X X X X       |
        # |==============|
        # |0 1 2 3 4 5 6 |
        b = Board()
        b.drop_piece(0, PLAYER_1)
        b.drop_piece(0, PLAYER_2)
        b.drop_piece(1, PLAYER_1)
        b.drop_piece(1, PLAYER_2)
        b.drop_piece(2, PLAYER_1)
        b.drop_piece(2, PLAYER_2)
        b.drop_piece(3, PLAYER_1)

        self.assertTrue(b.check_winner(PLAYER_1))
        self.assertFalse(b.check_winner(PLAYER_2))

        self.assertEqual(heuristic_1.feature1(b, PLAYER_1), np.inf)

    def test_feature2(self):
        # |==============|
        # |              |
        # |              |
        # |              |
        # |              |
        # |  O O         |
        # |  X X X O     |
        # |==============|
        # |0 1 2 3 4 5 6 |
        b_3_in_row_1_neighbour = Board()
        b_3_in_row_1_neighbour.drop_piece(1, PLAYER_1)
        b_3_in_row_1_neighbour.drop_piece(1, PLAYER_2)
        b_3_in_row_1_neighbour.drop_piece(2, PLAYER_1)
        b_3_in_row_1_neighbour.drop_piece(2, PLAYER_2)
        b_3_in_row_1_neighbour.drop_piece(3, PLAYER_1)
        b_3_in_row_1_neighbour.drop_piece(4, PLAYER_2)

        self.assertEqual(heuristic_1.feature2(b_3_in_row_1_neighbour, PLAYER_1), float(900000))

        # |==============|
        # |              |
        # |              |
        # |              |
        # |              |
        # |O   O         |
        # |X   X X O     |
        # |==============|
        # |0 1 2 3 4 5 6 |
        b_3_in_row_1_gap = Board()
        b_3_in_row_1_gap.drop_piece(0, PLAYER_1)
        b_3_in_row_1_gap.drop_piece(0, PLAYER_2)
        b_3_in_row_1_gap.drop_piece(2, PLAYER_1)
        b_3_in_row_1_gap.drop_piece(2, PLAYER_2)
        b_3_in_row_1_gap.drop_piece(3, PLAYER_1)
        b_3_in_row_1_gap.drop_piece(4, PLAYER_2)

        self.assertEqual(heuristic_1.feature2(b_3_in_row_1_gap, PLAYER_1), float(900000))

    def test_feature3(self):
        # |==============|
        # |              |
        # |              |
        # |              |
        # |              |
        # |  O O         |
        # |  X X         |
        # |==============|
        # |0 1 2 3 4 5 6 |
        b_2_in_row_both_nbs_free = Board()
        b_2_in_row_both_nbs_free.drop_piece(1, PLAYER_1)
        b_2_in_row_both_nbs_free.drop_piece(1, PLAYER_2)
        b_2_in_row_both_nbs_free.drop_piece(2, PLAYER_1)
        b_2_in_row_both_nbs_free.drop_piece(2, PLAYER_2)

        self.assertEqual(heuristic_1.feature3(b_2_in_row_both_nbs_free, PLAYER_1), float(50000))

        # |==============|
        # |              |
        # |              |
        # |              |
        # |              |
        # |  O           |
        # |O X X         |
        # |==============|
        # |0 1 2 3 4 5 6 |
        b_2_in_row_4_free = Board()
        b_2_in_row_4_free.drop_piece(1, PLAYER_1)
        b_2_in_row_4_free.drop_piece(1, PLAYER_2)
        b_2_in_row_4_free.drop_piece(2, PLAYER_1)
        b_2_in_row_4_free.drop_piece(0, PLAYER_2)

        self.assertEqual(heuristic_1.feature3(b_2_in_row_4_free, PLAYER_1), float(30000))

    def test_feature4(self):
        # |==============|
        # |              |
        # |              |
        # |              |
        # |              |
        # |    O         |
        # |    X         |
        # |==============|
        # |0 1 2 3 4 5 6 |
        b_1_col2 = Board()
        b_1_col2.drop_piece(2, PLAYER_1)
        b_1_col2.drop_piece(2, PLAYER_2)

        self.assertEqual(heuristic_1.feature4(b_1_col2, PLAYER_1), float(120))

        # |==============|
        # |              |
        # |              |
        # |              |
        # |              |
        # |    O     X   |
        # |    X     O   |
        # |==============|
        # |0 1 2 3 4 5 6 |
        b_1_col2_1_col5 = Board()
        b_1_col2_1_col5.drop_piece(2, PLAYER_1)
        b_1_col2_1_col5.drop_piece(5, PLAYER_2)
        b_1_col2_1_col5.drop_piece(5, PLAYER_1)
        b_1_col2_1_col5.drop_piece(2, PLAYER_2)

        self.assertEqual(heuristic_1.feature4(b_1_col2_1_col5, PLAYER_1), float(190))


if __name__ == '__main__':
    unittest.main()
