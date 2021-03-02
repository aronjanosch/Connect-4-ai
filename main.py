import numpy as np
import time
from typing import Optional, Callable, Tuple

from agents.agent_random import generate_move as gm_rnd
from agents.agent_minimax import generate_move as gm_mm
from agents.agent_alphazero import generate_move as gm_a0
from common.base import SavedState, PlayerAction
from common.players import Player

from common.board import Board


GenMove = Callable[
    [np.ndarray, Player, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


def user_move_console(board: Board, _player: Player, saved_state: Optional[SavedState]):
    action = PlayerAction(-1)
    while not 0 <= action < board.shape[1]:
        try:
            action = PlayerAction(input("Column? "))
        except Exception as e:
            print(e)
            pass
    return action, saved_state


def agent_vs_agent(
    generate_move_1: GenMove,
    generate_move_2: GenMove,
    player_1: str = "Player 1",
    player_2: str = "Player 2",
    args_1: tuple = (),
    args_2: tuple = (),
    init_1: Callable = lambda board, player: None,
    init_2: Callable = lambda board, player: None
):
    from agents.common import PLAYER1, PLAYER2, GameState
    from agents.common import initialize_game_state, pretty_print_board, apply_player_action, check_end_state

    players = (PLAYER1, PLAYER2)
    for play_first in (1, -1):
        for init, player in zip((init_1, init_2)[::play_first], players):
            init(initialize_game_state(), player)

        saved_state = {PLAYER1: None, PLAYER2: None}
        board = initialize_game_state()
        gen_moves = (generate_move_1, generate_move_2)[::play_first]
        player_names = (player_1, player_2)[::play_first]
        gen_args = (args_1, args_2)[::play_first]

        playing = True
        while playing:
            for player, player_name, gen_move, args in zip(
                players, player_names, gen_moves, gen_args,
            ):
                t0 = time.time()
                print(pretty_print_board(board))

                c_board = Board(board.copy(), player)

                action, saved_state[player] = gen_move(
                    c_board, player, saved_state[player], *args
                )
                print(f"Move time: {time.time() - t0:.3f}s")

                apply_player_action(board, action, player)

                end_state = check_end_state(board, player)
                if end_state != GameState.STILL_PLAYING:
                    print(pretty_print_board(board))
                    if end_state == GameState.IS_DRAW:
                        print("Game ended in draw")
                    else:
                        print(
                            f'{player_name} won playing {"X" if player == PLAYER1 else "O"}'
                        )

                    time.sleep(4)

                    playing = False


if __name__ == "__main__":
    while True:
        try:
            print("How do you want to play?")
            print("a: Player vs AI")
            print("b: AI vs AI")
            play_mode = input("Your choice:")
            if play_mode == "a":
                print("Player vs AI")
                print("Which Agent you want to play against?")
                print("a: AlphaZero")
                print("b: Minimax")
                enemy = input("Your Choice:")
                if enemy == "a":
                    agent_vs_agent(user_move_console, gm_a0)
                elif enemy == "b":
                    agent_vs_agent(user_move_console, gm_mm)
                else:
                    print("Wrong input")
            elif play_mode == "b":
                print("AI vs AI")
                print("Which Agents should play?")
                print("a: AlphaZero vs Minimax")
                print("b: AlphaZero vs Random")
                print("c: Minimax vs Random")
                enemy = input("Your Choice:")
                if enemy == "a":
                    agent_vs_agent(gm_a0, gm_mm)
                elif enemy == "b":
                    agent_vs_agent(gm_a0, gm_rnd)
                elif enemy == "c":
                    agent_vs_agent(gm_mm, gm_rnd)
                else:
                    print("Wrong input")
            else:
                print("Wrong input")
        except Exception as ex:
            print(f"Error: {ex}")

        print("Play again? (y/n)")
        i = input("Your Choice:")
        if i == "y":
            print("Restart!")
        elif i == "n":
            break
        else:
            print("Wrong input, restarting")