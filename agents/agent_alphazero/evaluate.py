import os
from typing import NamedTuple, Optional, Tuple, List

import torch
import torch.multiprocessing as tmp
import numpy as np

from common.board import Board
from . import util
from .alpha_zero_args import AlphaZeroArgs
from .neural_net import Connect4Network

# from connect_board import board as cboard
# import encoder_decoder_c4 as ed
import copy
from .monte_carlo import UCT_search, get_policy
import datetime

from common.players import PLAYER_1, PLAYER_2
from .timer import timer


class ArenaResult(NamedTuple):
    best_win_ratio: float
    num_games: int


class Arena(object):
    current: Connect4Network
    best: Connect4Network

    def __init__(self, current_cnet: Connect4Network, best_cnet: Connect4Network):
        self.current = current_cnet
        self.best = best_cnet

    @timer
    def play_round(self, num_reads: int) -> Tuple[Optional[str], List[np.ndarray]]:
        """
        Evaluate the trained network by playing matches between the current and the previous NN
        @param num_reads: see args
        """
        print("Starting game round...")
        # randomly choose starting player
        if np.random.uniform(0, 1) <= 0.5:
            white = self.current
            black = self.best
            w = "current"
            b = "best"
        else:
            white = self.best
            black = self.current
            w = "best"
            b = "current"

        # initializing
        current_board = Board()
        game_won = False
        dataset = []
        value = 0
        temperature = 0.1       # exploration vs exploitation factor (smaller -> more exploitation)

        while not game_won and current_board.is_playable():
            dataset.append(copy.deepcopy(current_board.encode()))
            # get Policy
            if current_board.player == PLAYER_1:
                root = UCT_search(current_board, num_reads, white)
                policy = get_policy(root, temperature)
                print("Policy: ", policy, "white = %s" % (str(w)))
            elif current_board.player == PLAYER_2:
                root = UCT_search(current_board, num_reads, black)
                policy = get_policy(root, temperature)
                print("Policy: ", policy, "black = %s" % (str(b)))
            else:
                raise AssertionError("Invalid player.")
            # Chose a Column with given policy
            col_choice = np.random.choice(
                np.array([0, 1, 2, 3, 4, 5, 6]),
                p=policy
            )

            current_board.drop_piece(col_choice)  # move piece
            print(current_board)
            if current_board.check_winner():  # someone wins
                if current_board.player == PLAYER_1:  # black wins
                    value = -1
                elif current_board.player == PLAYER_2:  # white wins
                    value = 1
                game_won = True
        # Append new board to the dataset encoded in one-hot-encoding manner
        dataset.append(current_board.encode())
        if value == -1:
            dataset.append(f"{b} as black wins")
            return b, dataset
        elif value == 1:
            dataset.append(f"{w} as white wins")
            return w, dataset
        else:
            dataset.append("Nobody wins")
            return None, dataset

    @timer
    def evaluate(self, num_games: np.int, num_reads: int, cpu: np.int = 0) -> ArenaResult:
        """
        determines the better NN by win-ratio and saves the results
        """
        current_wins = 0
        print(f"CPU={cpu}: Starting games...")
        for i in range(num_games):
            with torch.no_grad():
                winner, dataset = self.play_round(num_reads)
                print("%s wins!" % winner)
            if winner == "current":
                current_wins += 1
            date_string = datetime.datetime.today().strftime("%Y-%m-%d")
            dest_name = f"evaluate_net_dataset_cpu{cpu}_{i}_{date_string}_{winner}"
            util.pickle_save(dest_name, dataset)
            print(f"Current_net wins ratio: {(current_wins / num_games):.5f}")

        print(f"Current_net wins ratio: {(current_wins / num_games):.5f}")

        evaluate_result = ArenaResult(
            best_win_ratio=current_wins / num_games,
            num_games=num_games
        )

        util.pickle_save(f"wins_cpu_{cpu}", evaluate_result)
        print(f"[CPU {cpu}]: Finished arena games!")


def f_process(arena: Arena, num_games: np.int, num_reads, cpu):
    """
    Helper functions for multiprocessing
    @param arena: Arena
    @param num_games: Number of MCTS games
    @param num_reads: Number of MCTS reads
    @param cpu: index of CPU
    """
    arena.evaluate(num_games=num_games, num_reads=num_reads, cpu=cpu)

@timer
def evaluate_nets(arguments: AlphaZeroArgs, iteration_1: np.int, iteration_2: np.int) -> np.int:
    """
    Function handling the whole evaluation process, Returning the better performing NN
    @param arguments: AlphaZeroArgs
    @param iteration_1: index of the previous iteration
    @param iteration_2: index of the current iteration
    @return: Neural Network
    """
    print("Loading nets...")
    current_net_filename = util.get_model_file_path(arguments.neural_net_name, iteration_2)
    best_net_filename = util.get_model_file_path(arguments.neural_net_name, iteration_1)

    print(f"Current net: {current_net_filename}")
    print(f"Previous (Best) net: {best_net_filename}")

    current_cnet = Connect4Network()
    best_cnet = Connect4Network()

    # Checks for CUDA availability
    cuda = torch.cuda.is_available()
    if cuda:
        current_cnet.cuda()
        best_cnet.cuda()

    if not os.path.isdir("./evaluator_data/"):
        os.mkdir("evaluator_data")

    # Starts multiple threads for evaluation
    if arguments.mcts_num_processes > 1:
        # initializing the pytorch multiprocessing
        tmp.set_start_method("spawn", force=True)
        # configure NN
        current_cnet.share_memory()
        current_cnet.eval()

        # Load the model
        checkpoint = torch.load(current_net_filename)
        current_cnet.load_state_dict(checkpoint["state_dict"])
        checkpoint = torch.load(best_net_filename)
        best_cnet.load_state_dict(checkpoint["state_dict"])

        process_list = []

        # Sets to max number of CPUs available
        if arguments.mcts_num_processes > tmp.cpu_count():
            num_processes = tmp.cpu_count()
            print(f"Number of processes bigger than number of CPUs! Setting mcts_num_processes to {num_processes}")
        else:
            num_processes = arguments.mcts_num_processes

        print(f"Spawning {num_processes} processes...")
        with torch.no_grad():
            for i in range(num_processes):
                # defines which Process to run with given Arguments
                process = tmp.Process(target=f_process,
                                      args=(Arena(current_cnet, best_cnet), arguments.num_evaluator_games, arguments.num_reads_mcts, i))
                process.start()
                process_list.append(process)
            for p in process_list:
                p.join()

        wins_ratio = 0.0
        # Loads Results and determines the better performing NN
        for i in range(num_processes):
            arena_results = util.pickle_load(f"wins_cpu_{i}")
            wins_ratio = arena_results.best_win_ratio
        if wins_ratio >= 0.55:
            return iteration_2
        else:
            return iteration_1
    # Single Core Evaluation
    elif arguments.mcts_num_processes == 1:
        current_cnet.eval()
        best_cnet.eval()

        checkpoint = torch.load(current_net_filename)
        current_cnet.load_state_dict(checkpoint["state_dict"])
        checkpoint = torch.load(best_net_filename)
        best_cnet.load_state_dict(checkpoint["state_dict"])

        arena = Arena(
            current_cnet=current_cnet,
            best_cnet=best_cnet
        )

        arena_result = util.pickle_load(f"wins_cpu_0")

        if arena_result.best_win_ratio >= 0.55:
            return iteration_2
        else:
            return iteration_1
