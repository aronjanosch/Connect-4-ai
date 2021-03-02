import collections
import math

import torch
import torch.multiprocessing as tmp
import os
import datetime
from typing import Optional, List, Dict

import numpy as np

from .alpha_zero_args import AlphaZeroArgs

from . import util
from common.board import Board, Column
from common.players import PLAYER_1, PLAYER_2
from .neural_net import Connect4Network
from .timer import timer


class MCNode(object):
    parent: Optional['MCNode']
    board: Board
    move: Column
    is_expanded: bool

    children: Dict[Column, 'MCNode']
    child_priorities: np.ndarray
    child_total_value: np.ndarray
    child_number_visits: np.ndarray

    action_indices: List[Column]

    def __init__(self, board: Board, move: Optional[Column], parent: Optional['MCNode'] = None):
        self.parent = parent
        self.board = board
        self.move = move

        self.is_expanded = False

        self.children = {}
        self.child_priorities = np.zeros([board.shape.columns], dtype=np.float32)
        self.child_total_value = np.zeros([board.shape.columns], dtype=np.float32)
        self.child_number_visits = np.zeros([board.shape.columns], dtype=np.float32)
        self.action_indices = []

    def get_total_value(self):
        return self.parent.child_total_value[self.move]

    def set_total_value(self, value: np.float):
        self.parent.child_total_value[self.move] = value

    def select_leaf(self) -> 'MCNode':
        """
        Selecting the Leaf of given MCNode
        @return: Leaf Node
        """
        leaf = self
        while leaf.is_expanded:
            best_move = leaf.best_child()
            leaf = leaf.add_child(best_move)

        return leaf

    def best_child(self) -> Column:
        """
        Selects best Child based in Q Value and UCB
        @return: bestmove
        """
        if len(self.action_indices):
            best_move_q = self.child_Q() + self.child_U()
            best_move = self.action_indices[np.argmax(best_move_q[self.action_indices])]
        else:
            best_move = np.argmax(self.child_Q() + self.child_U())

        return best_move

    def add_child(self, move: Column):
        """
        Generates a new Child for current board by playing new move
        @param move: Column index
        @return: Child
        """
        if move not in self.children:
            board_copy = self.board.copy()
            board_copy.drop_piece(move)
            self.children[move] = MCNode(
                board=board_copy,
                move=move,
                parent=self
            )

        return self.children[move]

    def number_visits(self) -> np.float32:
        return self.parent.child_number_visits[self.move]

    def set_number_visits(self, value: np.float32):
        self.parent.child_number_visits[self.move] = value

    def child_Q(self) -> np.ndarray:
        # Q Value
        return self.child_total_value / (1 + self.child_number_visits)

    def child_U(self) -> np.ndarray:
        # UCB Value
        return math.sqrt(self.number_visits()) * (abs(self.child_priorities) / (1 + self.child_number_visits))

    def expand(self, children_priorities: np.array):
        """
        Expands nodes that result from valid moves
        Invalid moves will be dealt by setting the probabilities to play that invalid column to 0
        @param children_priorities:
        @return:
        """
        self.is_expanded = True

        action_indices = self.board.actions()

        if len(action_indices) == 0:
            self.is_expanded = False

        self.action_indices = action_indices

        for i in range(len(children_priorities)):
            if i not in action_indices:
                # mask all illegal actions as zero
                children_priorities[i] = 0.0

        if self.parent.parent is None:
            # add dirichlet noise
            children_priorities = self.add_dirichlet_noise(action_indices, children_priorities)

        self.child_priorities = children_priorities

    def add_dirichlet_noise(self, action_indices: List[Column], children_priorities: np.ndarray) -> np.ndarray:
        """
        Adds dirchlet noise for more exploration -> Research ws suggesting this is necessary
        @param action_indices:
        @param children_priorities:
        @return:
        """
        valid_children_priorities = children_priorities[action_indices]

        dchl = np.random.dirichlet(np.zeros([len(valid_children_priorities)], dtype=np.float32) + 192)

        valid_children_priorities = 0.75 * valid_children_priorities + 0.25 * dchl
        children_priorities[action_indices] = valid_children_priorities
        return children_priorities

    def backup(self, estimated_value: np.float):
        """
        Backpropagation the value to all parent nodes based on who is currently playing
        @param estimated_value: Value of Leaf Node
        """
        current = self
        while current.parent is not None:
            current.set_number_visits(current.number_visits() + 1)
            if current.board.player == PLAYER_1:
                # positive for Player_1
                current.set_total_value(current.get_total_value() + estimated_value)
            elif current.board.player == PLAYER_2:
                # negative for Player_2
                current.set_total_value(current.get_total_value() - estimated_value)

            current = current.parent


class DummyNode(MCNode):
    def __init__(self):
        # super().__init__(None, None)
        super(object, self).__init__()
        self.parent = None

        # use defaultdict so all entries are 0.0 by default
        self.child_total_value = collections.defaultdict(float)
        self.child_number_visits = collections.defaultdict(float)


def get_policy(root: MCNode, temp: np.float = 1.0) -> np.array:
    """
    Formula to get the policy for a given Node
    @param root: MCNode
    @param temp: Decides weight of number_visits
    @return: The policy which Column to play
    """
    return (root.child_number_visits ** (1 / temp)) \
           / sum(root.child_number_visits ** (1 / temp))


@timer
def UCT_search(board: Board, num_reads: np.int, net: Connect4Network):
    """
    Upper Confidence Tree Search utilizing the Neural Net, wich gets fed a encoded board state
    @param board: board state
    @param num_reads: iterations of the UCT
    @param net: NeuralNet
    @return: Node
    """
    root = MCNode(board, move=None, parent=DummyNode())
    # number of iterations for the UCT
    for i in range(num_reads):
        leaf = root.select_leaf()

        # encodes board
        encoded_state = leaf.board.encode() \
            .transpose(2, 0, 1)

        # Torch from encoded state
        encoded_state_torch = torch.from_numpy(encoded_state).float()

        # Check for CUDA availability
        if torch.cuda.is_available():
            encoded_state_torch = encoded_state_torch.cuda()

        # Usage of the NN on the Torch
        child_priorities, value_estimation = net(encoded_state_torch)

        child_priorities = child_priorities.detach().cpu().numpy().reshape(-1)
        value_estimation = value_estimation.item()

        # Check for a winner, Backpropagation if so, skip expansion
        if leaf.board.check_winner() or not leaf.board.is_playable():
            leaf.backup(value_estimation)
            continue

        # Expand and backpropagate
        leaf.expand(child_priorities)
        leaf.backup(value_estimation)

    return root


@timer
def generate_move(net: Connect4Network, current_board: Board, args: AlphaZeroArgs) -> Column:
    """
    Generates move with low randomness of given Board.
    Used for actual playing
    @param net: Neural Network Object
    @param current_board: current board state
    @param args: AlphaZeroArgs
    @return: Column index
    """
    if current_board.check_winner() or current_board.is_final():
        raise AssertionError("Game is already over.")

    # Low temperature
    t = 0.1

    # Run a UCT_search for current board
    root = UCT_search(current_board, args.num_reads_mcts, net)

    policy = get_policy(root, t)
    print(f"Policy: {policy}")

    # Determines move from policy
    move = np.random.choice(
        np.array([0, 1, 2, 3, 4, 5, 6]),
        p=policy
    )
    print(f"Playing col: {move}")
    return move


@timer
def self_play(net: Connect4Network, start_index: np.int, cpu_index: np.int, num_games: np.int, args: AlphaZeroArgs,
              iteration: np.int):
    """
    Self Play of AlphaZero, generating and saving Datasets for the training of the Neural Network
    @param net:
    @param start_index: Start index of Self Play games
    @param cpu_index:
    @param num_games:
    @param args:
    @param iteration: current Iteration
    """

    # number of more random moves, before lowering temp
    n_max_moves = 11

    print(f"CPU={cpu_index}: Starting MCTS")
    iteration_dir = f"./datasets/iter_{iteration}"

    if not os.path.isdir(iteration_dir):
        os.makedirs(iteration_dir)

    # Play self play games
    for idx in range(start_index, num_games + start_index):
        print(f"Game {idx}")

        current_board = Board()
        game_won = False  # indicates that a game is won

        dataset = []
        states = []
        value = 0
        move_count = 0

        while not game_won and current_board.is_playable():
            t = 0.1
            # less random further into the game
            if move_count < n_max_moves:
                t = args.temperature_mcts

            # save current board state (encoded and unencoded)
            states.append(current_board.current_board.copy())
            board_state = current_board.encode().copy()

            root = UCT_search(current_board, args.num_reads_mcts, net)

            policy = get_policy(root, t)
            print(f"Game {idx} policy: {policy}")

            col_choice = np.random.choice(
                np.array([0, 1, 2, 3, 4, 5, 6]),
                p=policy
            )

            current_board.drop_piece(col_choice)  # move piece

            dataset.append([board_state, policy])
            print(f"[Iteration: {iteration}]: Game {idx} CURRENT BOARD:\n", current_board)

            move_count += 1
            if current_board.check_winner():  # if somebody won
                if current_board.player == PLAYER_1:  # black wins
                    print("Black wins")
                    value = -1
                elif current_board.player == PLAYER_2:  # white wins
                    print("White wins")
                    value = 1
                game_won = True

        dataset_p = []

        for idx, data in enumerate(dataset):
            s, p = data
            if idx == 0:
                dataset_p.append([s, p, 0])
            else:
                dataset_p.append([s, p, value])

        # Save the dataset
        time_string = datetime.datetime.today().strftime("%Y-%m-%d")
        pickle_file = f"iter_{iteration}/dataset_iter{iteration}_cpu{cpu_index}_{idx}_{time_string}"
        util.pickle_save(pickle_file, dataset_p)


@timer
def runMonteCarloTreeSearch(args: AlphaZeroArgs, start_index: np.int, iteration: np.int):
    """
    Function handling the MCTS. Loading NN model, starting multicore self-play
    @param args:
    @param start_index: starting index of games
    @param iteration: current iteration
    """
    net = Connect4Network()

    # use CUDA if available
    if torch.cuda.is_available():
        net.cuda()

    print("Preparing MCTS model")
    net.eval()

    # Start multiprocessing
    if args.mcts_num_processes > 1:
        print("multicore MCTS")
        # Configure Pytorch for multiprocessing
        tmp.set_start_method("spawn", force=True)
        net.share_memory()
        net.eval()

        # Get current Model
        current_net_file = util.get_model_file_path(args.neural_net_name, iteration)

        if os.path.isfile(current_net_file):
            # Loads model of current iteration
            checkpoint = torch.load(current_net_file)
            net.load_state_dict(checkpoint["state_dict"])
            print(f"Model loaded from {os.path.abspath(current_net_file)}")
        else:
            # Initialize Model with random weights
            util.create_model_directory()
            torch.save({
                "state_dict": net.state_dict()
            }, current_net_file)
            print(f"Model intialization done at {os.path.abspath(current_net_file)}")

        process_list = []

        # Sets max number of CPUs availible
        if args.mcts_num_processes > tmp.cpu_count():
            num_processes = tmp.cpu_count()
            print(f"Number of processes bigger than number of CPUs! Setting mcts_num_processes to {num_processes}")
        else:
            num_processes = args.mcts_num_processes

        print(f"Spawning {num_processes} processes...")
        with torch.no_grad():
            for i in range(num_processes):
                # defines which Process to run with given Arguments
                process = tmp.Process(target=self_play,
                                      args=(net, start_index, i, args.num_games_per_mcts_process, args, iteration))
                process.start()
                process_list.append(process)
            for p in process_list:
                p.join()
        print("Multicore MCTS finished!")

    # Single Core processing
    elif args.mcts_num_processes == 1:
        current_net_file = util.get_model_file_path(args.neural_net_name, iteration)

        if os.path.isfile(current_net_file):
            checkpoint = torch.load(current_net_file)
            net.load_state_dict(checkpoint["state_dict"])
            print(f"Model loaded from {os.path.abspath(current_net_file)}")
        else:  # initialize model
            util.create_model_directory()

            torch.save({
                "state_dict": net.state_dict()
            }, current_net_file)
            print(f"Model intialization done at {os.path.abspath(current_net_file)}")

        with torch.no_grad():
            self_play(net, start_index=start_index, cpu_index=0, num_games=args.num_games_per_mcts_process, args=args,
                      iteration=iteration)

        print("MCTS done.")
