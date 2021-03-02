from agents.agent_alphazero.monte_carlo import runMonteCarloTreeSearch

from agents.agent_alphazero.alpha_zero_args import AlphaZeroArgs
from agents.agent_alphazero.evaluate import evaluate_nets
from agents.agent_alphazero.train import train_connect4

if __name__ == "__main__":
    # change values of arguments as you like
    arguments = AlphaZeroArgs()
    current_iteration = arguments.iteration # set to resume from a previous iteration

    for i in range(current_iteration, arguments.total_iterations):
        # Starting with MCTS
        runMonteCarloTreeSearch(arguments, start_index=0, iteration=i)
        train_connect4(
            arguments,
            iteration=i,
            new_optim_state=True
        )

        if i >= 1:  # start after first iteration
            winner = evaluate_nets(arguments, i, i + 1)
            counts = 0

            while winner != i + 1:
                print("Trained net didn't perform better, generating more MCTS games for retraining...")
                # generate more datasets using MCTS
                mcs_start_index = (counts + 1) * arguments.num_games_per_mcts_process
                runMonteCarloTreeSearch(arguments, start_index=mcs_start_index, iteration=i)
                counts += 1
                train_connect4(arguments, iteration=i, new_optim_state=True)
                winner = evaluate_nets(arguments, i, i + 1)
