# Programming Project in Python

![Linting and Testing](https://github.com/programming-project-in-python/programming-project-in-python/workflows/Linting%20and%20Testing/badge.svg)

### Group: Aron & Jonathan

The goal of this project was to develop two agents playing connect four: a minimax agent with alpha/beta pruning abilities and an advanced agent.<br>
For the advanced agent, we chose a reinforcement learning approach using the basics of [AlphaZero](https://de.wikipedia.org/wiki/AlphaZero).

The following table lists the directories contained in this repository as well as a short description for each.

| Directory              | Description                                                                              |
|------------------------|------------------------------------------------------------------------------------------|
| .github/workflows      | GitHub Workflow configuration used on each push to the `master` branch to run our tests. |
| agents/agent_alphazero | Our AlphaZero agent                                                                      |
| agents/agent_minimax   | Our MiniMax agent together with utility functions and the heuristic we used              |
| agents/agent_random    | Our Random agent                                                                         |
| alphazero_pipeline     | Training pipeline for the AlphaZero agent                                                |
| common                 | Common functionality used across the project                                             |

## Usage

The agents should work from Python versions 3.6 up to 3.9. Requirements are listed in `requirements.txt` and can easily be installed:

```sh
pip install -r requirements.txt
```

The script `main.py` can be executed to let different agents play against each other or to interactively challenge an agent.
