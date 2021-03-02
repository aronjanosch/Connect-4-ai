from kaggle_environments import evaluate, make, utils


if __name__ == "__main__":
    env = make("connectx", debug=True)
    env.render()
