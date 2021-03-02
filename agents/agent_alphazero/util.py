import os
import pickle
from typing import Any
import pathlib

from agents.agent_alphazero.timer import timer


@timer
def pickle_save(destination_file: str, data: Any) -> None:
    """
    Save data to file
    @param destination_file:
    @param data:
    @return:
    """
    full_path = os.path.join("./datasets", destination_file)
    p = pathlib.Path(full_path)
    if not os.path.exists(p.parent):
        os.makedirs(p.parent)
    with open(full_path, "wb") as f:
        pickle.dump(data, f)

@timer
def pickle_load(source_file: str) -> Any:
    """
    Load data from file
    @param source_file:
    @return:
    """
    full_path = os.path.join("./datasets", source_file)
    with open(full_path, "rb") as f:
        return pickle.load(f)


def create_model_directory():
    model_dir = get_model_directory()
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)


def get_model_directory() -> str:
    return os.path.join("./model/")


def get_model_file_path(net_name: str, iteration: int) -> str:
    return os.path.join(get_model_directory(), f"{net_name}_iter_{iteration}.pth.tar")

def get_losses_file(iteration: int) -> str:
    return os.path.join("./model", f"losses_per_epoch_iter_{iteration}.pkl")

def get_datasets_dir(iteration: int) -> str:
    return f"./datasets/iter_{iteration}/"