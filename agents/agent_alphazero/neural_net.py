import os
from typing import NamedTuple

import numpy as np
import torch

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from agents.agent_alphazero import util
from agents.agent_alphazero.alpha_zero_args import AlphaZeroArgs


class ConvolutionalBlock(nn.Module):
    action_size: int
    conv1: nn.Conv2d
    bn1: nn.BatchNorm2d

    def __init__(self):
        super(ConvolutionalBlock, self).__init__()
        self.action_size = 7
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=128)

    def forward(self, s: Tensor):
        s = s.view(-1, 3, 6, 7)
        s = F.relu(self.bn1(self.conv1(s)))
        return s


class ResidualBlock(nn.Module):
    conv1: nn.Conv2d
    bn1: nn.BatchNorm2d
    bn2: nn.BatchNorm2d

    def __init__(self, inplanes=128, planes=128, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        result: Tensor = F.relu(self.bn1(self.conv1(x)))
        result = self.bn2(self.conv2(result))
        result += residual

        result = F.relu(result)

        return result


class OutputBlockForward(NamedTuple):
    tensor1: Tensor
    tensor2: Tensor


class OutputBlock(nn.Module):
    conv: nn.Conv2d
    conv1: nn.Conv2d

    bn: nn.BatchNorm2d
    bn1: nn.BatchNorm2d

    fc: nn.Linear
    fc1: nn.Linear
    fc2: nn.Linear

    logsoftmax: nn.LogSoftmax

    def __init__(self):
        super(OutputBlock, self).__init__()
        self.conv = nn.Conv2d(128, 3, kernel_size=1)  # value head
        self.bn = nn.BatchNorm2d(3)
        self.fc1 = nn.Linear(3 * 6 * 7, 32)
        self.fc2 = nn.Linear(32, 1)

        self.conv1 = nn.Conv2d(128, 32, kernel_size=1)  # policy head
        self.bn1 = nn.BatchNorm2d(32)

        # apply log(Softmax(x)) to tensor
        self.logsoftmax = nn.LogSoftmax(dim=1)

        # apply linear transformation
        self.fc = nn.Linear(6 * 7 * 32, 7)

    def forward(self, s) -> OutputBlockForward:
        v = F.relu(self.bn(self.conv(s)))  # value head
        v = v.view(-1, 3 * 6 * 7)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s)))  # policy head
        p = p.view(-1, 6 * 7 * 32)
        p = self.fc(p)
        p = self.logsoftmax(p).exp()
        return OutputBlockForward(p, v)


class Connect4Network(nn.Module):
    conv: ConvolutionalBlock
    output_block: OutputBlock

    def __init__(self):
        super(Connect4Network, self).__init__()
        self.conv = ConvolutionalBlock()

        self.res_blocks = nn.Sequential(*(ResidualBlock() for _ in range(19)))
        self.output_block = OutputBlock()

    def forward(self, s: Tensor) -> Tensor:
        s = self.conv(s)
        for block in range(19):
            s = self.res_blocks[block](s)

        s = self.output_block(s)
        return s


def load_connect4network(arguments: AlphaZeroArgs, iteration: int = 0) -> Connect4Network:
    current_net_file = util.get_model_file_path(arguments.neural_net_name, iteration)

    net = Connect4Network()

    # use CUDA if available
    if torch.cuda.is_available():
        net.cuda()

    # use single thread processing for now
    print("Preparing MCTS model")
    net.eval()

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

    return net


class AlphaLoss(nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum(
            (-policy * (1e-8 + y_policy.float()).float().log()),
            1
        )

        total_error = (value_error.view(-1).float() + policy_error).mean()
        return total_error


class BoardData(torch.utils.data.Dataset):
    def __init__(self, dataset: np.ndarray):  # dataset = (s, p, v)
        self.X = dataset[:, 0]
        self.y_p, self.y_v = dataset[:, 1], dataset[:, 2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return np.int64(self.X[idx].transpose(2, 0, 1)), self.y_p[idx], self.y_v[idx]
