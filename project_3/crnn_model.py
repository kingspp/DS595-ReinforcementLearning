#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import convolutional_rnn


class CrnnDQN(nn.Module):
    """Initialize a deep Q-learning network

    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

    This is just a hint. You can build your own structure.
    """

    def __init__(self, env):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(CrnnDQN, self).__init__()
        ###########################
        # YOUR IMPLEMENTATION HERE #
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.crnn = convolutional_rnn.Conv2dLSTM(in_channels=64,  # Corresponds to input size
                                     out_channels=4,  # Corresponds to hidden size
                                     kernel_size=3,  # Int or List[int]
                                     num_layers=2,
                                     bidirectional=False,
                                     stride=1, batch_first=True)
        self.fc1 = nn.Linear(9*9*4, 4)
        # self.fc2 = nn.Linear(512, env.action_space.n)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        ###########################
        # YOUR IMPLEMENTATION HERE #
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.crnn(x.view(-1,1,64,9,9).permute(0,4,2,3,1))[0])
        x = self.fc1(x.reshape(x.size(0), -1))


        ###########################
        return x
