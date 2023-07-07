import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, obs_space, action_space, embedding_dim=64):
        super(DQN, self).__init__()

        self.embed = nn.Embedding(obs_space, embedding_dim)
        self.fully_connected = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_space),
        )

    def forward(self, x):
        x = self.embedding_layer(x.long())
        return self.fully_connected(x)


dqn = DQN(16, 4)
