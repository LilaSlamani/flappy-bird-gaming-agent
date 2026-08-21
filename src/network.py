# Réseau de neurones du DQN : donne une observation, sort un Q-value par
# action possible. Utilisé par train.py (entraînement) et play.py
# (rechargement).

import torch
import torch.nn as nn

OBS_SIZE = 12  # taille du vecteur d'observation de FlappyBird-v0
N_ACTIONS = 2  # ne rien faire / battre des ailes


class QNetwork(nn.Module):
    def __init__(self, obs_size=OBS_SIZE, n_actions=N_ACTIONS, hidden_size=128):
        super().__init__()
        # feedforward simple : observation -> 2 couches cachées -> Q-value par action
        self.layers = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        # x : batch d'observations, shape (batch_size, obs_size)
        # sortie : shape (batch_size, n_actions), un Q-value par action possible
        return self.layers(x)
