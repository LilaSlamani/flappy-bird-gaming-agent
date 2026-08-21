# Mémoire de replay : stocke les transitions jouées et permet d'en tirer
# des lots au hasard pour l'entraînement, pour casser la corrélation
# entre expériences consécutives. Utilisé par train.py.

import random
from collections import deque
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity=50000):
        # deque à taille fixe : les plus vieilles transitions sont
        # supprimées automatiquement une fois la capacité atteinte
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # tire batch_size transitions au hasard et les convertit en
        # tenseurs PyTorch prêts à être donnés au réseau
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
