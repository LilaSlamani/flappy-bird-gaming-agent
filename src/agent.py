# Politique epsilon-greedy (choix d'action) et fonction de perte du DQN
# (équation de Bellman). Utilisé par train.py.

import random
import torch
import torch.nn.functional as F

from network import N_ACTIONS


def select_action(policy_net, obs, epsilon):
    # exploration : action au hasard avec probabilité epsilon
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)

    # exploitation : la meilleure action selon le réseau
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        q_values = policy_net(obs_tensor)
        return int(torch.argmax(q_values, dim=1).item())


def compute_loss(policy_net, target_net, batch, gamma):
    states, actions, rewards, next_states, dones = batch

    # Q-value prédite par le réseau principal pour l'action réellement prise
    q_values = policy_net(states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # cible calculée avec le réseau cible (target network), pas le réseau principal
    with torch.no_grad():
        next_q_values = target_net(next_states)
        next_q_value = next_q_values.max(dim=1).values
        target = rewards + gamma * next_q_value * (1 - dones)

    return F.smooth_l1_loss(q_value, target)
