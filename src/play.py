# Recharge un agent entraîné et sauvegardé depuis un script neuf, pour
# vérifier que la sauvegarde fonctionne vraiment (voir carnet_essais.md,
# essai 1 : ce script a permis de détecter un vrai bug de sauvegarde).

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import flappy_bird_gymnasium
import gymnasium as gym

from network import QNetwork

CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"


def play_episode(env, policy_net, render=False):
    obs, info = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        # aucune exploration ici, on joue toujours la meilleure action connue
        # (politique gloutonne / greedy)
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = int(torch.argmax(policy_net(obs_tensor), dim=1).item())

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

        if render:
            env.render()
            time.sleep(1 / 30)

    return info["score"], total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best_agent_run1.pt")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    checkpoint_path = CHECKPOINTS_DIR / args.checkpoint

    # réseau reconstruit à vide, poids chargés depuis le fichier sauvegardé
    # c'est ce qui prouve que la sauvegarde fonctionne vraiment, pas juste
    # que le modèle entraîné en mémoire donne un bon score
    policy_net = QNetwork()
    policy_net.load_state_dict(torch.load(checkpoint_path))
    policy_net.eval()

    render_mode = "human" if args.render else None
    env = gym.make("FlappyBird-v0", use_lidar=False, render_mode=render_mode)

    scores = []
    for episode in range(args.episodes):
        score, total_reward = play_episode(env, policy_net, render=args.render)
        scores.append(score)
        print(f"partie {episode} : score {score}")

    env.close()
    print(f"Score moyen sur {args.episodes} parties : {np.mean(scores):.2f}")
    print(f"Checkpoint charge : {checkpoint_path}")


if __name__ == "__main__":
    main()
