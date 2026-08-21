# Agent aléatoire : joue N parties sans aucune logique, sert de score de
# référence pour comparer l'agent entraîné (voir train.py).

import argparse
import csv
import time
from pathlib import Path
import numpy as np
import flappy_bird_gymnasium
import gymnasium as gym

# chemin absolu vers results/, peu importe d'où le script est lancé
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def play_episode(env, render=False):
    # use_lidar=False donne un vecteur de 12 valeurs, plus simple à traiter
    # qu'une observation lidar pour un premier agent
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    steps = 0

    while not done:
        # action tirée au hasard parmi les 2 actions possibles (battre ou pas)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

        if render:
            env.render()
            time.sleep(1 / 30)

    # info["score"] = nombre de tuyaux passés, c'est la métrique de référence
    # total_reward inclut aussi le shaping (proximité du centre du trou)
    return info["score"], steps, total_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true", help="affiche la fenetre du jeu")
    args = parser.parse_args()

    render_mode = "human" if args.render else None
    env = gym.make("FlappyBird-v0", use_lidar=False, render_mode=render_mode)

    scores = []
    steps_list = []
    rewards_list = []

    for episode in range(args.episodes):
        score, steps, total_reward = play_episode(env, render=args.render)
        scores.append(score)
        steps_list.append(steps)
        rewards_list.append(total_reward)

    env.close()

    # sauvegarde partie par partie pour pouvoir revérifier/retracer plus tard
    RESULTS_DIR.mkdir(exist_ok=True)
    output_file = RESULTS_DIR / "baseline_results.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "score", "steps", "total_reward"])
        for i in range(args.episodes):
            writer.writerow([i, scores[i], steps_list[i], rewards_list[i]])

    print(f"Episodes joues : {args.episodes}")
    print(f"Score moyen : {np.mean(scores):.2f} (ecart-type {np.std(scores):.2f})")
    print(f"Score max : {np.max(scores)}")
    print(f"Steps moyens survecus : {np.mean(steps_list):.1f}")
    print(f"Resultats sauvegardes dans {output_file}")


if __name__ == "__main__":
    main()
