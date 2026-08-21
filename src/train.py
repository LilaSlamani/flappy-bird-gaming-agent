# Boucle d'entraînement du DQN : joue, stocke dans le replay buffer,
# apprend par batches, évalue périodiquement sans exploration pour
# décider quel agent sauvegarder (voir carnet_essais.md, essai 1 et 2).

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import flappy_bird_gymnasium
import gymnasium as gym

from network import QNetwork
from replay_buffer import ReplayBuffer
from agent import select_action, compute_loss

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CHECKPOINTS_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
BUFFER_CAPACITY = 50000
MIN_BUFFER_SIZE = 1000
TARGET_UPDATE_EVERY = 10  # episodes
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY_EPISODES = 1500
EVAL_EVERY = 100  # episodes
EVAL_EPISODES = 10


def epsilon_for_episode(episode):
    fraction = min(1.0, episode / EPSILON_DECAY_EPISODES)
    return EPSILON_START + fraction * (EPSILON_END - EPSILON_START)


def evaluate_greedy(policy_net, env, n_episodes):
    # aucune exploration ici : ça mesure la vraie politique apprise,
    # pas ce que le hasard résiduel d'epsilon fait gagner en plus
    scores = []
    for i in range(n_episodes):
        obs, info = env.reset(seed=10000 + i)
        done = False
        while not done:
            action = select_action(policy_net, obs, epsilon=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        scores.append(info["score"])
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", type=str, default="run1")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = gym.make("FlappyBird-v0", use_lidar=False)

    policy_net = QNetwork()
    target_net = QNetwork()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_CAPACITY)

    RESULTS_DIR.mkdir(exist_ok=True)
    CHECKPOINTS_DIR.mkdir(exist_ok=True)
    log_path = RESULTS_DIR / f"training_log_{args.run_name}.csv"
    best_checkpoint_path = CHECKPOINTS_DIR / f"best_agent_{args.run_name}.pt"

    best_eval_score = -1
    log_rows = []

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        epsilon = epsilon_for_episode(episode)
        done = False
        total_reward = 0.0

        while not done:
            action = select_action(policy_net, obs, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer.push(obs, action, reward, next_obs, float(done))
            obs = next_obs
            total_reward += reward

            if len(buffer) >= MIN_BUFFER_SIZE:
                batch = buffer.sample(BATCH_SIZE)
                loss = compute_loss(policy_net, target_net, batch, GAMMA)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        score = info["score"]
        log_rows.append([episode, score, total_reward, epsilon])

        if episode % TARGET_UPDATE_EVERY == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % EVAL_EVERY == 0 and episode > 0:
            eval_score = evaluate_greedy(policy_net, env, EVAL_EPISODES)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                torch.save(policy_net.state_dict(), best_checkpoint_path)
            print(
                f"[eval greedy] episode {episode} | score moyen sur "
                f"{EVAL_EPISODES} parties sans exploration : {eval_score:.2f} | "
                f"meilleur eval : {best_eval_score:.2f}"
            )

        if episode % 50 == 0:
            recent_scores = [row[1] for row in log_rows[-50:]]
            print(
                f"episode {episode} | score {score} | "
                f"moyenne 50 derniers {np.mean(recent_scores):.2f} | "
                f"epsilon {epsilon:.3f}"
            )

    env.close()

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "score", "total_reward", "epsilon"])
        writer.writerows(log_rows)

    print(f"Entrainement termine. Meilleur score en evaluation greedy : {best_eval_score}")
    print(f"Log sauvegarde dans {log_path}")
    if best_eval_score > -1:
        print(f"Meilleur agent sauvegarde dans {best_checkpoint_path}")
    else:
        print("Aucun checkpoint sauvegarde (entrainement trop court pour atteindre une evaluation)")


if __name__ == "__main__":
    main()
