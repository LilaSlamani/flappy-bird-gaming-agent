# Lit les logs d'entraînement (CSV générés par train.py) et trace la
# courbe de progression, comparée à la référence de l'agent aléatoire.

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

BASELINE_SCORE = 0.0  # score moyen mesuré avec l'agent aléatoire
MOVING_AVG_WINDOW = 50


def load_log(run_name):
    log_path = RESULTS_DIR / f"training_log_{run_name}.csv"
    episodes, scores = [], []
    with open(log_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            scores.append(float(row["score"]))
    return np.array(episodes), np.array(scores)


def moving_average(values, window):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_curve(run_names):
    plt.figure(figsize=(10, 6))

    for run_name in run_names:
        episodes, scores = load_log(run_name)
        avg = moving_average(scores, MOVING_AVG_WINDOW)
        avg_episodes = episodes[MOVING_AVG_WINDOW - 1:] if len(scores) >= MOVING_AVG_WINDOW else episodes
        plt.plot(avg_episodes, avg, label=f"{run_name} (moyenne glissante {MOVING_AVG_WINDOW})")

    plt.axhline(y=BASELINE_SCORE, color="red", linestyle="--", label="référence agent aléatoire")
    plt.xlabel("Episode")
    plt.ylabel("Score (tuyaux passés)")
    plt.title("Progression de l'agent DQN vs agent aléatoire")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = RESULTS_DIR / "learning_curve.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Courbe sauvegardee dans {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["run1"])
    args = parser.parse_args()
    plot_curve(args.runs)


if __name__ == "__main__":
    main()
