# Flappy Bird - Gaming Agent

Projet de groupe (cours Deep Learning). Un agent qui apprend à jouer à Flappy Bird
par renforcement (DQN), comparé à un agent qui joue au hasard.

## Installation

```
git clone https://github.com/LilaSlamani/flappy-bird-gaming-agent.git
cd flappy-bird-gaming-agent
pip install -r requirements.txt
```

## Architecture du repo

```
flappy-bird-gaming-agent/
├── README.md
├── carnet_essais.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── random_agent.py   agent aléatoire, baseline
│   ├── network.py          réseau de neurones du DQN
│   ├── replay_buffer.py    mémoire de replay
│   ├── agent.py             politique epsilon-greedy, loss (Bellman)
│   ├── train.py               boucle d'entraînement
│   ├── evaluate.py              courbe de progression
│   └── play.py                    recharge et vérifie le meilleur agent
├── results/
│   ├── baseline_results.csv     [généré par random_agent.py]
│   ├── training_log_run{1,2,3}.csv [généré par train.py, seeds 0/1/2]
│   └── learning_curve.png         [généré par evaluate.py, comparaison des 3 runs]
└── checkpoints/                    [poids des 3 agents entraînés, ignoré par git]
```

Chaque script se lance directement depuis la racine du repo, les chemins
vers `results/` et `checkpoints/` sont calculés automatiquement.

## Lancer l'agent aléatoire

```
python src/random_agent.py
```

Joue 100 parties au hasard, affiche le score moyen et sauvegarde les résultats
dans `results/baseline_results.csv`.

Options :
- `--episodes N` : change le nombre de parties (défaut 100)
- `--render` : ouvre la fenêtre du jeu pour regarder l'agent jouer

## Entraîner un agent

```
python src/train.py --episodes 3000 --seed 0 --run-name run1
```

Options :
- `--episodes` : nombre d'épisodes (défaut 2000)
- `--seed` : graine aléatoire
- `--run-name` : nom utilisé pour les fichiers de sortie (log et checkpoint)

## Tracer la courbe de progression

```
python src/evaluate.py --runs run1 run2 run3
```

Génère `results/learning_curve.png`, moyenne glissante du score comparée
à la référence de l'agent aléatoire.

## Recharger et vérifier le meilleur agent

```
python src/play.py --checkpoint best_agent_run1.pt --episodes 20
```

Reconstruit un réseau neuf, charge les poids sauvegardés, fait jouer
l'agent sans aucune exploration (`--render` pour voir la fenêtre du jeu).
C'est cette étape qui a révélé un vrai bug de sauvegarde en cours de
projet, voir `carnet_essais.md`.

## Le jeu

Flappy Bird, via le package `flappy-bird-gymnasium` (environnement Gymnasium).
Une seule action possible à chaque instant, ce qui rend l'algorithme simple à
implémenter tout en gardant un vrai jeu reconnaissable pour la démo.

## Ce que l'agent observe, peut faire, et ce qui le récompense

**Observation** : vecteur de 12 valeurs continues, normalisées entre -1 et 1
(mode `use_lidar=False`). Elles décrivent la position de l'oiseau, sa vitesse,
et la position des prochains tuyaux.

**Actions** : 2 actions discrètes possibles.
- 0 : ne rien faire (l'oiseau tombe sous l'effet de la gravité)
- 1 : battre des ailes (l'oiseau monte)

**Récompense** (par défaut dans l'environnement) :
- +0.1 à chaque frame où l'oiseau est encore en vie
- +1.0 quand il passe un tuyau
- -1.0 quand il meurt (collision ou sortie de l'écran)

## Score de référence (agent aléatoire)

Sur 100 parties : score moyen 0 (aucun tuyau passé), l'agent meurt
systématiquement après 50 steps. Détails dans `results/baseline_results.csv`
et `carnet_essais.md`.

## Méthode d'apprentissage

DQN (Deep Q-Network). Choisi parce que les actions sont discrètes (2 choix)
et l'état est de petite dimension (12 valeurs) : c'est le cas d'usage
classique pour lequel DQN a été conçu, pas besoin d'une méthode pour
actions continues (PPO, SAC...).

Réseau : 12 entrées, deux couches cachées de 128 neurones (ReLU), 2 sorties
(une valeur Q par action). Replay buffer de 50000 transitions, epsilon
décroissant de 1.0 à 0.02 sur 1500 épisodes, réseau cible resynchronisé
tous les 10 épisodes, gamma=0.99.

Une évaluation greedy (sans exploration, 10 parties) est effectuée tous
les 100 épisodes pendant l'entraînement, et sert à décider quel checkpoint
sauvegarder. Voir `carnet_essais.md` (essai 1) pour la raison technique de
ce choix : sauvegarder selon le score brut de l'épisode d'entraînement
était biaisé par l'exploration résiduelle.

## Résultats

Entraînement sur 3000 épisodes, relancé sur 3 graines différentes (seed
0, 1, 2) pour vérifier la stabilité. Vérification faite avec `play.py`
(checkpoint rechargé depuis un script neuf, aucune exploration) :

| | Agent aléatoire (100 parties) | run1 (seed 0) | run2 (seed 1) | run3 (seed 2) |
|---|---|---|---|---|
| Meilleur score en évaluation greedy | 0 | 7.7 | 26.5 | 1.1 |
| Score moyen après rechargement (20 parties) | 0 | 8.10 | - | - |

Les 3 graines apprennent toutes mieux que le hasard (0), mais avec un
écart important entre elles. C'est une observation documentée, pas
cachée : voir `carnet_essais.md` (essai 3) pour le détail et
l'interprétation.

Courbe de progression comparant les 3 runs : `results/learning_curve.png`.
Détail complet des essais (y compris la tentative ratée sur la
sauvegarde du meilleur agent) dans `carnet_essais.md`.

## Limites et pistes d'amélioration

- l'écart entre graines (1.1 à 26.5) montre que 3000 épisodes ne suffit
  pas à garantir un résultat stable : plus d'épisodes, ou une meilleure
  stratégie d'exploration au début de l'entraînement, réduirait
  probablement cet écart
- la fonction de récompense utilisée est celle par défaut de
  l'environnement (+0.1 vivant, +1 tuyau, -1 mort), jamais modifiée :
  une piste serait de tester un shaping différent pour voir si
  l'apprentissage devient plus rapide ou plus stable
- pas de recherche d'hyperparamètres (learning rate, taille du réseau,
  vitesse de décroissance d'epsilon) : les valeurs utilisées sont des
  choix de départ raisonnables, pas le résultat d'un tuning

## Vidéo

Lien : à ajouter une fois enregistrée.
