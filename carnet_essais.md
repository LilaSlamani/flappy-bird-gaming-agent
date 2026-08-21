# Carnet d'essais

## Résumé

| Essai | Changement | Résultat |
|---|---|---|
| Baseline | agent aléatoire | score moyen 0 |
| Essai 1 | DQN, 2000 épisodes, sauvegarde sur score brut | échec : l'agent rechargé ne rejoue pas (score 0), bug identifié |
| Essai 2 | même DQN, sauvegarde sur évaluation greedy | succès : score moyen 8.10 après rechargement vérifié |
| Essai 3 | 3 graines différentes (seeds 0/1/2), 3000 épisodes | toutes apprennent mieux que le hasard, mais résultat final très inégal (1.1 à 26.5) |

## Baseline : agent aléatoire

- 100 parties, score moyen 0, écart-type 0
- meurt systématiquement à exactement 50 steps (voir README pour l'explication)

## Essai 1 : DQN, 2000 épisodes, sélection du meilleur agent par score brut

**Paramètres :** gamma=0.99, lr=1e-3, batch=64, buffer=50000,
epsilon 1.0 -> 0.02 sur 1500 épisodes, target network resynchronisé tous
les 10 épisodes.

**Résultat pendant l'entraînement :** la moyenne glissante (50 épisodes)
progresse de 0 à environ 0.7 en fin d'entraînement, meilleur score brut
observé : 4.

**Problème découvert :** en rechargeant le checkpoint sauvegardé (celui
qui avait obtenu le score de 4) et en le faisant rejouer sans aucune
exploration (`play.py`), l'agent obtient un score de 0 sur 20 parties. En
inspectant les Q-values pendant une partie réelle, l'agent choisit presque
toujours "ne rien faire", laisse tomber l'oiseau jusqu'à la mort.

**Cause :** le score utilisé pour décider quel checkpoint sauvegarder
était le score de l'épisode d'entraînement, qui incluait encore 2%
d'exploration aléatoire (epsilon=0.02, jamais totalement nul). Les bons
scores pendant l'entraînement venaient en partie de ce hasard résiduel,
pas uniquement d'une politique vraiment apprise. Résultat : le
"meilleur" checkpoint sauvegardé ne représente pas la vraie performance
de la politique gloutonne (100% exploitation).

**Le tester ne suffit pas, il faut mesurer la bonne chose :** un score
qui monte pendant l'entraînement ne garantit pas un agent qui rejoue bien
une fois rechargé sans exploration. C'est justement pour ça que la tâche
"recharger depuis un script neuf pour vérifier" existe dans le brief,
et ça a permis d'attraper ce problème avant de le présenter comme un
succès.

## Essai 2 : DQN, mêmes paramètres, évaluation greedy périodique

**Changement :** toutes les 100 épisodes, on joue 10 parties avec
epsilon=0 (aucune exploration) et on sauvegarde le checkpoint seulement
si cette moyenne d'évaluation greedy s'améliore. Ça découple la
décision de sauvegarde du bruit d'exploration pendant l'entraînement.

**Entraînement étendu à 3000 épisodes** (au lieu de 2000) pour laisser
plus de temps à la politique de converger une fois epsilon proche de son
minimum.

**Résultat pendant l'entraînement :** l'évaluation greedy (10 parties sans
exploration, mesurée tous les 100 épisodes) reste à 0 jusqu'à l'épisode
~1700, puis progresse jusqu'à un pic de 7.7 vers l'épisode 2800.

**Vérification indépendante (`play.py`, checkpoint rechargé depuis un
script neuf, 20 parties, aucune exploration) :** score moyen 8.10,
score max 26, minimum 1 (jamais 0). À comparer aux 100 parties de l'agent
aléatoire : score moyen 0, jamais un seul tuyau passé.

**Conclusion de cet essai :** contrairement à l'essai 1, l'agent rechargé
rejoue vraiment et de façon cohérente avec ce que la courbe
d'entraînement montrait. Le fait de découpler la mesure (évaluation
greedy) de la collecte d'expérience (qui garde un peu d'exploration) a
résolu le problème de l'essai 1.

## Essai 3 : relance multi-seeds (même code, mêmes hyperparamètres)

**Ce qui change :** uniquement la graine aléatoire (seed 0, 1, 2 pour
run1, run2, run3), tout le reste identique à l'essai 2.

**Résultats (meilleur score en évaluation greedy sur 3000 épisodes) :**

| Run | Seed | Meilleur score eval |
|---|---|---|
| run1 | 0 | 7.7 |
| run2 | 1 | 26.5 |
| run3 | 2 | 1.1 |

**Observation :** les 3 courbes ne se ressemblent pas du tout (voir
`results/learning_curve.png`). run1 et run2 décollent à peu près au même
moment (épisode ~1500-1750) et suivent une trajectoire proche jusqu'à
l'épisode 2500, avant que run2 prenne nettement le dessus. run3 décolle
beaucoup plus tard (~2400) et reste largement en dessous des deux autres
sur toute la durée de l'entraînement.

**Ce que ça apprend sur l'agent :** avec seulement 3000 épisodes,
l'entraînement n'est pas encore stable d'une graine à l'autre. Le
résultat final dépend beaucoup du hasard de l'exploration au début de
l'entraînement (quelles séquences d'actions aléatoires ont, par chance,
permis de passer un premier tuyau et donc de commencer à recevoir un
signal d'apprentissage utile). Ce n'est pas un bug, c'est une limite
réelle de cet entraînement avec ce nombre d'épisodes : plus d'épisodes,
ou une meilleure stratégie d'exploration, réduirait probablement cet
écart entre graines.
