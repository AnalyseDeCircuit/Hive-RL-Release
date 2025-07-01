[🇬🇧 English](README.en.md) | [🇨🇳 中文](README.md) | [🇪🇸 Español](README.es.md)

# Hive-RL : IA Hive basée sur l'Apprentissage par Renforcement

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📖 Introduction

Hive-RL est un projet avancé d'apprentissage par renforcement dédié à l'entraînement d'une IA de haut niveau pour le jeu **Hive**. Ce projet utilise des techniques modernes d'apprentissage par renforcement profond, implémentant un moteur de jeu complet, un système de récompenses scientifique et divers algorithmes d'entraînement avancés.

**Hive** est un jeu de stratégie primé qui ne nécessite pas de plateau, avec des règles simples mais une profondeur stratégique exceptionnelle. L'objectif des joueurs est d'encercler la reine adverse en plaçant et déplaçant diverses pièces d'insectes.

## ✨ Fonctionnalités Principales

### 🎮 Moteur de Jeu Complet
- **Implémentation précise des règles** : Conforme aux règles officielles de Hive
- **Support des extensions DLC** : Inclut la coccinelle, le moustique, le cloporte et autres pièces officielles
- **Plateau haute performance** : Structures de données optimisées et accélération Numba
- **Validation des actions** : Vérification stricte de la légalité et gestion d'erreurs

### 🧠 Système IA Avancé
- **Deep Q-Network (DQN)** : Architecture de réseau neuronal moderne basée sur PyTorch
- **Façonnage de récompenses scientifique** : Système de récompenses multi-niveaux soigneusement conçu
- **Replay d'expérience** : Réutilisation efficace des échantillons et stabilité d'apprentissage
- **Stratégie ε-greedy** : Stratégie dynamique équilibrant exploration et exploitation

### 🚀 Framework d'Entraînement Haute Performance
- **Auto-jeu parallèle** : Échantillonnage parallèle multi-processus, amélioration significative de l'efficacité d'entraînement
- **Apprentissage par curriculum** : Apprentissage progressif des règles de base aux stratégies avancées
- **Entraînement adversarial** : Amélioration de la robustesse de l'IA par échantillons adversariaux
- **Fusion de modèles** : Système de décision par vote multi-modèles

### 📊 Visualisation et Analyse
- **Surveillance en temps réel** : Courbes de récompenses, pertes et taux de victoire pendant l'entraînement
- **Analyse de performance** : Statistiques détaillées de fin de partie et analyse comportementale
- **Évaluation de modèle** : Tests de performance automatisés

## 🏗️ Architecture du Projet

```
Hive-RL/
├── Moteur Principal
│   ├── game.py              # Logique principale du jeu
│   ├── board.py             # Représentation et opérations du plateau
│   ├── piece.py             # Types de pièces et règles de mouvement
│   └── player.py            # Classe de base des joueurs
├── Apprentissage par Renforcement
│   ├── hive_env.py          # Environnement Gymnasium
│   ├── ai_player.py         # Implémentation du joueur IA
│   ├── neural_network.py    # Architecture du réseau neuronal
│   └── improved_reward_shaping.py  # Système de façonnage des récompenses
├── Framework d'Entraînement
│   ├── ai_trainer.py        # Entraîneur principal
│   ├── parallel_sampler.py  # Échantillonneur parallèle
│   └── ai_evaluator.py      # Évaluateur de performance
├── Outils d'Analyse
│   ├── analyze_model.py     # Analyse de modèle
│   └── plot_*.py           # Outils de visualisation
└── Interface Utilisateur
    └── main.py             # Menu principal
```

## 🚀 Démarrage Rapide

### Prérequis
- Python 3.10+
- PyTorch 2.0+
- NumPy, Matplotlib, Gymnasium
- Numba (optimisation des performances)

### Installation des Dépendances
```bash
pip install -r requirements.txt
```

### Lancement du Projet
```bash
python main.py
```

### Options du Menu Principal
1. **Human vs Human** - Combat local à deux joueurs
2. **Human vs AI** - Combat homme-machine
3. **AI Training** - Entraînement de l'IA
4. **Evaluate AI & Plots** - Évaluation des performances
5. **Exit Game** - Quitter

## 🎯 Entraînement de l'IA

### Modes d'Entraînement
1. **Entraînement de base par échantillonnage parallèle** - Entraînement multi-processus efficace
2. **Entraînement de raffinement par auto-jeu** - Optimisation stratégique approfondie
3. **Entraînement par vote d'ensemble** - Fusion multi-modèles
4. **Entraînement de robustification adversariale** - Amélioration de la résistance aux perturbations
5. **Apprentissage par curriculum** - Acquisition progressive des compétences

### Phases d'Apprentissage par Curriculum
- **Foundation (0-40k épisodes)** - Apprentissage des règles de base
- **Strategy (40k-90k épisodes)** - Développement de la pensée stratégique
- **Mastery (90k-120k épisodes)** - Maîtrise des stratégies avancées

### Caractéristiques d'Entraînement
- **Sauvegarde automatique** : Progression d'entraînement sauvegardée en temps réel, support de reprise
- **Surveillance des performances** : Affichage en temps réel de la vitesse d'entraînement et de l'état de convergence
- **Planification intelligente** : Ajustement dynamique d'epsilon et du taux d'apprentissage
- **Optimisation multi-processus** : 10 workers parallèles, amélioration de 10x de la vitesse d'entraînement

## 🔬 Principes Techniques

### Framework d'Apprentissage par Renforcement
- **Espace d'états** : Vecteur de 820 dimensions incluant l'état du plateau, informations de main, progression du jeu
- **Espace d'actions** : 20 000 actions discrètes couvrant tous les placements et mouvements possibles
- **Système de récompenses** : Conception de récompenses multi-niveaux, de la survie de base aux stratégies avancées

### Système de Façonnage des Récompenses
```python
Récompenses Terminales (Poids : 60-63%)
├── Victoire: +5.0 + bonus de vitesse
├── Défaite: -6.0 (reine encerclée)
├── Timeout: -3.0 (pénalité de délai)
└── Match nul: Ajustement fin selon l'avantage

Récompenses Stratégiques (Poids : 25-40%)
├── Progrès d'encerclement: Récompenses progressives
├── Amélioration défensive: Récompenses de position sûre
└── Coordination des pièces: Évaluation de la valeur positionnelle

Récompenses de Base (Poids : 5-15%)
├── Récompense de survie: Valeur positive minimale
└── Récompense d'action: Encouragement d'actions légales
```

### Architecture du Réseau Neuronal
- **Couche d'entrée** : Vecteur d'état de 820 dimensions
- **Couches cachées** : Multiples couches entièrement connectées, activation ReLU
- **Couche de sortie** : Prédiction de valeurs Q de 20 000 dimensions
- **Optimiseur** : Adam, taux d'apprentissage dynamique
- **Régularisation** : Dropout, écrêtage de gradient

## 📈 Métriques de Performance

### Efficacité d'Entraînement
- **Vitesse parallèle** : >1000 épisodes/heure
- **Temps de convergence** : 3-4 heures pour compléter l'apprentissage par curriculum
- **Efficacité d'échantillons** : Niveau expert atteint en 120k épisodes

### Capacités de l'IA
- **Performance de taux de victoire** : >90% de taux de victoire contre joueurs aléatoires
- **Profondeur stratégique** : Profondeur de réflexion moyenne de 15-20 coups
- **Vitesse de réaction** : <0.1 seconde/coup

### Stabilité
- **Variance des récompenses** : <0.1 en fin d'entraînement
- **Cohérence stratégique** : >95% de taux de reproduction des décisions pour la même situation
- **Robustesse** : Maintien de haute performance sous perturbations adversariales

## 🔧 Configuration Avancée

### Récompenses Personnalisées
```python
# Créer un façonneur de récompenses personnalisé
from improved_reward_shaping import HiveRewardShaper

shaper = HiveRewardShaper('custom')
shaper.config['terminal_weight'] = 0.7  # Augmenter le poids des récompenses terminales
shaper.config['strategy_weight'] = 0.3  # Ajuster le poids des récompenses stratégiques
```

### Optimisation des Paramètres d'Entraînement
```python
# Ajuster les hyperparamètres dans ai_trainer.py
batch_size = 32          # Taille de lot
learning_rate = 0.001    # Taux d'apprentissage
epsilon_start = 0.9      # Taux d'exploration initial
epsilon_end = 0.05       # Taux d'exploration final
discount_factor = 0.95   # Facteur de remise
```

### Configuration Parallèle
```python
# Ajuster le nombre de workers parallèles
num_workers = 10         # Ajuster selon le nombre de cœurs CPU
episodes_per_worker = 100 # Nombre d'épisodes par worker
queue_maxsize = 100      # Taille de la file d'attente
```

## 🐛 Dépannage

### Problèmes Courants
1. **Entraînement lent**
   - Vérifier la configuration des workers parallèles
   - Confirmer que la file d'attente n'est pas bloquée
   - Vérifier la transmission correcte du reward_shaper

2. **Comportement anormal de l'IA**
   - Vérifier la configuration du système de récompenses
   - Valider la raisonnabilité des statistiques terminales
   - Analyser la courbe de décroissance d'epsilon

3. **Mémoire insuffisante**
   - Réduire batch_size
   - Ajuster la taille du tampon de replay d'expérience
   - Utiliser moins de workers parallèles

### Outils de Débogage
```bash
# Analyser le dernier modèle d'entraînement
python analyze_model.py

# Visualiser les courbes d'entraînement
python plot_reward_curve.py

# Tester la configuration de l'environnement
python test_environment.py
```

## 🤝 Guide de Contribution

Nous accueillons les contributions de la communauté ! Veuillez consulter les directives suivantes :

### Environnement de Développement
```bash
# Cloner le dépôt
git clone <repository-url>
cd Hive-RL

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate  # Windows

# Installer les dépendances de développement
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Standards de Code
- Suivre le style de code PEP 8
- Ajouter des annotations de type
- Écrire des tests unitaires
- Mettre à jour la documentation

### Processus de Soumission
1. Fork le projet
2. Créer une branche de fonctionnalité
3. Committer les changements de code
4. Créer une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- **Jeu Hive** conçu par John Yianni
- Merci aux communautés open source PyTorch et Gymnasium
- Remerciements spéciaux à tous les contributeurs et utilisateurs testeurs

## 📞 Contact

- **Issues** : [GitHub Issues](../../issues)
- **Discussions** : [GitHub Discussions](../../discussions)
- **Email** : [your-email@example.com](mailto:your-email@example.com)

---

**Hive-RL** : Où l'IA rencontre l'élégance du Hive ! 🐝♟️🤖
