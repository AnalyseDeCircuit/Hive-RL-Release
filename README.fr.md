# Projet Hive Game Python (Français)

[中文 | English | Français | Español]

## Présentation du projet
Ce projet est une implémentation Python de Hive, prenant en charge humain vs humain, humain vs IA, entraînement et évaluation IA. Il prend en charge les pièces de base et DLC, avec logique complète, gestion des joueurs, IA et entraînement réseau de neurones.

## Fonctionnalités principales
- **Humain vs Humain** : Deux joueurs locaux, règles complètes de Hive.
- **Humain vs IA** : Affrontez une IA, qui peut charger un modèle ou jouer aléatoirement.
- **Entraînement IA** : Auto-apprentissage par renforcement pour améliorer l'IA.
- **Évaluation IA** : Évaluation en lot des performances de l'IA.
- **Plateau & Règles** : Toutes les pièces de base et DLC, règles de pose/déplacement, conditions de victoire.

## Architecture
- **Langage** : Python 3
- **Modules principaux** :
  - `main.py` : Point d'entrée, menu, boucle principale
  - `game.py` : Logique de jeu et gestion d'état
  - `player.py` / `ai_player.py` : Joueur et IA
  - `board.py` : Plateau et gestion des pièces
  - `hive_env.py` / `game_state.py` : Environnement IA et encodage d'état
  - `neural_network.py` : Réseau de neurones
  - `ai_trainer.py` / `ai_evaluator.py` : Entraînement et évaluation IA
  - `utils.py` : Constantes et utilitaires
- **IA** : Apprentissage par renforcement (Q-Learning/DQN), réseau de neurones personnalisé, replay buffer
- **Structure de données** : POO, supporte clone/copie profonde/simulation d'état

## Lancement
1. Installer les dépendances :

   ```bash
   pip install -r requirements.txt
   ```

2. Lancer le jeu :

   ```bash
   python main.py
   ```

3. Choisir le mode dans le menu.

## Cas d'utilisation
- Développement de jeux de société & recherche IA
- Pratique RL & IA de jeu
- Apprentissage POO Python & architecture projet

## Autres notes
- Prend en charge les pièces de base et DLC (Coccinelle, Moustique, Cloporte)
- Code propre, facile à étendre
- Voir Q&S.md pour les détails techniques et correctifs

## Structure des modules

Ce projet utilise une architecture POO en couches et découplée :

- **Interface & Processus principal**
  - `main.py` : Menu, interaction utilisateur, boucle principale, dispatch modules
- **Logique de jeu & état**
  - `game.py` : Flux de jeu, changement de tour, victoire/défaite, gestion joueurs
  - `board.py` : Structure du plateau, pose/déplacement/validation des pièces
  - `player.py` : Joueur humain, inventaire, pose/déplacement
  - `piece.py` : Types, attributs, comportements des pièces
- **IA & Environnement**
  - `ai_player.py` : Joueur IA, hérite de Player, RL et réseau de neurones
  - `hive_env.py` : Environnement d'entraînement IA, état/action/récompense, style OpenAI Gym
  - `game_state.py` : Encodage d'état et extraction de caractéristiques pour l'IA
- **Entraînement & Évaluation IA**
  - `ai_trainer.py` : Auto-entrainement, collecte d'expérience, mise à jour modèle
  - `ai_evaluator.py` : Évaluation IA et statistiques
  - `neural_network.py` : Réseau de neurones personnalisé pour la prédiction
- **Utilitaires & Constantes**
  - `utils.py` : Constantes, helpers, config globale

---

## Principe RL (détaillé)

L'IA utilise Q-Learning/DQN avec un réseau de neurones personnalisé. Points clés :

### 1. Encodage d'état
- Chaque état est un vecteur 814D :
  - 800 : plateau 10x10, one-hot du type de pièce au sommet (8 types)
  - 10 : inventaire main des deux joueurs (5 pièces de base, normalisé)
  - 4 : joueur courant, tour, reines posées
- Voir RLTechReport.md pour le code.

### 2. Espace d'action
- Toutes les poses/déplacements légaux sont encodés en int (Action.encode_*) :
  - Pose : type de pièce et coordonnées
  - Déplacement : coordonnées de/à
- L'IA énumère toutes les actions possibles et filtre la légalité.
- Voir RLTechReport.md pour le code.

### 3. Réseau de neurones
- MLP personnalisé :
  - Entrée : 814
  - Caché : 256, ReLU
  - Sortie : 1 (valeur d'état, extensible à Q(s,a))
- Voir RLTechReport.md pour le code.

### 4. Politique & exploration
- Epsilon-greedy :
  - Avec proba epsilon, action aléatoire (exploration), sinon action max (exploitation)
  - Epsilon décroît (ex : *0.995 par partie), min 0.01
- Voir RLTechReport.md pour le code.

### 5. Replay & entraînement
- Chaque étape (s, a, r, s', done) va dans le buffer
- À chaque entraînement, batch aléatoire, calcul des cibles :
  - Non terminal : cible = r + γ * V(s')
  - Terminal : cible = r
- Entraînement par MSE

### 7. Auto-jeu & mise à jour modèle
- L'IA s'auto-affronte, accumule l'expérience, sauvegarde le modèle tous les 100 jeux
- Modèle entraîné chargeable pour jeu/éval

### 8. Évaluation & généralisation
- Après entraînement, utiliser ai_evaluator.py pour parties batch, taux de victoire, pas moyen
- Comparaison possible sous différents epsilon/modèles

---

Pour toute suggestion ou bug, ouvrez une issue ou PR !
