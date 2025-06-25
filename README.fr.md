[🇨🇳 中文](README.md) | [🇬🇧 English](README.en.md) | [🇪🇸 Español](README.es.md)

# Hive-RL : Une IA pour le jeu de société Hive basée sur l'Apprentissage par Renforcement

## Introduction

Hive-RL est un projet Python basé sur l'Apprentissage par Renforcement (AR) qui vise à entraîner une Intelligence Artificielle (IA) de haut niveau pour le jeu de société **Hive**. Ce projet met en œuvre la logique complète du jeu, un environnement d'apprentissage par renforcement compatible avec la norme OpenAI Gym/Gymnasium, et un entraîneur d'IA utilisant un Réseau Q Profond (DQN).

## Caractéristiques du Projet

* **Implémentation Complète du Jeu** : Implémente avec précision les règles de base de Hive et le mouvement de toutes les pièces, y compris les **pièces d'extension DLC** officielles (Coccinelle, Moustique, Cloporte).
* **Architecture Modulaire** : Le code est clairement structuré en modules pour la logique du jeu, l'environnement AR, le joueur IA, l'entraîneur, l'évaluateur, etc., ce qui le rend facile à comprendre et à étendre.
* **Piloté par l'Apprentissage par Renforcement** : Utilise un Réseau Q Profond (DQN) comme algorithme de base, permettant à l'IA d'apprendre à partir de zéro et d'évoluer continuellement grâce à l'Auto-Jeu (Self-Play) et à diverses stratégies d'entraînement avancées.
* **Stratégies d'Entraînement Avancées** :
  * **Auto-Jeu Parallèle** : Utilise le multiprocessing pour échantillonner en parallèle, accélérant considérablement l'entraînement.
  * **Apprentissage Curriculaire** : Permet à l'IA de commencer à apprendre à partir de tâches simplifiées (par exemple, apprendre à placer la Reine des Abeilles en premier) et de passer progressivement au jeu complet, améliorant l'efficacité de l'apprentissage.
  * **Entraînement Antagoniste** : Améliore la robustesse de l'IA en jouant contre un adversaire qui choisit spécifiquement les "pires" coups.
  * **Entraînement d'Ensemble** : Entraîne plusieurs modèles d'IA indépendants et utilise le vote lors de la prise de décision pour améliorer la précision et la stabilité des choix.
* **Visualisation et Évaluation** : Fournit divers outils de visualisation pour tracer les courbes de récompense, les courbes de perte, les courbes de taux de victoire et d'autres statistiques pendant le processus d'entraînement, ce qui facilite l'analyse des progrès d'apprentissage de l'IA.
* **Interface Conviviale** : Offre un menu principal en ligne de commande qui prend en charge divers modes, y compris Humain contre Humain, Humain contre IA, Entraînement de l'IA et Évaluation de l'IA.

## Architecture du Projet

* `main.py` : Le point d'entrée principal du projet, fournissant un menu interactif en ligne de commande.
* `game.py` : La logique principale du jeu, gérant le déroulement du jeu, les joueurs et les tours.
* `board.py` : Représentation du plateau et opérations de base.
* `piece.py` : Définit les propriétés et les règles de mouvement pour toutes les pièces (y compris le DLC).
* `player.py` : La classe de base pour les joueurs, gérant la main et les actions de base.
* `ai_player.py` : La classe du joueur IA, implémentant la sélection d'actions et la relecture d'expérience basées sur un réseau de neurones.
* `hive_env.py` : L'environnement de jeu Hive suivant l'API Gymnasium, utilisé pour l'entraînement par apprentissage par renforcement.
* `neural_network.py` : Une implémentation de réseau de neurones profond basée sur PyTorch.
* `ai_trainer.py` : L'entraîneur d'IA, comprenant divers modes d'entraînement (auto-jeu parallèle, apprentissage curriculaire, entraînement antagoniste, etc.).
* `ai_evaluator.py` : L'évaluateur d'IA, utilisé pour tester le taux de victoire de l'IA contre un joueur aléatoire.
* `utils.py` : Fournit des fonctions et des outils d'aide.
* `requirements.txt` : Bibliothèques de dépendances du projet.

## Comment Exécuter

### 1. Configuration de l'Environnement

Tout d'abord, assurez-vous d'avoir installé Python 3.10 ou une version ultérieure. Ensuite, installez toutes les dépendances requises :

```bash
pip install -r requirements.txt
```

### 2. Exécuter le Programme Principal

Démarrez le menu principal du projet avec la commande suivante :

```bash
python main.py
```

Vous verrez les options suivantes :

1. **Humain contre Humain** : Jouez contre un autre joueur local.
2. **Humain contre IA** : Jouez contre une IA entraînée.
3. **Entraînement de l'IA** : Entraînez un nouveau modèle d'IA ou continuez l'entraînement à partir d'un point de contrôle.
4. **Évaluer l'IA & Graphiques** : Évaluez les performances de l'IA et tracez les courbes d'entraînement.
5. **Quitter le Jeu** : Quittez le programme.

### 3. Entraîner l'IA

* Sélectionnez l'option `Entraînement de l'IA` dans le menu principal.
* Vous pouvez choisir de **commencer un nouvel entraînement** ou de **continuer à partir d'un point de contrôle précédent**.
* Ensuite, sélectionnez un mode d'entraînement, tel que **l'entraînement de base avec échantillonnage parallèle** ou **l'auto-jeu**.
* Pendant l'entraînement, le modèle et les statistiques seront automatiquement sauvegardés dans le répertoire `models/`, dans un dossier nommé avec un horodatage et le statut du DLC.
* Vous pouvez interrompre l'entraînement à tout moment avec `Ctrl+C`, et le programme sauvegardera automatiquement la progression actuelle pour la reprendre plus tard.

### 4. Jouer Contre l'IA

* Sélectionnez l'option `Humain contre IA` dans le menu principal.
* Le programme listera automatiquement tous les modèles d'IA disponibles dans le répertoire `models/`. Vous pouvez en choisir un pour jouer contre.
* Pendant le jeu, entrez vos coups comme demandé.

## Principes de l'Apprentissage par Renforcement

L'IA de ce projet est basée sur un **Réseau Q Profond (DQN)**, un algorithme d'apprentissage par renforcement basé sur la valeur. L'idée principale est d'entraîner un réseau de neurones à approximer la **fonction Q** `Q(s, a)`, qui prédit le retour à long terme (récompense) de l'action `a` dans un état donné `s`.

* **État** : Une représentation vectorielle de la situation de jeu actuelle, comprenant le type de pièce à chaque position sur le plateau, le nombre de pièces restantes dans la main de chaque joueur, le numéro du tour actuel, etc.
* **Action** : L'une de toutes les opérations légales de "placement" ou de "déplacement".
* **Récompense** : Le signal de retour que l'IA reçoit de l'environnement après avoir effectué une action.
  * **Gagner** : Reçoit une grande récompense positive.
  * **Perdre** : Reçoit une grande récompense négative.
  * **Match Nul** : Reçoit une récompense nulle ou une petite récompense positive/négative.
  * **Mise en Forme de la Récompense (Reward Shaping)** : Pour guider l'IA à apprendre plus rapidement, nous avons conçu une série de récompenses intermédiaires, telles que :
    * Une récompense positive pour avoir encerclé la Reine des Abeilles de l'adversaire.
    * Une pénalité si sa propre Reine des Abeilles est encerclée.
    * Une petite récompense positive pour avoir effectué un mouvement ou un placement légal.
    * Une très petite récompense négative pour chaque pas effectué, afin d'encourager l'IA à gagner le plus rapidement possible.
* **Processus d'Entraînement** :
  1. **Échantillonnage** : L'IA (ou plusieurs IA parallèles) joue le jeu par auto-jeu dans l'environnement, collectant un grand nombre de tuples d'expérience `(état, action, récompense, état_suivant)`.
  2. **Relecture d'Expérience** : Les expériences collectées sont stockées dans un "pool d'expériences".
  3. **Entraînement** : Un petit lot d'expériences est tiré au hasard du pool d'expériences pour entraîner le réseau de neurones. L'objectif de l'entraînement est de rendre la valeur prédite de `Q(s, a)` aussi proche que possible de la **valeur Q cible** (généralement `récompense + facteur_de_réduction * max(Q(état_suivant, toutes_les_actions_légales))`).
  4. **Exploration vs. Exploitation** : L'IA utilise une stratégie **ε-greedy** pour sélectionner les actions. C'est-à-dire qu'avec une probabilité de ε, elle choisit une action légale aléatoire (exploration), et avec une probabilité de 1-ε, elle choisit l'action avec la valeur Q la plus élevée (exploitation). Au fur et à mesure que l'entraînement progresse, ε diminue progressivement, ce qui amène l'IA à passer de l'exploration aléatoire à une dépendance accrue aux stratégies optimales qu'elle a apprises.

Grâce à des dizaines de milliers de parties d'auto-jeu et d'entraînement, le réseau de neurones de l'IA peut progressivement apprendre les motifs et les stratégies complexes du plateau de Hive, atteignant ainsi un haut niveau de jeu.
