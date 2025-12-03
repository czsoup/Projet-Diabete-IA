# Analyse Prédictive du Diabète & Dashboard Interactif

> **Projet Universitaire - Master 1 Big Data** > *Techniques d'Apprentissage Artificiel*

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)](https://scikit-learn.org/)
[![React](https://img.shields.io/badge/Frontend-React-61DAFB)](https://reactjs.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Présentation du Projet

Ce projet vise à développer une solution complète de **Machine Learning** pour la détection précoce du diabète. Il s'appuie sur une analyse approfondie de données de santé, l'entraînement de modèles prédictifs robustes et le déploiement d'une interface utilisateur web.

L'objectif est de fournir un outil d'aide à la décision capable d'estimer le risque de diabète à partir d'indicateurs cliniques et comportementaux (IMC, tension, cholestérol, etc.).

### Accès au Dashboard
Une interface interactive permettant de tester le modèle en temps réel est disponible ici :
 **[DÉMO EN LIGNE : Diabetes Prediction Dashboard](https://czsoup.github.io/diabete-dashboard-react/)**

---

##  Architecture du Projet

```text
├── data
│   └── raw            <- Données brutes (diabetes2015.csv)
├── diabetes-dashboard <- Code source de l'application Web (React.js)
├── models             <- Modèles sérialisés (.pkl) et historiques d'entraînement
├── reports            
│   └── figures        <- Visualisations générées (Matrices de confusion, ROC, etc.)
├── src                <- Cœur du projet (Pipeline ETL & ML)
│   └── pipeline.py    <- Script unique d'exécution (End-to-End)
├── requirements.txt   <- Liste des dépendances Python
└── README.md          <- Documentation du projet

##  Installation et Exécution (Partie Data)

## Prérequis

git clone [https://github.com/czsoup/Projet-Diabete-IA.git](https://github.com/czsoup/Projet-Diabete-IA.git)
cd Projet-Diabete-IA
pip install -r requirements.txt

### Lancer le Pipeline ML
Le script principal automatise l'intégralité du processus : chargement des données, prétraitement, entraînement, évaluation et sauvegarde des artefacts.
python src/pipeline.py


## Méthodologie Data Science

### Préparation des Données

Nettoyage : Traitement des valeurs manquantes et doublons.

Feature Engineering : Sélection des variables les plus corrélées (Top 15 features).

Normalisation : Mise à l'échelle via StandardScaler.

Rééquilibrage : Utilisation de ADASYN (Adaptive Synthetic Sampling) pour corriger le déséquilibre des classes (Diabétique vs Non-Diabétique).

### Modélisation

Comparaison de plusieurs algorithmes supervisés :

Random Forest Classifier (Retenu pour sa robustesse).

XGBoost Classifier.

### Évaluation & Optimisation

L'accent a été mis sur la minimisation des Faux Négatifs (cas critiques en médecine).

Optimisation du seuil de décision (Threshold Tuning).

Métriques clés : Recall (Sensibilité) et F1-Score.


## Interface Utilisateur (React)

Le dossier diabetes-dashboard/ contient le code source de l'interface front-end. Elle permet à un utilisateur de saisir ses paramètres de santé et d'obtenir une prédiction instantanée via le modèle entraîné.

Framework : React.js

Hébergement : GitHub Pages

## Auteures

Projet réalisé dans le cadre du Master 1 Informatique - Parcours Big Data.

Ibtissam ZAID

Caroline ZHENG




