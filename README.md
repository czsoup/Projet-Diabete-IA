# Analyse et Prédiction du Diabète (Projet IA)

Ce projet vise à prédire le risque de diabète en utilisant des techniques d'apprentissage automatique (Machine Learning) sur des données de santé. Le projet suit la structure standard **Cookiecutter Data Science**.

##  Structure du Projet

```text
├── data
│   └── raw            <- Données originales (diabetes2015.csv)
├── diabetes-dashboard <- Application Web React (Interface Utilisateur)
├── models             <- Modèles entraînés (.pkl) et métriques (.json)
├── notebooks          <- Jupyter Notebooks (Exploration)
├── reports            
│   └── figures        <- Graphiques générés par l'analyse
├── src                <- Code source Python
│   └── pipeline.py    <- Script principal (Entraînement & Évaluation)
├── requirements.txt   <- Dépendances Python
└── README.md          <- Ce fichier

🚀 Installation et Exécution
Installer les dépendances :

Bash
pip install -r requirements.txt

Lancer l'analyse complète : Ce script charge les données, entraîne les modèles (RandomForest & XGBoost), évalue les performances et sauvegarde les résultats.

Bash
python src/pipeline.py

Accéder au Dashboard Web : L'interface utilisateur est située dans le dossier diabetes-dashboard.

Version en ligne : https://czsoup.github.io/diabete-dashboard-react/

Code source : Voir le dossier diabetes-dashboard/

📊 Méthodologie
Données : Dataset diabetes2015.csv (indicateurs de santé).

Préparation : Nettoyage, normalisation (StandardScaler) et rééquilibrage des classes via ADASYN.

Modèles testés : RandomForestClassifier et XGBoostClassifier.

Évaluation : Optimisation du seuil de décision basée sur le F1-Score et le Recall (pour minimiser les faux négatifs médicaux).

👤 Auteur
[Ibtissam ZAID / Caroline ZHENG] Master 1 Big Data - Techniques d'Apprentissage Artificiel

