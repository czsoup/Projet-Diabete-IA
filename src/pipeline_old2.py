import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                            roc_auc_score, roc_curve, precision_recall_curve, 
                            f1_score, matthews_corrcoef, make_scorer)
from imblearn.over_sampling import ADASYN
from collections import Counter
import joblib
import warnings
from datetime import datetime
import json
import os
import sys
from pathlib import Path  

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ==========================================
#  CONFIGURATION DES CHEMINS (Cookiecutter)
# ==========================================
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes2015.csv"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print(f" Racine du projet : {PROJECT_ROOT}")
print(f" Données attendues : {DATA_PATH}")
print(f" Modèles sauvegardés dans : {MODELS_DIR}")
print(f" Graphiques sauvegardés dans : {FIGURES_DIR}")

# -------------------------------
# 1. Chargement et Analyse Exploratoire
# -------------------------------
print("\n" + "="*70)
print("ANALYSE EXPLORATOIRE DES DONNÉES")
print("="*70)

if not DATA_PATH.exists():
    sys.exit(f" ERREUR CRITIQUE : Le fichier est introuvable ici : {DATA_PATH}\n--> Vérifiez que 'diabetes2015.csv' est bien dans le dossier 'data/raw/'.")

df = pd.read_csv(DATA_PATH)
df['Diabetes_binary'] = df['Diabetes_012'].apply(lambda x: 1 if x >= 1 else 0)

print(f"\n Taille du dataset : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
print(f"\n Distribution de Diabetes_012 :")
print(df['Diabetes_012'].value_counts().sort_index())
print(f"\n Distribution binaire :")
print(df['Diabetes_binary'].value_counts())
print(f" Taux de diabétiques : {df['Diabetes_binary'].mean()*100:.2f}%")
print(f"\n🔍 Valeurs manquantes : {df.isnull().sum().sum()}")

correlations = df.corr()['Diabetes_binary'].sort_values(ascending=False)
print("\n Top 10 variables corrélées avec le diabète :")
print(correlations.head(11)[1:])  # Exclure la corrélation avec elle-même

# Visualisation des classes
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
df['Diabetes_012'].value_counts().plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Distribution Diabetes_012 (0: Non, 1: Pré, 2: Diabète)')
df['Diabetes_binary'].value_counts().plot(kind='bar', ax=axes[1], color=['green', 'red'])
axes[1].set_title('Distribution Binaire (0: Sain, 1: À risque)')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '01_distribution_classes.png', dpi=300, bbox_inches='tight')
plt.show()

# Corrélation top variables
top_features = correlations.head(11).index[1:]
correlation_matrix = df[list(top_features) + ['Diabetes_binary']].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=1)
plt.title('Matrice de corrélation - Top 10 variables')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '02_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

X = df.drop(['Diabetes_012', 'Diabetes_binary'], axis=1)
y = df['Diabetes_binary']

# -------------------------------
# 2. Split train / val / test
# -------------------------------
print("\n" + "="*70)
print("PRÉPARATION DES DONNÉES")
print("="*70)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)

print(f"Train set : {X_train.shape[0]:,} échantillons")
print(f"Validation set : {X_val.shape[0]:,} échantillons")
print(f"Test set : {X_test.shape[0]:,} échantillons")

# -------------------------------
# 3. Normalisation
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# -------------------------------
# 4. Sur-échantillonnage ADASYN
# -------------------------------
print("\n" + "="*70)
print("RÉÉQUILIBRAGE DES CLASSES (ADASYN)")
print("="*70)
print(f"Distribution avant ADASYN: {Counter(y_train)}")
adasyn = ADASYN(random_state=42, n_neighbors=5)
X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
print(f"Distribution après ADASYN: {Counter(y_train_resampled)}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
pd.Series(y_train).value_counts().plot(kind='bar', ax=ax[0], color=['green', 'red'])
ax[0].set_title('Avant ADASYN')
pd.Series(y_train_resampled).value_counts().plot(kind='bar', ax=ax[1], color=['green', 'red'])
ax[1].set_title('Après ADASYN')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '03_rebalancing.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------
# 5. Entraînement des modèles
# -------------------------------
print("\n" + "="*70)
print("ENTRAÎNEMENT DES MODÈLES")
print("="*70)

models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=10,
        min_samples_leaf=4, class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'XGBoost': XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        eval_metric='logloss',
        random_state=42
    )
}

trained_models = {}
validation_scores = {}

for name, model in models.items():
    print(f"\n🔧 Entraînement : {name}...")
    model.fit(X_train_resampled, y_train_resampled)
    trained_models[name] = model
    cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, cv=5, scoring='f1', n_jobs=-1)
    validation_scores[name] = cv_scores
    print(f"   ✓ CV F1-Score: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")

# -------------------------------
# 6. Sélection du meilleur modèle sur validation
# -------------------------------
print("\n" + "="*70)
print("SÉLECTION DU MEILLEUR MODÈLE")
print("="*70)

best_model_name = None
best_f1 = 0
for name, model in trained_models.items():
    y_val_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_val_pred)
    print(f"{name}: F1-Score = {f1:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name

print(f"\n🏆 Meilleur modèle : {best_model_name} (F1={best_f1:.4f})")
model = trained_models[best_model_name]

# -------------------------------
# 7. Optimisation du seuil de décision
# -------------------------------
print("\n" + "="*70)
print("OPTIMISATION DU SEUIL DE DÉCISION")
print("="*70)

y_pred_proba = model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.2, 0.8, 0.01)
metrics_by_threshold = []

for t in thresholds:
    y_pred_temp = (y_pred_proba >= t).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_temp).ravel()
    
    metrics_by_threshold.append({
        'threshold': t,
        'accuracy': accuracy_score(y_test, y_pred_temp),
        'recall': tp / (tp + fn),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'f1': f1_score(y_test, y_pred_temp),
        'specificity': tn / (tn + fp),
        'fp': fp,
        'fn': fn
    })

metrics_df = pd.DataFrame(metrics_by_threshold)

# Trouver le meilleur seuil (maximise F1 avec recall >= 0.75)
valid_thresholds = metrics_df[metrics_df['recall'] >= 0.75]
if len(valid_thresholds) > 0:
    best_idx = valid_thresholds['f1'].idxmax()
    best_threshold = metrics_df.loc[best_idx, 'threshold']
else:
    best_idx = metrics_df['f1'].idxmax()
    best_threshold = metrics_df.loc[best_idx, 'threshold']

print(f"\n✓ Seuil optimal trouvé : {best_threshold:.3f}")
print(f"  Recall : {metrics_df.loc[best_idx, 'recall']:.3f}")
print(f"  Precision : {metrics_df.loc[best_idx, 'precision']:.3f}")
print(f"  F1-Score : {metrics_df.loc[best_idx, 'f1']:.3f}")
print(f"  Faux Positifs : {int(metrics_df.loc[best_idx, 'fp'])}")
print(f"  Faux Négatifs : {int(metrics_df.loc[best_idx, 'fn'])}")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0,0].plot(metrics_df['threshold'], metrics_df['recall'], label='Recall', linewidth=2)
axes[0,0].plot(metrics_df['threshold'], metrics_df['precision'], label='Precision', linewidth=2)
axes[0,0].axvline(best_threshold, color='red', linestyle='--', label=f'Optimal={best_threshold:.2f}')
axes[0,0].set_xlabel('Seuil')
axes[0,0].set_ylabel('Score')
axes[0,0].set_title('Precision vs Recall')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(metrics_df['threshold'], metrics_df['f1'], linewidth=2, color='purple')
axes[0,1].axvline(best_threshold, color='red', linestyle='--', label=f'Optimal={best_threshold:.2f}')
axes[0,1].set_xlabel('Seuil')
axes[0,1].set_ylabel('F1-Score')
axes[0,1].set_title('F1-Score par seuil')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(metrics_df['threshold'], metrics_df['fp'], label='Faux Positifs', color='orange', linewidth=2)
axes[1,0].plot(metrics_df['threshold'], metrics_df['fn'], label='Faux Négatifs', color='red', linewidth=2)
axes[1,0].axvline(best_threshold, color='green', linestyle='--', label=f'Optimal={best_threshold:.2f}')
axes[1,0].set_xlabel('Seuil')
axes[1,0].set_ylabel('Nombre d\'erreurs')
axes[1,0].set_title('Erreurs par seuil')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(metrics_df['recall'], metrics_df['precision'], linewidth=2)
axes[1,1].scatter(metrics_df.loc[best_idx, 'recall'], 
                 metrics_df.loc[best_idx, 'precision'], 
                 color='red', s=100, zorder=5, label='Optimal')
axes[1,1].set_xlabel('Recall')
axes[1,1].set_ylabel('Precision')
axes[1,1].set_title('Courbe Precision-Recall')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '04_threshold_optimization.png', dpi=300, bbox_inches='tight')
plt.show()

y_pred = (y_pred_proba >= best_threshold).astype(int)

# -------------------------------
# 8. Évaluation complète du modèle
# -------------------------------
print("\n" + "="*70)
print("ÉVALUATION FINALE DU MODÈLE")
print("="*70)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
mcc = matthews_corrcoef(y_test, y_pred)

print(f"\n MÉTRIQUES GLOBALES")
print(f"  Accuracy : {accuracy:.4f}")
print(f"  F1-Score : {f1:.4f}")
print(f"  AUC-ROC : {auc:.4f}")
print(f"  Matthews Correlation Coefficient : {mcc:.4f}")

print("\n" + classification_report(y_test, y_pred, 
                                   target_names=['Non-diabétique', 'Diabétique/À risque'],
                                   digits=4))

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n MATRICE DE CONFUSION")
print(f"  Vrais Négatifs (TN) : {tn:,}")
print(f"  Faux Positifs (FP) : {fp:,}")
print(f"  Faux Négatifs (FN) : {fn:,}")
print(f"  Vrais Positifs (TP) : {tp:,}")
print(f"\n  Sensibilité (Recall) : {tp/(tp+fn):.4f}")
print(f"  Spécificité : {tn/(tn+fp):.4f}")
print(f"  Précision : {tp/(tp+fp):.4f}")

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-diabétique', 'Diabétique'],
            yticklabels=['Non-diabétique', 'Diabétique'],
            cbar_kws={'label': 'Nombre de cas'})
plt.xlabel("Prédiction", fontsize=12)
plt.ylabel("Réalité", fontsize=12)
plt.title(f'Matrice de Confusion - {best_model_name}\n(Seuil optimisé: {best_threshold:.3f})', fontsize=14)

for i in range(2):
    for j in range(2):
        percentage = cm[i, j] / cm[i].sum() * 100
        plt.text(j+0.5, i+0.7, f'({percentage:.1f}%)', 
                ha='center', va='center', color='red', fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '05_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------
# 9. Courbes ROC et Precision-Recall
# -------------------------------
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Courbe ROC
axes[0].plot(fpr, tpr, linewidth=2, label=f'{best_model_name} (AUC = {auc:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aléatoire (AUC = 0.5)')
axes[0].set_xlabel('Taux de Faux Positifs (FPR)', fontsize=11)
axes[0].set_ylabel('Taux de Vrais Positifs (TPR)', fontsize=11)
axes[0].set_title('Courbe ROC', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Courbe Precision-Recall
axes[1].plot(recall, precision, linewidth=2, label=f'{best_model_name}')
axes[1].axhline(y=y_test.mean(), color='r', linestyle='--', 
               label=f'Baseline ({y_test.mean():.3f})')
axes[1].set_xlabel('Recall', fontsize=11)
axes[1].set_ylabel('Precision', fontsize=11)
axes[1].set_title('Courbe Precision-Recall', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '06_roc_pr_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------
# 10. Feature Importance
# -------------------------------
print("\n" + "="*70)
print("IMPORTANCE DES VARIABLES")
print("="*70)

feature_names = X.columns if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(X.shape[1])]
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 Top 15 variables les plus importantes :")
print(feature_importance.head(15).to_string(index=False))

plt.figure(figsize=(12, 8))
top_n = 15
top_features = feature_importance.head(top_n)
plt.barh(range(top_n), top_features['Importance'], color='steelblue')
plt.yticks(range(top_n), top_features['Feature'])
plt.xlabel('Importance', fontsize=11)
plt.ylabel('Variable', fontsize=11)
plt.title(f'Top {top_n} Variables les Plus Importantes', fontsize=13)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(FIGURES_DIR / '07_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------
# 11. Courbe d'apprentissage
# -------------------------------
print("\n" + "="*70)
print("ANALYSE DE LA COURBE D'APPRENTISSAGE")
print("="*70)

from sklearn.metrics import make_scorer, f1_score

# Créer un scorer F1 sûr qui gère zero_division
custom_scorer = make_scorer(f1_score, zero_division=0)

train_sizes = np.linspace(0.3, 1.0, 8)  

try:
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_train_resampled, y_train_resampled, 
        cv=3,  #  3 folds pour plus de stabilité
        scoring=custom_scorer, 
        n_jobs=-1,
        train_sizes=train_sizes,
        shuffle=True,
        random_state=42
    )
    
    train_mean = np.nanmean(train_scores, axis=1)
    train_std = np.nanstd(train_scores, axis=1)
    val_mean = np.nanmean(val_scores, axis=1)
    val_std = np.nanstd(val_scores, axis=1)
    
    plt.figure(figsize=(12, 7))
    plt.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Score Train')
    plt.plot(train_sizes_abs, val_mean, 'o-', color='green', label='Score Validation')
    plt.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.1, color='green')
    plt.xlabel('Taille du set d\'entraînement', fontsize=11)
    plt.ylabel('F1-Score', fontsize=11)
    plt.title('Courbe d\'Apprentissage', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '08_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Courbe d'apprentissage générée avec succès")
except Exception as e:
    print(f"⚠ Erreur lors de la génération de la courbe d'apprentissage: {e}")
    print("  Cette étape a été ignorée pour continuer l'analyse")

# -------------------------------
# 12. Analyse des erreurs
# -------------------------------
print("\n" + "="*70)
print("ANALYSE DES ERREURS")
print("="*70)

false_positives_idx = np.where((y_test.values == 0) & (y_pred == 1))[0]
false_negatives_idx = np.where((y_test.values == 1) & (y_pred == 0))[0]

print(f"\n Faux Positifs : {len(false_positives_idx)} cas")
print(f" Faux Négatifs : {len(false_negatives_idx)} cas")

errors_analysis = pd.DataFrame({
    'Type': ['Faux Positif']*len(false_positives_idx) + ['Faux Négatif']*len(false_negatives_idx),
    'Probabilité': np.concatenate([
        y_pred_proba[false_positives_idx],
        y_pred_proba[false_negatives_idx]
    ])
})

plt.figure(figsize=(12, 6))
sns.boxplot(data=errors_analysis, x='Type', y='Probabilité', palette=['orange', 'red'])
plt.axhline(y=best_threshold, color='green', linestyle='--', linewidth=2, label=f'Seuil optimal ({best_threshold:.2f})')
plt.title('Distribution des Probabilités pour les Erreurs de Classification', fontsize=13)
plt.ylabel('Probabilité prédite', fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / '09_errors_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# -------------------------------
# 13. Export des résultats
# -------------------------------
print("\n" + "="*70)
print("EXPORT DES RÉSULTATS")
print("="*70)

joblib.dump(model, MODELS_DIR / f'best_model_{best_model_name}.pkl')
joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
print(f"✓ Modèle sauvegardé : {MODELS_DIR}/best_model_{best_model_name}.pkl")
print(f"✓ Scaler sauvegardé : {MODELS_DIR}/scaler.pkl")

metrics_summary = {
    'model_name': best_model_name,
    'optimal_threshold': float(best_threshold),
    'accuracy': float(accuracy),
    'f1_score': float(f1),
    'auc_roc': float(auc),
    'mcc': float(mcc),
    'sensitivity': float(tp/(tp+fn)),
    'specificity': float(tn/(tn+fp)),
    'precision': float(tp/(tp+fp)),
    'confusion_matrix': {
        'TN': int(tn), 'FP': int(fp),
        'FN': int(fn), 'TP': int(tp)
    },
    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}

with open(MODELS_DIR / 'metrics_summary.json', 'w') as f:
    json.dump(metrics_summary, f, indent=4)
print(f"✓ Métriques sauvegardées : {MODELS_DIR}/metrics_summary.json")

feature_importance.to_csv(MODELS_DIR / 'feature_importance.csv', index=False)
print(f"✓ Feature importance sauvegardée : {MODELS_DIR}/feature_importance.csv")

metrics_df.to_csv(MODELS_DIR / 'threshold_metrics.csv', index=False)
print(f"✓ Métriques par seuil sauvegardées : {MODELS_DIR}/threshold_metrics.csv")

print("\n" + "="*70)
print(" ANALYSE TERMINÉE AVEC SUCCÈS")
print("="*70)
print(f"\n📁 Tous les résultats sont dans :")
print(f"  • Modèles : {MODELS_DIR}")
print(f"  • Figures : {FIGURES_DIR}")


# -------------------------------
# 11. Prédiction Interactive
# -------------------------------

def predict_new_person(model, scaler, feature_names, threshold):
    """
    Demande les données d'une personne à l'utilisateur et prédit le risque
    de diabète en utilisant le modèle entraîné.
    """
    print("\n" + "="*70)
    print("DEMANDE DE PRÉDICTION INDIVIDUELLE")
    print("="*70)
    print("Veuillez entrer les informations demandées. Suivez les types (1 ou 0, Flottant).")
    
    features_info = {
        'HighBP': {'type': int, 'range': [0, 1], 'desc': 'Tension artérielle élevée (1=Oui, 0=Non)'},
        'HighChol': {'type': int, 'range': [0, 1], 'desc': 'Taux de cholestérol élevé (1=Oui, 0=Non)'},
        'CholCheck': {'type': int, 'range': [0, 1], 'desc': 'Contrôle du cholestérol (1=Oui, 0=Non)'},
        'BMI': {'type': float, 'range': [12.0, 70.0], 'desc': 'Indice de Masse Corporelle (ex: 25.5)'},
        'Smoker': {'type': int, 'range': [0, 1], 'desc': 'Fumeur (1=Oui, 0=Non)'},
        'Stroke': {'type': int, 'range': [0, 1], 'desc': 'Antécédent d\'AVC (1=Oui, 0=Non)'},
        'HeartDiseaseorAttack': {'type': int, 'range': [0, 1], 'desc': 'Maladie cardiaque/crise (1=Oui, 0=Non)'},
        'PhysActivity': {'type': int, 'range': [0, 1], 'desc': 'Activité physique (1=Oui, 0=Non)'},
        'Fruits': {'type': int, 'range': [0, 1], 'desc': 'Consommation de fruits (1=Oui, 0=Non)'},
        'Veggies': {'type': int, 'range': [0, 1], 'desc': 'Consommation de légumes (1=Oui, 0=Non)'},
        'HeavyAlcoholConsumption': {'type': int, 'range': [0, 1], 'desc': 'Consommation d\'alcool excessive (1=Oui, 0=Non)'},
        'AnyHealthcare': {'type': int, 'range': [0, 1], 'desc': 'Couverture santé (1=Oui, 0=Non)'},
        'NoDocbcCost': {'type': int, 'range': [0, 1], 'desc': 'Pas de médecin pour raison de coût (1=Oui, 0=Non)'},
        'GenHlth': {'type': float, 'range': [1.0, 5.0], 'desc': 'Santé générale (1=Excellent, 5=Mauvais)'},
        'MentHlth': {'type': float, 'range': [0.0, 30.0], 'desc': 'Jours de mauvaise santé mentale (0-30)'},
        'PhysHlth': {'type': float, 'range': [0.0, 30.0], 'desc': 'Jours de mauvaise santé physique (0-30)'},
        'DiffWalk': {'type': int, 'range': [0, 1], 'desc': 'Difficulté à marcher/se vêtir (1=Oui, 0=Non)'},
        'Sex': {'type': int, 'range': [0, 1], 'desc': 'Sexe (1=Homme, 0=Femme)'},
        'Age': {'type': float, 'range': [1.0, 13.0], 'desc': 'Catégorie d\'âge (1=18-24, 13=80+)'},
        'Education': {'type': float, 'range': [1.0, 6.0], 'desc': 'Niveau d\'éducation (1=Jamais allé, 6=Diplômé universitaire)'},
        'Income': {'type': float, 'range': [1.0, 8.0], 'desc': 'Catégorie de revenu (1=<10k, 8=>75k)'}
    }
    
    input_data = {}
    
    for feature in feature_names:
        info = features_info.get(feature, {'type': float, 'desc': f'Entrez {feature}'})
        data_type = info['type']
        
        while True:
            try:
                user_input = input(f"[{data_type.__name__}] {info['desc']} ({feature}): ")
                value = data_type(user_input)
                
                if 'range' in info:
                    min_val, max_val = info['range']
                    if not (min_val <= value <= max_val):
                        print(f"⚠️ Erreur: La valeur doit être entre {min_val} et {max_val}.")
                        continue

                input_data[feature] = value
                break
            except ValueError:
                print(f"⚠️ Erreur: Entrée invalide. Veuillez entrer un(e) {data_type.__name__} valide.")
    
    new_data_df = pd.DataFrame([input_data])
    
    new_data_scaled = scaler.transform(new_data_df)
    
    probability = model.predict_proba(new_data_scaled)[:, 1][0]
    prediction = 1 if probability >= threshold else 0
    
    print("\n" + "="*70)
    print("RÉSULTAT DE LA PRÉDICTION")
    print("="*70)
    print(f"Probabilité de Diabète (P(Diabète=1)) : {probability:.4f}")
    print(f"Seuil de Classification utilisé : {threshold:.3f}")
    
    if prediction == 1:
        print("\n**CONCLUSION :DIABÉTIQUE")
    else:
        print("\n**CONCLUSION :NON-DIABÉTIQUE")
    
    print("\n" + "="*70)


# ----------------------------------------------------------------------
# EXÉCUTION DE LA PRÉDICTION INTERACTIVE APRÈS TOUT L'ENTRAÎNEMENT
# ----------------------------------------------------------------------

try:
    loaded_model = joblib.load(MODELS_DIR / f'best_model_{best_model_name}.pkl')
    loaded_scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
    
    with open(MODELS_DIR / 'metrics_summary.json', 'r') as f:
        metrics_summary = json.load(f)
    optimal_threshold = metrics_summary['optimal_threshold']
    
    predict_new_person(
        model=loaded_model, 
        scaler=loaded_scaler, 
        feature_names=feature_names, 
        threshold=optimal_threshold
    )
    
except FileNotFoundError:
    print(f"\n ERREUR: Fichiers de modèle/scaler non trouvés dans {MODELS_DIR}/. Veuillez vérifier que les sections 1-10 ont été exécutées avec succès.")
except NameError:
    print("\n ERREUR: Variables non définies (e.g., best_model_name). Assurez-vous d'exécuter l'intégralité du script.")

print("\n" + "="*70)
print("FIN DU SCRIPT")
print("="*70)