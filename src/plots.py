# src/plots.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from src.config import FIGURES_DIR, MODELS_DIR

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def plot_correlation_matrix(df_raw, target_col='Diabetes_binary'):
    """Trace la matrice de corrélation pour les top 10 variables (Section 1)."""
    correlations = df_raw.corr()[target_col].sort_values(ascending=False)
    top_features = correlations.head(11).index[1:]
    correlation_matrix = df_raw[list(top_features) + [target_col]].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=1)
    plt.title('Matrice de corrélation - Top 10 variables')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '02_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 02_correlation_matrix.png générée.")


def plot_threshold_optimization(metrics_df, best_idx, best_threshold):
    """Trace les courbes d'optimisation du seuil (Section 7)."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Graphique [0,0]: Precision vs Recall
    axes[0,0].plot(metrics_df['threshold'], metrics_df['recall'], label='Recall', linewidth=2)
    axes[0,0].plot(metrics_df['threshold'], metrics_df['precision'], label='Precision', linewidth=2)
    axes[0,0].axvline(best_threshold, color='red', linestyle='--', label=f'Optimal={best_threshold:.2f}')
    axes[0,0].set_xlabel('Seuil')
    axes[0,0].set_ylabel('Score')
    axes[0,0].set_title('Precision vs Recall')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Graphique [0,1]: F1-Score par seuil
    axes[0,1].plot(metrics_df['threshold'], metrics_df['f1'], linewidth=2, color='purple')
    axes[0,1].axvline(best_threshold, color='red', linestyle='--', label=f'Optimal={best_threshold:.2f}')
    axes[0,1].set_xlabel('Seuil')
    axes[0,1].set_ylabel('F1-Score')
    axes[0,1].set_title('F1-Score par seuil')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)

    # Graphique [1,0]: Erreurs par seuil (FP vs FN)
    axes[1,0].plot(metrics_df['threshold'], metrics_df['fp'], label='Faux Positifs', color='orange', linewidth=2)
    axes[1,0].plot(metrics_df['threshold'], metrics_df['fn'], label='Faux Négatifs', color='red', linewidth=2)
    axes[1,0].axvline(best_threshold, color='green', linestyle='--', label=f'Optimal={best_threshold:.2f}')
    axes[1,0].set_xlabel('Seuil')
    axes[1,0].set_ylabel('Nombre d\'erreurs')
    axes[1,0].set_title('Erreurs par seuil')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)

    # Graphique [1,1]: Courbe Precision-Recall
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
    plt.close()
    print("✓ Figure 04_threshold_optimization.png générée.")


def plot_confusion_matrix(cm, model_name, best_threshold):
    """Trace la Matrice de Confusion (Section 8)."""
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-diabétique', 'Diabétique'],
                yticklabels=['Non-diabétique', 'Diabétique'],
                cbar_kws={'label': 'Nombre de cas'})
    plt.xlabel("Prédiction", fontsize=12)
    plt.ylabel("Réalité", fontsize=12)
    plt.title(f'Matrice de Confusion - {model_name}\n(Seuil optimisé: {best_threshold:.3f})', fontsize=14)

    # Ajout des pourcentages
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / cm[i].sum() * 100
            plt.text(j+0.5, i+0.7, f'({percentage:.1f}%)', 
                    ha='center', va='center', color='red', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '05_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 05_confusion_matrix.png générée.")


def plot_roc_pr_curves(y_test, y_pred_proba, auc, model_name):
    """Trace les Courbes ROC et Precision-Recall (Section 9)."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Courbe ROC
    axes[0].plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aléatoire (AUC = 0.5)')
    axes[0].set_xlabel('Taux de Faux Positifs (FPR)', fontsize=11)
    axes[0].set_ylabel('Taux de Vrais Positifs (TPR)', fontsize=11)
    axes[0].set_title('Courbe ROC', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Courbe Precision-Recall
    axes[1].plot(recall, precision, linewidth=2, label=f'{model_name}')
    axes[1].axhline(y=y_test.mean(), color='r', linestyle='--', 
                   label=f'Baseline ({y_test.mean():.3f})')
    axes[1].set_xlabel('Recall', fontsize=11)
    axes[1].set_ylabel('Precision', fontsize=11)
    axes[1].set_title('Courbe Precision-Recall', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '06_roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 06_roc_pr_curves.png générée.")


def plot_feature_importance(model, feature_names):
    """Trace l'Importance des Variables (Section 10)."""
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    feature_importance.to_csv(MODELS_DIR / 'feature_importance.csv', index=False)
    
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
    plt.close()
    print("✓ Figure 07_feature_importance.png générée.")


def plot_errors_analysis(y_test, y_pred, y_pred_proba, best_threshold):
    """Trace l'Analyse des Erreurs (Section 12)."""
    
    false_positives_idx = np.where((y_test == 0) & (y_pred == 1))[0]
    false_negatives_idx = np.where((y_test == 1) & (y_pred == 0))[0]
    
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
    plt.close()
    print("✓ Figure 09_errors_analysis.png générée.")

# NOTE: La courbe d'apprentissage (Section 11) doit être placée ici, 
# mais elle nécessite les données d'entraînement rééchantillonnées (X_train_resampled, y_train_resampled) 
# qui ne sont pas facilement accessibles dans predict.py. 
# Si elle est jugée essentielle, elle devrait être déplacée et appelée depuis train.py ou run_pipeline.py.
# ajoute la courbe d'apprentissage dis moi les etapes necessaire