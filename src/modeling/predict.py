# src/modeling/predict.py

import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    roc_auc_score, f1_score, matthews_corrcoef
)
from src.config import MODELS_DIR, METRICS_PATH
from src.plots import (
    plot_threshold_optimization, plot_confusion_matrix, 
    plot_roc_pr_curves, plot_feature_importance, plot_errors_analysis
) 

def optimize_and_evaluate(model, X_test, y_test, model_name, feature_names):
    """
    Optimise le seuil, évalue le modèle sur le set de test et exporte les résultats.
    (Reprend les sections 7 et 8 de l'ancien pipeline)
    """
    print("\n" + "="*70)
    print("ÉVALUATION FINALE DU MODÈLE ET REPORTING")
    print("="*70)
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # --- 1. Optimisation du seuil (Section 7) ---
    thresholds = np.arange(0.2, 0.8, 0.01)
    metrics_by_threshold = []

    for t in thresholds:
        y_pred_temp = (y_pred_proba >= t).astype(int)
        
        # Gestion des cas où la prédiction est uni-classe pour éviter une erreur .ravel()
        if len(np.unique(y_pred_temp)) < 2:
            cm_temp = confusion_matrix(y_test, y_pred_temp, labels=[0, 1])
            tn, fp, fn, tp = cm_temp.ravel()
        else:
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_temp).ravel()
            
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        metrics_by_threshold.append({
            'threshold': t,
            'accuracy': accuracy_score(y_test, y_pred_temp),
            'recall': recall,
            'precision': precision,
            'f1': f1_score(y_test, y_pred_temp, zero_division=0),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'fp': fp,
            'fn': fn
        })

    metrics_df = pd.DataFrame(metrics_by_threshold)
    metrics_df.to_csv(MODELS_DIR / 'threshold_metrics.csv', index=False)
    
    # Sélection du meilleur seuil (logique F1 max avec Recall >= 0.75 ou F1 max tout court)
    valid_thresholds = metrics_df[metrics_df['recall'] >= 0.75]
    best_idx = valid_thresholds['f1'].idxmax() if len(valid_thresholds) > 0 else metrics_df['f1'].idxmax()
    best_threshold = metrics_df.loc[best_idx, 'threshold']
    
    # Génération du graphique d'optimisation (via plots.py)
    plot_threshold_optimization(metrics_df, best_idx, best_threshold)
    
    # --- 2. Évaluation Finale avec Seuil Optimal (Section 8) ---
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    # Affichage et Plotting
    print(f"\n MÉTRIQUES FINALES (Seuil: {best_threshold:.3f})")
    print(f"  Accuracy : {accuracy:.4f} | F1-Score : {f1:.4f} | AUC-ROC : {auc:.4f} | MCC : {mcc:.4f}")
    print(classification_report(y_test, y_pred, 
                                target_names=['Non-diabétique', 'Diabétique/À risque'], 
                                digits=4))
    
    print(f"\n MATRICE DE CONFUSION")
    print(f"  Vrais Négatifs (TN) : {tn:,} | Faux Positifs (FP) : {fp:,}")
    print(f"  Faux Négatifs (FN) : {fn:,} | Vrais Positifs (TP) : {tp:,}")
    
    plot_confusion_matrix(cm, model_name, best_threshold)
    plot_roc_pr_curves(y_test, y_pred_proba, auc, model_name)
    if hasattr(model, 'feature_importances_'):
        plot_feature_importance(model, feature_names)

    # Analyse des erreurs (Section 12)
    plot_errors_analysis(y_test.values, y_pred, y_pred_proba, best_threshold)

    # --- 3. Export des résultats (Partie de Section 13) ---
    metrics_summary = {
        'model_name': model_name,
        'optimal_threshold': float(best_threshold),
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'auc_roc': float(auc),
        'mcc': float(mcc),
        'sensitivity': float(tp/(tp+fn)) if (tp+fn) > 0 else 0,
        'specificity': float(tn/(tn+fp)) if (tn+fp) > 0 else 0,
        'precision': float(tp/(tp+fp)) if (tp+fp) > 0 else 0,
        'confusion_matrix': {
            'TN': int(tn), 'FP': int(fp),
            'FN': int(fn), 'TP': int(tp)
        },
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"✓ Métriques sauvegardées : {METRICS_PATH}")
    
    return best_threshold, metrics_summary


def predict_new_person(model, scaler, feature_names, threshold):
    """
    Demande les données d'une personne à l'utilisateur et prédit le risque
    de diabète en utilisant le modèle entraîné. (Section 11 de l'ancien pipeline)
    """
    print("\n" + "="*70)
    print("DEMANDE DE PRÉDICTION INDIVIDUELLE")
    print("="*70)
    print("Veuillez entrer les informations demandées. Suivez les types (1 ou 0, Flottant).")
    
    # La définition des features_info est copiée intégralement de votre pipeline_old2.py
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
        print("\n**CONCLUSION : DIABÉTIQUE / À RISQUE")
    else:
        print("\n**CONCLUSION : NON-DIABÉTIQUE")
    
    print("\n" + "="*70)