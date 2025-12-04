from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import joblib
from src.config import MODELS_DIR, RANDOM_SEED

def define_and_train_models(X_train_resampled, y_train_resampled):
    """Définit et entraîne les modèles de classification."""
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=10,
            min_samples_leaf=4, class_weight='balanced', 
            random_state=RANDOM_SEED, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1, 
            subsample=0.8, eval_metric='logloss', 
            random_state=RANDOM_SEED, n_jobs=-1
        )
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n Entraînement : {name}...")
        model.fit(X_train_resampled, y_train_resampled)
        trained_models[name] = model
        
        cv_scores = cross_val_score(model, X_train_resampled, y_train_resampled, 
                                    cv=5, scoring='f1', n_jobs=-1)
        print(f"    CV F1-Score: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        
    return trained_models

def select_and_save_best_model(trained_models, X_val, y_val):
    """Sélectionne le meilleur modèle sur l'ensemble de validation (F1-Score)."""
    print("\n" + "="*70)
    print("SÉLECTION DU MEILLEUR MODÈLE")
    print("="*70)
    
    best_model_name = None
    best_f1 = 0
    
    for name, model in trained_models.items():
        y_val_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_val_pred)
        print(f"{name} (Val): F1-Score = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name

    best_model = trained_models[best_model_name]
    joblib.dump(best_model, MODELS_DIR / f'best_model_{best_model_name}.pkl')
    print(f"\n Meilleur modèle sélectionné et sauvegardé : {best_model_name}")
    
    return best_model, best_model_name