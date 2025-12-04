import joblib
from src.dataset import load_and_split_data
from src.features import build_features
from src.modeling.train import define_and_train_models, select_and_save_best_model
from src.modeling.predict import optimize_and_evaluate, predict_new_person
from src.config import MODELS_DIR, METRICS_PATH
import json

def run_pipeline():
    """Exécute l'intégralité du pipeline CCDS de bout en bout."""
    
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test, feature_names = load_and_split_data()
    
    X_train_res, y_train_res, X_val_scaled, X_test_scaled = build_features(
        X_train_raw, X_val_raw, X_test_raw, y_train
    )
    
    trained_models = define_and_train_models(X_train_res, y_train_res)
    best_model, best_model_name = select_and_save_best_model(
        trained_models, X_val_scaled, y_val
    )
    
    best_threshold, metrics_summary = optimize_and_evaluate(
        best_model, X_test_scaled, y_test, best_model_name, feature_names
    )
    
    print("\n" + "="*70)
    print(" ANALYSE TERMINÉE AVEC SUCCÈS")
    print("="*70)

    try:
        loaded_scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
        predict_new_person(
            model=best_model, 
            scaler=loaded_scaler, 
            feature_names=feature_names, 
            threshold=best_threshold
        )
    except Exception as e:
        print(f"\n ERREUR lors de l'exécution de la prédiction interactive : {e}")

if __name__ == "__main__":
    run_pipeline()