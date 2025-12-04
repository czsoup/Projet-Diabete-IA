import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYN
from collections import Counter
import joblib
import matplotlib.pyplot as plt
from src.config import MODELS_DIR, FIGURES_DIR, RANDOM_SEED

def build_features(X_train, X_val, X_test, y_train):
    """
    Applique le Scaling et le Sur-échantillonnage ADASYN.
    Retourne les ensembles scalés/rééchantillonnés.
    """
    print("\n" + "="*70)
    print("INGÉNIERIE DES VARIABLES (Scaling & ADASYN)")
    print("="*70)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    print("✓ Scaler (StandardScaler) sauvegardé.")

    print(f"\nDistribution avant ADASYN: {Counter(y_train)}")
    adasyn = ADASYN(random_state=RANDOM_SEED, n_neighbors=5)
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_scaled, y_train)
    print(f"Distribution après ADASYN: {Counter(y_train_resampled)}")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    pd.Series(y_train).value_counts().plot(kind='bar', ax=ax[0], color=['green', 'red'])
    ax[0].set_title('Avant ADASYN')
    pd.Series(y_train_resampled).value_counts().plot(kind='bar', ax=ax[1], color=['green', 'red'])
    ax[1].set_title('Après ADASYN')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '03_rebalancing.png', dpi=300, bbox_inches='tight')
    plt.close()

    return X_train_resampled, y_train_resampled, X_val_scaled, X_test_scaled