# src/dataset.py

import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import DATA_RAW_PATH, FIGURES_DIR, RANDOM_SEED

def load_and_split_data():
    
    if not DATA_RAW_PATH.exists():
        sys.exit(f" ERREUR CRITIQUE : Fichier introuvable : {DATA_RAW_PATH}")

    df = pd.read_csv(DATA_RAW_PATH)
    df['Diabetes_binary'] = df['Diabetes_012'].apply(lambda x: 1 if x >= 1 else 0)
    
    print("\n" + "="*70)
    print("ANALYSE EXPLORATOIRE DES DONNÉES & SPLIT")
    print("="*70)
    print(f"\n Taille du dataset : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")
    print(f" Taux de diabétiques : {df['Diabetes_binary'].mean()*100:.2f}%")

    X = df.drop(['Diabetes_012', 'Diabetes_binary'], axis=1)
    y = df['Diabetes_binary']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df['Diabetes_012'].value_counts().plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Distribution Diabetes_012')
    df['Diabetes_binary'].value_counts().plot(kind='bar', ax=axes[1], color=['green', 'red'])
    axes[1].set_title('Distribution Binaire (0: Sain, 1: À risque)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / '01_distribution_classes.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_SEED, stratify=y_train_full
    )

    print(f"Train set : {X_train.shape[0]:,} échantillons (60%)")
    print(f"Validation set : {X_val.shape[0]:,} échantillons (20%)")
    print(f"Test set : {X_test.shape[0]:,} échantillons (20%)")

    return X_train, X_val, X_test, y_train, y_val, y_test, X.columns.tolist()