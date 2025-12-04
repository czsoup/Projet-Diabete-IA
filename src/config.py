import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if PROJECT_ROOT.name == 'src':
    PROJECT_ROOT = PROJECT_ROOT.parent # Remonter d'un niveau supplémentaire si on est bloqué dans src

#
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 

PROJECT_ROOT = Path(__file__).resolve().parents[2] 

PROJECT_ROOT = Path(__file__).resolve().parent.parent 

PROJECT_ROOT = Path(os.getcwd()) 

DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes2015.csv"

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_PATH = MODELS_DIR / 'metrics_summary.json'

for directory in [MODELS_DIR, FIGURES_DIR]:
    os.makedirs(directory, exist_ok=True)

RANDOM_SEED = 42

print(f" Racine du projet : {PROJECT_ROOT}")
print(f" Modèles sauvegardés dans : {MODELS_DIR}")
print(f" Graphiques sauvegardés dans : {FIGURES_DIR}")