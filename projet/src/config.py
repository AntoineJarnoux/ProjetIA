from pathlib import Path
import numpy as np

# Racine du projet = dossier parent de src
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Dossier data et chemin du CSV
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "heart.csv"

# Pour la reproductibilité
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
