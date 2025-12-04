import pandas as pd
from pathlib import Path

from .config import DATA_PATH, DATA_DIR


class HeartDataLoader:
    """
    Classe responsable du chargement (et éventuellement du téléchargement)
    des données de maladie cardiaque.
    """

    def __init__(self):
        # on s'assure que le dossier data existe
        DATA_DIR.mkdir(exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """
        Charge les données depuis data/heart.csv.
        """
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Fichier {DATA_PATH} introuvable. "
                f"Place le fichier heart.csv dans {DATA_DIR}."
            )

        df = pd.read_csv(DATA_PATH)
        return df
