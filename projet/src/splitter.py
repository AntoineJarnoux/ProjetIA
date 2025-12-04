from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE


class DataSplitter:
    """
    Gère la séparation du dataset en train / validation / test.
    """

    def __init__(self, test_size: float = 0.15, val_size: float = 0.15):
        self.test_size = test_size
        self.val_size = val_size

    def split(
        self,
        X: pd.DataFrame,
        y
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Retourne X_train, X_val, X_test, y_train, y_val, y_test
        avec un split 70 / 15 / 15 par défaut.
        """
        # d'abord on sépare le test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            stratify=y,
            random_state=RANDOM_STATE,
        )

        # puis on calcule la proportion de validation dans le reste
        val_ratio_in_temp = self.val_size / (1.0 - self.test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_ratio_in_temp,
            stratify=y_temp,
            random_state=RANDOM_STATE,
        )

        return X_train, X_val, X_test, y_train, y_val, y_test
