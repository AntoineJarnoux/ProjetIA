from typing import List

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


class FeaturePreprocessor:
    """
    Construit le préprocesseur (ici StandardScaler sur toutes les features numériques).
    """

    def build(self, feature_names: List[str]) -> ColumnTransformer:
        numeric_features = feature_names

        numeric_transformer = Pipeline(
            steps=[("scaler", StandardScaler())]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features)
            ]
        )

        return preprocessor
