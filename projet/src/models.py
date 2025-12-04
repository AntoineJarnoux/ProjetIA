from typing import Dict

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer

from .config import RANDOM_STATE


class ModelFactory:
    """
    Fabrique les modèles de classification à partir d'un préprocesseur.
    """

    def create_models(self, preprocessor: ColumnTransformer) -> Dict[str, Pipeline]:
        logreg_clf = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ])

        rf_clf = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                random_state=RANDOM_STATE
            ))
        ])

        return {
            "Logistic Regression": logreg_clf,
            "Random Forest": rf_clf
        }
