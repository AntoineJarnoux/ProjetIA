import pandas as pd

from .data_loader import HeartDataLoader
from .splitter import DataSplitter
from .preprocessor import FeaturePreprocessor
from .models import ModelFactory
from .evaluator import ModelEvaluator


def main():
    # 1) Chargement des données
    loader = HeartDataLoader()
    df: pd.DataFrame = loader.load_data()

    print("Aperçu du dataset :")
    print(df.head())
    print("\nInfo :")
    print(df.info())
    print("\nStatistiques descriptives :")
    print(df.describe())
    print("\nRépartition de la cible :")
    print(df["target"].value_counts())

    # 2) X / y
    target_col = "target"
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    feature_names = X.columns.tolist()
    print("\nNombre de features :", len(feature_names))
    print("Features :", feature_names)

    # 3) Split train / val / test
    splitter = DataSplitter(test_size=0.15, val_size=0.15)
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(X, y)

    print("\nTailles des ensembles :")
    print("Train :", X_train.shape)
    print("Validation :", X_val.shape)
    print("Test :", X_test.shape)

    # 4) Préprocesseur
    preprocessor_builder = FeaturePreprocessor()
    preprocessor = preprocessor_builder.build(feature_names)

    # 5) Modèles
    factory = ModelFactory()
    models = factory.create_models(preprocessor)

    evaluator = ModelEvaluator()

    # 6) Validation : on entraîne et évalue chaque modèle
    results_val = []
    for name, model in models.items():
        res = evaluator.evaluate_on_validation(
            name, model, X_train, y_train, X_val, y_val, plot_roc=True
        )
        results_val.append(res)

    # 7) Choix du meilleur modèle (sur F1)
    best_info = max(results_val, key=lambda r: r["f1"])
    best_model = best_info["model"]
    best_name = best_info["name"]

    print("\nMeilleur modèle sur validation (selon F1-score) :")
    print(f"{best_name} avec F1 = {best_info['f1']:.3f}")

    # 8) Évaluation finale sur test
    evaluator.evaluate_on_test(best_model, X_test, y_test, best_name, plot_roc=True)


if __name__ == "__main__":
    main()
