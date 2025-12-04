from typing import Dict, Any

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


class ModelEvaluator:
    """
    Évalue les modèles sur validation et test, affiche les métriques
    et, si possible, trace la courbe ROC.
    """

    def evaluate_on_validation(
        self,
        name,
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        plot_roc: bool = True
    ) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"Modèle : {name}")
        print("="*60)

        # entraînement
        model.fit(X_train, y_train)

        # prédiction
        y_pred = model.predict(X_val)
        y_proba = None
        if hasattr(model.named_steps["model"], "predict_proba"):
            y_proba = model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        print(f"Accuracy : {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall   : {rec:.3f}")
        print(f"F1-score : {f1:.3f}\n")

        print("Classification report :")
        print(classification_report(y_val, y_pred))

        print("Matrice de confusion :")
        print(confusion_matrix(y_val, y_pred))

        if plot_roc and y_proba is not None:
            fpr, tpr, _ = roc_curve(y_val, y_proba)
            auc = roc_auc_score(y_val, y_proba)
            print(f"\nAUC-ROC : {auc:.3f}")

            plt.figure()
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC - {name} (validation)")
            plt.legend()
            plt.show()

        return {
            "name": name,
            "model": model,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }

    def evaluate_on_test(
        self,
        model,
        X_test,
        y_test,
        model_name: str,
        plot_roc: bool = True
    ) -> None:

        print("\nÉvaluation finale sur l'ensemble de TEST :")
        y_test_pred = model.predict(X_test)
        y_test_proba = None
        if hasattr(model.named_steps["model"], "predict_proba"):
            y_test_proba = model.predict_proba(X_test)[:, 1]

        acc_test = accuracy_score(y_test, y_test_pred)
        prec_test = precision_score(y_test, y_test_pred)
        rec_test = recall_score(y_test, y_test_pred)
        f1_test = f1_score(y_test, y_test_pred)

        print(f"Accuracy (test) : {acc_test:.3f}")
        print(f"Precision (test): {prec_test:.3f}")
        print(f"Recall (test)   : {rec_test:.3f}")
        print(f"F1-score (test) : {f1_test:.3f}\n")

        print("Classification report (test) :")
        print(classification_report(y_test, y_test_pred))

        print("Matrice de confusion (test) :")
        print(confusion_matrix(y_test, y_test_pred))

        if plot_roc and y_test_proba is not None:
            fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
            auc_test = roc_auc_score(y_test, y_test_proba)
            print(f"\nAUC-ROC (test) : {auc_test:.3f}")

            plt.figure()
            plt.plot(fpr_test, tpr_test, label=f"{model_name} (AUC = {auc_test:.3f})")
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC - {model_name} (test)")
            plt.legend()
            plt.show()
