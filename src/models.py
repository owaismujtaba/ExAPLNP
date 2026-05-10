import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib

from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix




class ModelTrainer:
    """Trains and evaluates multiple machine learning models on a given dataset.
    This class manages model selection, cross-validation, metric calculation, and model persistence.
    """
    def __init__(self, config, logger=None):
        """Initializes the ModelTrainer with configuration and logger.
        Sets up the available machine learning models and ensemble methods for training and evaluation.

        Args:
            config (dict): Configuration dictionary for model training and evaluation.
            logger (logging.Logger, optional): Logger for status and progress messages.
        """
        self.config = config
        self.logger = logger
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "LDA": LinearDiscriminantAnalysis(),
            "Decision Tree": DecisionTreeClassifier(class_weight="balanced"),
            "Random Forest": RandomForestClassifier(class_weight="balanced"),
            "Gradient Boosting": GradientBoostingClassifier(),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", scale_pos_weight=1),
            "LightGBM": LGBMClassifier(class_weight="balanced"),
            "CatBoost": CatBoostClassifier(verbose=0, scale_pos_weight=1),
            "SVM (Linear)": SVC(kernel="linear", probability=True, class_weight="balanced"),
            "SVM (RBF)": SVC(kernel="rbf", probability=True, class_weight="balanced"),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "Neural Network": MLPClassifier(max_iter=500),
            "Bagging": BaggingClassifier(),
        }

        # Ensembles
        ensembles = {
            "Voting Ensemble": VotingClassifier(
                estimators=[
                    ('lr', LogisticRegression(max_iter=1000, class_weight="balanced")),
                    ('rf', RandomForestClassifier(class_weight="balanced")),
                    ('gb', GradientBoostingClassifier())
                ],
                voting='soft'
            ),
            "Stacking Ensemble": StackingClassifier(
                estimators=[
                    ('rf', RandomForestClassifier(class_weight="balanced")),
                    ('gb', GradientBoostingClassifier())
                ],
                final_estimator=LogisticRegression(class_weight="balanced")
            )
        }

        self.models.update(ensembles)

    def get_models1(self, X, y):
        output_dir = self.config['model']['results_dir']
        os.makedirs(output_dir, exist_ok=True)
        # KFold setup
        self.kf = KFold(n_splits=self.config['model']['n_folds'], shuffle=True, random_state=42)
        
        smote = SMOTE()
        X, y = smote.fit_resample(X, y)

        results = []  
        summary = []  
        cms = []

        for model_name, model in self.models.items():
            fold_metrics = []
            self.logger.info(f"Training and evaluating model: {model_name}")
            cm = []
            for fold, (train_idx, test_idx) in enumerate(self.kf.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Use class weights or sample weights for imbalanced data
                sample_weight = None
                if hasattr(model, "fit") and "sample_weight" in model.fit.__code__.co_varnames:
                    # Compute sample weights inversely proportional to class frequencies
                    classes = np.unique(y_train)
                    class_weights = {c: len(y_train) / (len(classes) * np.sum(y_train == c)) for c in classes}
                    sample_weight = np.array([class_weights[yi] for yi in y_train])
                    model.fit(X_train, y_train, sample_weight=sample_weight)
                else:
                    model.fit(X_train, y_train)

                self.save_model(model, model_name, fold+1)
                y_pred = model.predict(X_test)

                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = model.decision_function(X_test)
                    y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_prob)
                cm = confusion_matrix(y_test, y_prob)
                cms.append([model_name, cm])

                results.append([model_name, fold+1, acc, prec, rec, f1, auc])
                fold_metrics.append([acc, prec, rec, f1, auc])
                self.logger.info(f"Fold {fold+1} -")
                self.logger.info(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}, ROC AUC: {auc:.4f}")

            fold_metrics = np.array(fold_metrics)
            mean_acc, mean_prec, mean_rec, mean_f1, mean_auc = fold_metrics.mean(axis=0)
            summary.append([model_name, mean_acc, mean_prec, mean_rec, mean_f1, mean_auc])


        results_df = pd.DataFrame(results,
                                columns=["Model", "Fold", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"])

        summary_df = pd.DataFrame(summary,
                                columns=["Model", "Mean Accuracy", "Mean Precision", "Mean Recall", "Mean F1 Score", "Mean ROC AUC"])

        summary_df = summary_df.sort_values(by="Mean ROC AUC", ascending=False)
        
        results_df.to_csv(f"{output_dir}/fold_wise_results.csv", index=False)
        summary_df.to_csv(f"{output_dir}/summary_results.csv", index=False)
        print(f"Results saved to {output_dir}")


    def get_models(self, X, y):
        output_dir = self.config['model']['results_dir']
        os.makedirs(output_dir, exist_ok=True)

        self.kf = KFold(
            n_splits=self.config['model']['n_folds'],
            shuffle=True,
            random_state=42
        )

        results = []
        summary = []

        smote = SMOTE()

        for model_name, model in self.models.items():
            fold_metrics = []
            self.logger.info(f"Training and evaluating model: {model_name}")

            for fold, (train_idx, test_idx) in enumerate(self.kf.split(X)):
                # Split
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                X_train, y_train = smote.fit_resample(X_train, y_train)

                # Scaling
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Optional: sample weights for imbalance
                sample_weight = None
                if hasattr(model, "fit") and "sample_weight" in model.fit.__code__.co_varnames:
                    classes = np.unique(y_train)
                    class_weights = {
                        c: len(y_train) / (len(classes) * np.sum(y_train == c))
                        for c in classes
                    }
                    sample_weight = np.array([class_weights[yi] for yi in y_train])
                    model.fit(X_train, y_train, sample_weight=sample_weight)
                else:
                    model.fit(X_train, y_train)

                # Save model
                self.save_model(model, model_name, fold + 1)

                # Predictions
                y_pred = model.predict(X_test)
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_prob = model.decision_function(X_test)
                    y_prob = (y_prob - y_prob.min()) / (y_prob.max() - y_prob.min())

                # Metrics
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_prob)

                # ✅ Correct confusion matrix
                cm = confusion_matrix(y_test, y_pred)

                # ✅ Save confusion matrix to file
                cm_dir = f"{output_dir}/confusion_matrices/"
                os.makedirs(cm_dir, exist_ok=True)

                safe_name = model_name.replace(" ", "_")
                cm_path = f"{cm_dir}/{safe_name}_fold{fold+1}_cm.csv"
                pd.DataFrame(cm).to_csv(cm_path, index=False)

                self.logger.info(f"Saved confusion matrix for {model_name} fold {fold+1} at {cm_path}")

                # Store results
                results.append([model_name, fold + 1, acc, prec, rec, f1, auc])
                fold_metrics.append([acc, prec, rec, f1, auc])

                self.logger.info(
                    f"Fold {fold+1} - "
                    f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, "
                    f"F1 Score: {f1:.4f}, ROC AUC: {auc:.4f}"
                )

            # Average fold results
            fold_metrics = np.array(fold_metrics)
            mean_acc, mean_prec, mean_rec, mean_f1, mean_auc = fold_metrics.mean(axis=0)
            summary.append([model_name, mean_acc, mean_prec, mean_rec, mean_f1, mean_auc])

        # Save final CSVs
        results_df = pd.DataFrame(
            results,
            columns=["Model", "Fold", "Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
        )

        summary_df = pd.DataFrame(
            summary,
            columns=["Model", "Mean Accuracy", "Mean Precision", "Mean Recall", "Mean F1 Score", "Mean ROC AUC"]
        ).sort_values(by="Mean ROC AUC", ascending=False)

        results_df.to_csv(f"{output_dir}/fold_wise_results.csv", index=False)
        summary_df.sort_values(by='Mean ROC AUC')
        summary_df.to_csv(f"{output_dir}/summary_results.csv", index=False)

        print(f"Results saved to {output_dir}")


    def save_model(self, model, model_name, fold):
        output_dir = self.config['model']['results_dir']
        output_dir = f"{output_dir}/models/"
        os.makedirs(output_dir, exist_ok=True)

        safe_name = model_name.replace(" ", "_")
        file_path = os.path.join(output_dir, f"{safe_name}_fold{fold}.pkl")

        joblib.dump(model, file_path)
        print(f"Saved: {file_path}")