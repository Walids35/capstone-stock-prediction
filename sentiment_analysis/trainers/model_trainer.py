import os
import joblib
import json
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from stock_prediction.config import PROJ_ROOT
from loguru import logger

class ModelTrainer:
    def __init__(self, results_df, x_cols, target_col, use_grid_search=False):
        """
        Initializes the ModelTrainer with data and configuration.

        Args:
            results_df (pd.DataFrame): Your dataset.
            x_cols (list): Feature column names.
            target_col (str): Target column name.
            use_grid_search (bool): Whether to use grid search for SVM and Random Forest.
        """
        self.X = results_df[x_cols]
        self.y = results_df[target_col]
        self.save_dir = PROJ_ROOT / "models"
        self.use_grid_search = use_grid_search
        self.results = {}

        if not os.path.exists(PROJ_ROOT / "models"):
            os.makedirs(PROJ_ROOT / "models")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def _save_results_to_file(self, model_name, best_params, metrics, cv_score=None):
        """
        Save model results to a JSON file.
        
        Args:
            model_name (str): Name of the model
            best_params (dict): Best parameters found
            metrics (dict): Model performance metrics
            cv_score (float): Cross-validation score if available
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_results_{timestamp}.json"
        filepath = os.path.join(self.save_dir, filename)
        
        results_data = {
            "model_name": model_name,
            "timestamp": timestamp,
            "best_parameters": best_params,
            "metrics": metrics,
            "cv_score": cv_score,
            "use_grid_search": self.use_grid_search
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=4, default=str)
        
        logger.success(f"Results saved to: {filepath}")
        return filepath

    def _extract_metrics(self, y_true, y_pred, model_name):
        """
        Extract comprehensive metrics from predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name (str): Name of the model
            
        Returns:
            dict: Dictionary containing all metrics
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Get detailed classification report
        report = classification_report(y_true, y_pred, target_names=['Negative', 'Neutral', 'Positive'], output_dict=True)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support.tolist() if hasattr(support, 'tolist') else support,
            "detailed_report": report
        }
        
        return metrics

    def train_logistic_regression(self):
        logger.info("\nTraining Logistic Regression...")
        
        if self.use_grid_search:
            param_grid = {
                'C': [0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear'],
                'max_iter': [100, 500, 1000]
            }
            grid_search = GridSearchCV(
                estimator=LogisticRegression(solver='liblinear'),
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                verbose=2
            )
            grid_search.fit(self.X_train, self.y_train)
            logger.success(f"Best Logistic Regression Parameters: {grid_search.best_params_}")
            logger.success(f"Best Logistic Regression CV Score: {grid_search.best_score_}")
            logreg = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = grid_search.best_score_
        else:
            logreg = LogisticRegression(max_iter=1000)
            logreg.fit(self.X_train, self.y_train)
            best_params = logreg.get_params()
            cv_score = None

        y_pred = logreg.predict(self.X_test)
        metrics = self._extract_metrics(self.y_test, y_pred, "Logistic Regression")
        
        logger.success("Logistic Regression Classification Report:")
        logger.success(classification_report(self.y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

        # Save model and results
        joblib.dump(logreg, os.path.join(self.save_dir, "logistic_regression.pkl"))
        self._save_results_to_file("logistic_regression", best_params, metrics, cv_score)
        logger.success("Logistic Regression model and results saved.")

    def train_svm(self):
        if self.use_grid_search:
            logger.info("\nTraining SVM with Grid Search...")
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto'],
                'degree': [2, 3, 4]
            }
            grid_search = GridSearchCV(
                estimator=SVC(probability=True, random_state=42),
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                verbose=2
            )
            grid_search.fit(self.X_train, self.y_train)
            logger.success("Best SVM Parameters:", grid_search.best_params_)
            logger.success("Best SVM CV Score:", grid_search.best_score_)

            best_svm = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = grid_search.best_score_
        else:
            logger.info("\nTraining SVM without Grid Search...")
            best_svm = SVC(probability=True, random_state=42)
            best_svm.fit(self.X_train, self.y_train)
            best_params = best_svm.get_params()
            cv_score = None

        y_pred = best_svm.predict(self.X_test)
        metrics = self._extract_metrics(self.y_test, y_pred, "SVM")
        
        logger.success("SVM Classification Report:")
        logger.success(classification_report(self.y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

        # Save model and results
        joblib.dump(best_svm, os.path.join(self.save_dir, "svm_model.pkl"))
        self._save_results_to_file("svm", best_params, metrics, cv_score)
        logger.success("SVM model and results saved.")

    def train_random_forest(self):
        if self.use_grid_search:
            logger.info("\nTraining Random Forest with Grid Search...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            grid_search = GridSearchCV(
                estimator=RandomForestClassifier(random_state=42),
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                verbose=2
            )
            grid_search.fit(self.X_train, self.y_train)
            logger.success("Best Random Forest Parameters:", grid_search.best_params_)
            logger.success("Best Random Forest CV Score:", grid_search.best_score_)

            best_rf = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = grid_search.best_score_
        else:
            logger.info("\nTraining Random Forest without Grid Search...")
            best_rf = RandomForestClassifier(random_state=42)
            best_rf.fit(self.X_train, self.y_train)
            best_params = best_rf.get_params()
            cv_score = None

        y_pred = best_rf.predict(self.X_test)
        metrics = self._extract_metrics(self.y_test, y_pred, "Random Forest")
        
        logger.success("Random Forest Classification Report:")
        logger.success(classification_report(self.y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

        # Save model and results
        joblib.dump(best_rf, os.path.join(self.save_dir, "random_forest_model.pkl"))
        self._save_results_to_file("random_forest", best_params, metrics, cv_score)
        logger.success("Random Forest model and results saved.")
