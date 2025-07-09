import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
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

        if not os.path.exists(PROJ_ROOT / "models"):
            os.makedirs(PROJ_ROOT / "models")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

    def train_logistic_regression(self):
        logger.info("\nTraining Logistic Regression...")
        logreg = LogisticRegression(max_iter=1000)
        logreg.fit(self.X_train, self.y_train)
        y_pred = logreg.predict(self.X_test)

        logger.success("Logistic Regression Classification Report:")
        logger.success(classification_report(self.y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

        joblib.dump(logreg, os.path.join(self.save_dir, "logistic_regression.pkl"))
        logger.success("Logistic Regression model saved.")

    def train_svm(self):
        if self.use_grid_search:
            logger.info("\nTraining SVM with Grid Search...")
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf', 'poly'],
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
        else:
            logger.info("\nTraining SVM without Grid Search...")
            best_svm = SVC(probability=True, random_state=42)
            best_svm.fit(self.X_train, self.y_train)

        y_pred = best_svm.predict(self.X_test)
        logger.success("SVM Classification Report:")
        logger.success(classification_report(self.y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

        joblib.dump(best_svm, os.path.join(self.save_dir, "svm_model.pkl"))
        logger.success("SVM model saved.")

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
        else:
            logger.info("\nTraining Random Forest without Grid Search...")
            best_rf = RandomForestClassifier(random_state=42)
            best_rf.fit(self.X_train, self.y_train)

        y_pred = best_rf.predict(self.X_test)
        logger.success("Random Forest Classification Report:")
        logger.success(classification_report(self.y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

        joblib.dump(best_rf, os.path.join(self.save_dir, "random_forest_model.pkl"))
        logger.success("Random Forest model saved.")
