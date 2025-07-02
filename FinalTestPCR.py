import logging
from pathlib import Path

import joblib
import numpy as np
from joblib import dump, load
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (balanced_accuracy_score, accuracy_score, recall_score,
                           precision_score, f1_score, roc_auc_score, make_scorer)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                            HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from constants import Constants

class Classifier:
    """
    | Classifier class handling model training, evaluation, and ensemble methods.
    | Authors: Benjamin Adonis
    """

    def __init__(self, visualiser=None):
        """
        | Initialise classifier with model selection and parameter grids.
        :param visualiser: Optional DataVisualiser instance for plotting results
        | Authors: Benjamin Adonis
        """
        self.random_state = Constants.RANDOM_SEED
        self.visualiser = visualiser
        self.trained_models = {}
        self.results = {}
        self.base_models = {}
        self._initialise_base_models()

    def _initialise_base_models(self):
        """
        | Initialise all base models with default configurations
        | Authors: Benjamin Adonis
        """
        self.base_models = {
            'logistic_regression': LogisticRegression(
                class_weight='balanced', max_iter=3000,
                random_state=self.random_state, n_jobs=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=1000, class_weight='balanced_subsample',
                random_state=self.random_state, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=500, random_state=self.random_state
            ),
            'hist_gradient_boosting': HistGradientBoostingClassifier(
                random_state=self.random_state
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=1000, random_state=self.random_state, n_jobs=-1, verbose=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=1000, random_state=self.random_state, n_jobs=-1
            ),
            'catboost': CatBoostClassifier(
                iterations=1000, random_state=self.random_state, verbose=False
            ),
            'svm': SVC(
                probability=True, random_state=self.random_state
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50), random_state=self.random_state
            )
        }

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        | Main training and evaluation method based on training mode
        :param X_train: Training features
        :param X_test: Test features
        :param y_train: Training labels
        :param y_test: Test labels
        | Authors: Benjamin Adonis
        """
        if Constants.TRAINING_MODE == 'base_train_models':
            return self._run_base_training(X_train, X_test, y_train, y_test)
        elif Constants.TRAINING_MODE == 'hp_train_models':
            return self._run_hp_training(X_train, X_test, y_train, y_test)
        elif Constants.TRAINING_MODE == 'best_train_model':
            return self._run_best_model_training(X_train, X_test, y_train, y_test)
        elif Constants.TRAINING_MODE == 'best_saved_model':
            return self._load_and_evaluate_best_model(X_test, y_test)

    def _create_pipeline(self, model, use_smote=True):
        """
        | Create a preprocessing pipeline with optional SMOTE
        :param model: The classifier model to include in the pipeline
        :param use_smote: Boolean indicating whether to include SMOTE
        :return: Pipeline with preprocessing and classifier
        | Authors: Benjamin Adonis
        """
        steps = [('scaler', StandardScaler())]
        if use_smote:
            steps.append(('smote', SMOTE(random_state=self.random_state)))
        steps.append(('classifier', model))
        return Pipeline(steps)

    def _evaluate_model(self, y_true, y_pred, y_prob=None):
        """
        | Comprehensive model evaluation using predictions and probabilities
        :param y_true: True labels
        :param y_pred: Predicted labels
        :param y_prob: Predicted probabilities (optional)
        :return: Dictionary of evaluation metrics
        | Authors: Benjamin Adonis
        """
        metrics = {
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred),
            'sensitivity': recall_score(y_true, y_pred),
            'specificity': recall_score(y_true, y_pred, pos_label=0),
            'precision': precision_score(y_true, y_pred),
            'npv': precision_score(y_true, y_pred, pos_label=0),
            'f1': f1_score(y_true, y_pred)
        }

        # Add AUC-ROC if probabilities are provided
        if y_prob is not None:
            metrics['auc_roc'] = roc_auc_score(y_true, y_prob)

        return metrics

    def _run_base_training(self, X_train, X_test, y_train, y_test):
        """
        | Run base model training with and without SMOTE
        :param X_train: Training features
        :param X_test: Test features
        :param y_train: Training labels
        :param y_test: Test labels
        | Authors: Benjamin Adonis
        """
        results = {}

        for model_name in Constants.BASE_MODEL_NAMES:
            model_results = {}
            logging.info(f"\nTraining base model: {model_name}")

            try:
                # Train without SMOTE
                pipeline = self._create_pipeline(self.base_models[model_name], use_smote=False)
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                y_prob = pipeline.predict_proba(X_test)[:, 1]

                # Store metrics and predictions
                metrics_no_smote = self._evaluate_model(y_test, y_pred, y_prob)
                model_results['no_smote'] = {
                    'balanced_accuracy': metrics_no_smote['balanced_accuracy'],
                    'predictions': y_pred,
                    'probabilities': y_prob,
                    **metrics_no_smote  # Include all metrics
                }

                # Train with SMOTE
                pipeline_smote = self._create_pipeline(self.base_models[model_name], use_smote=True)
                pipeline_smote.fit(X_train, y_train)
                y_pred_smote = pipeline_smote.predict(X_test)
                y_prob_smote = pipeline_smote.predict_proba(X_test)[:, 1]

                # Store metrics and predictions for SMOTE
                metrics_smote = self._evaluate_model(y_test, y_pred_smote, y_prob_smote)
                model_results['smote'] = {
                    'balanced_accuracy': metrics_smote['balanced_accuracy'],
                    'predictions': y_pred_smote,
                    'probabilities': y_prob_smote,
                    **metrics_smote  # Include all metrics
                }

                results[model_name] = model_results

                # Generate visualisations
                if self.visualiser:
                    try:
                        self.visualiser.plot_model_performance(model_name, pipeline, pipeline_smote,
                                                               X_test, y_test, 'base_results')
                    except Exception as viz_error:
                        logging.error(f"Error generating visualisations for {model_name}: {str(viz_error)}")

            except Exception as e:
                logging.error(f"Error training {model_name}: {str(e)}")
                continue

        # Save results before returning
        try:
            self._save_results(results, 'base_results')
        except Exception as save_error:
            logging.error(f"Error saving results: {str(save_error)}")

        return results

    def _run_hp_training(self, X_train, X_test, y_train, y_test):
        """
        | Run hyperparameter optimisation for top models
        :param X_train: Training features
        :param X_test: Test features
        :param y_train: Training labels
        :param y_test: Test labels
        | Authors: Benjamin Adonis
        """
        results = {}

        for model_name, model_info in Constants.TOP_MODELS.items():
            logging.info(f"\nTraining HP model: {model_name}")
            try:
                use_smote = model_info['sampling_method'] == 'smote'

                pipeline = self._create_pipeline(model_info['model'], use_smote=use_smote)
                param_grid = self._get_hp_param_grid(model_name)

                grid_search = GridSearchCV(
                    pipeline, param_grid,
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                    scoring='balanced_accuracy', n_jobs=-1
                )

                grid_search.fit(X_train, y_train)
                best_model = grid_search.best_estimator_

                # Get predictions from best model
                y_pred = best_model.predict(X_test)
                y_prob = best_model.predict_proba(X_test)[:, 1]
                metrics = self._evaluate_model(y_test, y_pred, y_prob)

                results[model_name] = {
                    'metrics': metrics,
                    'best_params': grid_search.best_params_,
                    'predictions': y_pred,
                    'probabilities': y_prob
                }

                if self.visualiser:
                    self.visualiser.plot_model_performance(model_name, best_model, None,
                                                           X_test, y_test, 'hp_results')

            except Exception as e:
                logging.error(f"Error in HP training for {model_name}: {str(e)}")
                continue

        self._save_results(results, 'hp_results')
        return results

    def _run_best_model_training(self, X_train, X_test, y_train, y_test):
            """
            | Train the best model with deep hyperparameter tuning
            :param X_train: Training features
            :param X_test: Test features
            :param y_train: Training labels
            :param y_test: Test labels
            | Authors: Benjamin Adonis"""
            model_name = Constants.BEST_MODEL['name']
            use_smote = Constants.BEST_MODEL['sampling_method'] == 'smote'
            logging.info(f"\nTraining best model: {model_name}")
            try:
                pipeline = self._create_pipeline(Constants.BEST_MODEL['model'], use_smote=use_smote)
                param_grids = Constants.DEEP_PARAM_GRIDS[model_name]
                best_score = float('-inf')
                best_params = None
                best_model = None
                # Train with each parameter grid
                for i, param_grid in enumerate(param_grids, 1):
                    logging.info(f"\nTesting parameter grid {i} of {len(param_grids)}")
                    grid_search = GridSearchCV(
                        pipeline, param_grid,
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                        scoring='balanced_accuracy',
                        n_jobs=-1,
                        verbose=1
                    )
                    grid_search.fit(X_train, y_train)
                    logging.info(f"Grid {i} best score: {grid_search.best_score_:.4f}")
                    if grid_search.best_score_ > best_score:
                        best_score = grid_search.best_score_
                        best_params = grid_search.best_params_
                        best_model = grid_search.best_estimator_
                # Get predictions from best model
                y_pred = best_model.predict(X_test)
                y_prob = best_model.predict_proba(X_test)[:, 1]
                metrics = self._evaluate_model(y_test, y_pred, y_prob)
                # Structure results consistently with other modes
                results = {
                    model_name: {
                        'metrics': metrics,
                        'best_params': best_params,
                        'predictions': y_pred,
                        'probabilities': y_prob
                    }
                }
                # Save model
                model_path = Path(Constants.MODEL_PATHS['trained_models']) / f"{model_name}.joblib"
                Path(Constants.MODEL_PATHS['trained_models']).mkdir(parents=True, exist_ok=True)
                dump(best_model, model_path)
                logging.info(f"Best model saved to {model_path}")
                # Save results
                self._save_results(results, 'best_results')
                return results
            except Exception as e:
                logging.error(f"Error in best model training: {str(e)}")
                return None

    def _get_hp_param_grid(self, model_name):
        """
        | Get hyperparameter grid for HP optimisation
        :param model_name: Name of the model
        | Authors: Benjamin Adonis
        """
        return Constants.PARAM_GRIDS.get(model_name, {})

    def _get_deep_hp_param_grid(self, model_name):
        """
        | Get deep hyperparameter grid for best model optimisation
        :param model_name: Name of the model
        | Authors: Benjamin Adonis
        """
        return Constants.DEEP_PARAM_GRIDS.get(model_name, {})

    def _save_results(self, results, results_type):
        """
        | Save results to specified directory
        :param results: Dictionary containing results
        :param results_type: Type of results to save
        | Authors: Benjamin Adonis
        """
        try:
            results_dir = Path(Constants.MODEL_PATHS[results_type])
            results_dir.mkdir(parents=True, exist_ok=True)
            results_path = results_dir / 'results.joblib'
            dump(results, results_path)
            logging.info(f"Results saved to {results_path}")
        except Exception as e:
            logging.error(f"Error saving results: {str(e)}")

    def _load_and_evaluate_best_model(self, X_test, y_test):
        """
        | Load and evaluate the saved best model
        :param X_test: Test features
        :param y_test: Test labels
        | Authors: Benjamin Adonis
        """
        model_name = Constants.BEST_MODEL['name']
        model_path = f"{Constants.MODEL_PATHS['trained_models']}/{model_name}.joblib"
        try:
            # Directly load the model without modification
            model = joblib.load(model_path)
            # If it's a pipeline, use predict and predict_proba from the pipeline
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = self._evaluate_model(y_test, y_pred, y_prob)
            if self.visualiser:
                self.visualiser.plot_model_performance(model_name, model, None,
                                                       X_test, y_test, 'best_results')
            return {
                model_name: {
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': y_prob
                }
            }
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            return None

    def predict(self, model_name, X_test):
        """
        | Prediction with probability threshold adjustment
        :param model_name: Name of the model to use
        :param X_test: Test features
        | Authors: Benjamin Adonis
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not found or not trained")

        try:
            probas = self.trained_models[model_name].predict_proba(X_test)
            return (probas[:, 1] >= 0.4).astype(int)
        except Exception as e:
            logging.error(f"Error in prediction for {model_name}: {str(e)}")
            return None