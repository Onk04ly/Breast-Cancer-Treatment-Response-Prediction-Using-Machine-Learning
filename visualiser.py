import logging
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.gridspec import GridSpec
from sklearn.metrics import (roc_curve, precision_recall_curve, average_precision_score,
                             confusion_matrix, auc, balanced_accuracy_score, precision_score, recall_score, f1_score)


class DataVisualiser:
    """
    | Handles all visualisation tasks for the pipeline with error handling and consistent styling.
    | Authors: Benjamin Adonis
    """

    def __init__(self, results_dir):
        """
        | Initialise the visualiser with consistent styling
        :param results_dir: Directory to save plots
        | Authors: Benjamin Adonis
        """
        try:
            self.results_dir = Path(results_dir)
            self.results_dir.mkdir(parents=True, exist_ok=True)

            # Set consistent styling
            plt.style.use('default')
            sns.set_theme(style="whitegrid")
            sns.set_palette("husl")

            plt.rcParams.update({
                'figure.figsize': [10, 6],
                'figure.dpi': 100,
                'figure.autolayout': True,
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 10
            })

        except Exception as e:
            logging.error(f"Failed to initialise DataVisualiser: {str(e)}")
            raise

    def plot_model_performance(self, model_name, model, comparison_model,
                               X_test, y_test, results_type):
        """
        | Generate comprehensive performance visualisations for a model
        :param model_name: Name of the model
        :param model: Trained model
        :param comparison_model: Comparison model for comparative metrics
        :param X_test: Test features
        :param y_test: Test labels
        :param results_type: Type of results (base, hp, best)
        | Authors: Benjamin Adonis
        """
        # Create subdirectory for this model
        model_dir = self.results_dir / results_type / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Plot ROC curve
        self.plot_roc_curve(y_test, y_prob, model_name, model_dir)

        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred, model_name, model_dir)

        # Plot precision-recall curve
        self.plot_precision_recall_curve(y_test, y_prob, model_name, model_dir)

        # Plot prediction distribution
        self.plot_prediction_distribution(y_prob, y_test, model_name, model_dir)

        # If comparison model exists, plot comparative metrics
        if comparison_model is not None:
            self.plot_comparative_metrics(
                model, comparison_model, X_test, y_test, model_name, model_dir)

        # Plot feature importance if available
        if hasattr(model, 'feature_importances_'):
            if isinstance(model.feature_importances_, np.ndarray):
                self.plot_feature_importance(
                    list(X_test.columns),
                    model.feature_importances_,
                    model_name,
                    model_dir
                )

    def plot_roc_curve(self, y_true, y_prob, model_name, save_dir):
        """
        | Plot ROC curve with AUC
        :param y_true: True labels
        :param y_prob: Predicted probabilities
        :param model_name: Name of the model
        :param save_dir: Directory to save the plot
        | Authors: Benjamin Adonis
        """
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Curve - {model_name}')
            ax.legend(loc="lower right")

            self._save_plot(fig, save_dir / 'roc_curve.png')

        except Exception as e:
            logging.error(f"Failed to create ROC curve: {str(e)}")

    def plot_confusion_matrix(self, y_true, y_pred, model_name, save_dir):
        """
        | Plot confusion matrix with annotations and percentages
        :param y_true: True labels
        :param y_pred: Predicted labels
        :param model_name: Name of the model
        :param save_dir: Directory to save the plot
        | Authors: Benjamin Adonis
        """
        try:
            cm = confusion_matrix(y_true, y_pred)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Raw counts
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title('Confusion Matrix (Counts)')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')

            # Percentages
            sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues', ax=ax2)
            ax2.set_title('Confusion Matrix (Normalized)')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')

            plt.suptitle(f'Confusion Matrices - {model_name}')

            self._save_plot(fig, save_dir / 'confusion_matrix.png')

        except Exception as e:
            logging.error(f"Failed to create confusion matrix: {str(e)}")

    def plot_precision_recall_curve(self, y_true, y_prob, model_name, save_dir):
        """
        | Plot precision-recall curve with error handling
        :param y_true: True labels
        :param y_prob: Predicted probabilities
        :param model_name: Name of the model
        :param save_dir: Directory to save the plot
        | Authors: Benjamin Adonis
        """
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.plot(recall, precision,
                    label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'Precision-Recall Curve - {model_name}')
            ax.legend(loc="lower left")

            self._save_plot(fig, save_dir / 'precision_recall_curve.png')

        except Exception as e:
            logging.error(f"Failed to create precision-recall curve: {str(e)}")

    def plot_prediction_distribution(self, y_prob, y_true, model_name, save_dir):
        """
        | Plot distribution of prediction probabilities
        :param y_prob: Predicted probabilities
        :param y_true: True labels
        :param model_name: Name of the model
        :param save_dir: Directory to save the plot
        | Authors: Benjamin Adonis
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot distributions for each class
            for label in [0, 1]:
                mask = y_true == label
                sns.kdeplot(y_prob[mask], label=f'Class {label}', ax=ax)

            ax.set_xlabel('Prediction Probability')
            ax.set_ylabel('Density')
            ax.set_title(f'Prediction Probability Distribution - {model_name}')
            ax.legend()

            self._save_plot(fig, save_dir / 'prediction_distribution.png')

        except Exception as e:
            logging.error(f"Failed to create prediction distribution plot: {str(e)}")

    def plot_comparative_metrics(self, model1, model2, X_test, y_test,
                                 model_name, save_dir):
        """
        | Plot comparative performance metrics between two models
        :param model1: First model
        :param model2: Second model
        :param X_test: Test features
        :param y_test: Test labels
        :param model_name: Name of the model
        :param save_dir: Directory to save the plot
        | Authors: Benjamin Adonis
        """
        try:
            # Get predictions and metrics for both models
            y_pred1 = model1.predict(X_test)
            y_pred2 = model2.predict(X_test)

            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            model1_scores = []
            model2_scores = []

            # Calculate metrics
            for metric in metrics:
                if metric == 'Accuracy':
                    model1_scores.append(balanced_accuracy_score(y_test, y_pred1))
                    model2_scores.append(balanced_accuracy_score(y_test, y_pred2))
                elif metric == 'Precision':
                    model1_scores.append(precision_score(y_test, y_pred1))
                    model2_scores.append(precision_score(y_test, y_pred2))
                elif metric == 'Recall':
                    model1_scores.append(recall_score(y_test, y_pred1))
                    model2_scores.append(recall_score(y_test, y_pred2))
                else:  # F1-Score
                    model1_scores.append(f1_score(y_test, y_pred1))
                    model2_scores.append(f1_score(y_test, y_pred2))

            # Plot comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            x = np.arange(len(metrics))
            width = 0.35

            ax.bar(x - width / 2, model1_scores, width, label='Model 1')
            ax.bar(x + width / 2, model2_scores, width, label='Model 2')

            ax.set_ylabel('Score')
            ax.set_title(f'Comparative Performance - {model_name}')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()

            self._save_plot(fig, save_dir / 'comparative_metrics.png')

        except Exception as e:
            logging.error(f"Failed to create comparative metrics plot: {str(e)}")

    def _save_plot(self, fig, filepath):
        """
        | Save plot with error handling
        :param fig: Figure to save
        :param filepath: Path to save the plot
        | Authors: Benjamin Adonis
        """
        try:
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"Plot saved: {filepath}")
        except Exception as e:
            logging.error(f"Failed to save plot: {str(e)}")
            plt.close(fig)

    ### Legacy Visualisation Methods ###

    def save_plot(self, fig, filename):
        """
        Save plot with timestamp and handle errors
        :param fig: Figure to save
        :param filename: Base filename for the plot
        | Authors: Benjamin Adonis
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.results_dir / f'{filename}_{timestamp}.png'
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logging.info(f"Plot saved: {filepath}")
        except Exception as e:
            logging.error(f"Failed to save plot {filename}: {str(e)}")
            plt.close(fig)  # Figure is closed even if save fails

    def plot_class_distribution(self, y, title):
        """
        Plot class distribution with error handling
        :param y: Target values
        :param title: Title for the plot
        | Authors: Benjamin Adonis
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x=y, ax=ax)
            ax.set_title(f'Class Distribution - {title}', pad=20)
            ax.set_xlabel('PCR Outcome')
            ax.set_ylabel('Count')

            # Add count labels
            for p in ax.patches:
                height = p.get_height()
                ax.annotate(f'{int(height)}',
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom', fontsize=10)

            self.save_plot(fig, f'class_distribution_{title.lower().replace(" ", "_")}')
        except Exception as e:
            logging.error(f"Failed to create class distribution plot: {str(e)}")

    def plot_feature_importance(self, features, importance_scores, title):
        """
        | Plot feature importance with error handling#
        :param features: Feature names
        :param importance_scores: Importance scores
        :param title: Title for the plot
        | Authors: Benjamin Adonis
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Create and sort importance DataFrame
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importance_scores
            }).sort_values('Importance', ascending=True)

            # Plot top 15 features
            top_features = importance_df.tail(15)
            sns.barplot(data=top_features, y='Feature', x='Importance', ax=ax)
            ax.set_title(f'Top 15 {title}', pad=20)
            ax.set_xlabel('Importance Score')

            # Add value labels
            for i, v in enumerate(top_features['Importance']):
                ax.text(v, i, f'{v:.3f}', va='center', fontsize=8)

            self.save_plot(fig, f'feature_importance_{title.lower().replace(" ", "_")}')
        except Exception as e:
            logging.error(f"Failed to create feature importance plot: {str(e)}")