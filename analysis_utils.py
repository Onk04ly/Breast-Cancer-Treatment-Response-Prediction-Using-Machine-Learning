import logging
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


def analyse_smote_results(y_train_before, y_train_after, visualiser):
    """
    | Analyse and log SMOTE results
    :param y_train_before: Target labels before SMOTE
    :param y_train_after: Target labels after SMOTE
    :param visualiser: Instance of DataVisualiser for plotting
    | Authors: Benjamin Adonis
    """
    logging.info("\nSMOTE Analysis:")
    # Before SMOTE
    class_dist_before = pd.Series(y_train_before).value_counts()
    logging.info("\nClass distribution before SMOTE:")
    logging.info(class_dist_before)
    visualiser.plot_class_distribution(y_train_before, "Before SMOTE")
    # After SMOTE
    class_dist_after = pd.Series(y_train_after).value_counts()
    logging.info("\nClass distribution after SMOTE:")
    logging.info(class_dist_after)
    visualiser.plot_class_distribution(y_train_after, "After SMOTE")
    # Calculate and log changes
    for class_label in class_dist_after.index:
        before_count = class_dist_before.get(class_label, 0)
        after_count = class_dist_after.get(class_label, 0)
        percent_change = ((after_count - before_count) / before_count) * 100
        logging.info(f"Class {class_label}:")
        logging.info(f"  Before: {before_count}")
        logging.info(f"  After: {after_count}")
        logging.info(f"  Change: {percent_change:.1f}%")

def analyse_model_performance(model_name, y_true, y_pred, probabilities=None):
    """
    | Analyse and log detailed model performance metrics
    :param model_name: Name of the model
    :param y_true: True target labels
    :param y_pred: Predicted target labels
    :param probabilities: Predicted probabilities (if available)
    :return: Dictionary of performance metrics
    | Authors: Benjamin Adonis
    """
    results = {}

    results = {}

    # Basic metrics
    conf_matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    # Calculate metrics
    results['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    results['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    results['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    results['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    results['f1'] = 2 * (results['precision'] * results['sensitivity']) / (
                results['precision'] + results['sensitivity']) if (results['precision'] + results[
        'sensitivity']) > 0 else 0

    # Log results
    logging.info(f"\nDetailed Analysis for {model_name}:")
    logging.info(f"Confusion Matrix:\n{conf_matrix}")
    for metric, value in results.items():
        logging.info(f"{metric.capitalize()}: {value:.3f}")

    return results


def generate_preprocessing_summary(data_before, data_after):
    """
    | Generate summary statistics for preprocessing steps
    :param data_before: Data before preprocessing
    :param data_after: Data after preprocessing
    :return: Dictionary of statistics
    | Authors: Benjamin Adonis
    """
    stats = {}

    try:
        # Missing values analysis for original data
        stats['missing_counts'] = data_before.isnull().sum()
        # Get numeric columns that exist in both datasets
        numeric_cols_before = data_before.select_dtypes(include=[np.number]).columns
        numeric_cols_after = data_after.select_dtypes(include=[np.number]).columns
        common_numeric_cols = list(set(numeric_cols_before) & set(numeric_cols_after))
        if common_numeric_cols:
            # Calculate statistics for common numeric columns only
            stats['original_stats'] = data_before[common_numeric_cols].describe()
            stats['processed_stats'] = data_after[common_numeric_cols].describe()
            # Log the changes in features
            logging.info("\nPreprocessing Summary:")
            logging.info(f"Original features: {len(data_before.columns)}")
            logging.info(f"Processed features: {len(data_after.columns)}")
            logging.info(f"Common numeric features: {len(common_numeric_cols)}")
        else:
            # Create empty DataFrames if no common columns
            stats['original_stats'] = pd.DataFrame()
            stats['processed_stats'] = pd.DataFrame()
            logging.warning("No common numeric columns found between original and processed data")
        # Feature type summary
        feature_types_before = data_before.dtypes.value_counts()
        feature_types_after = data_after.dtypes.value_counts()
        stats['feature_types'] = {
            'before': feature_types_before.to_dict(),
            'after': feature_types_after.to_dict()
        }
    except Exception as e:
        logging.error(f"Error generating preprocessing summary: {str(e)}")
        stats['error'] = str(e)
    return stats