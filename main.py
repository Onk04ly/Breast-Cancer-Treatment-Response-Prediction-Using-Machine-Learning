import logging
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from Assignment_2.analysis_utils import analyse_model_performance
import FinalTestRFS

# Filter warnings
logging.getLogger('matplotlib.font_manager').disabled = True
warnings.filterwarnings('ignore', category=UserWarning)

# Local imports
from preprocessing import Preprocessor
from FinalTestPCR import Classifier
from visualiser import DataVisualiser
from constants import Constants


def setup_logging(training_mode):
    """
    | Configure logging with timestamp and suppress unwanted messages.
    :return: Tuple of timestamp and results directory
    | Authors: Benjamin Adonis
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create mode-specific results directory
    if training_mode == 'base_train_models':
        results_dir = Path(Constants.MODEL_PATHS['base_results'])
    elif training_mode == 'hp_train_models':
        results_dir = Path(Constants.MODEL_PATHS['hp_results'])
    elif training_mode in ['best_train_model', 'best_saved_model']:
        results_dir = Path(Constants.MODEL_PATHS['best_results'])
    else:
        raise ValueError(f"Invalid training mode: {training_mode}")

    results_dir = results_dir / timestamp
    log_dir = results_dir / 'logs'

    # Create directories
    results_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_file = log_dir / 'training.log'
    logging.basicConfig(
        level=Constants.LOGGING_LEVEL,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return timestamp, results_dir


def validate_dataset(data):
    """
    | Validate dataset for required features and quality checks.
    :param data: Input dataset
    :return: Tuple of error and warning messages
    | Authors: Benjamin Adonis
    """
    errors = []
    warnings = []

    if data is None or data.empty:
        errors.append("Dataset is empty or None")
        return errors, warnings

    # Check mandatory features
    missing_features = set(Constants.MANDATORY_FEATURES) - set(data.columns)
    if missing_features:
        errors.append(f"Missing mandatory features: {missing_features}")

    # Check class distribution if target exists
    if Constants.PCR_TARGET in data.columns:
        class_counts = data[Constants.PCR_TARGET].value_counts()
        for class_label, count in class_counts.items():
            if count < Constants.MIN_SAMPLES_PER_CLASS:
                warnings.append(f"Low sample count for class {class_label}: {count}")

    return errors, warnings

def save_predictions(predictions, patient_ids, results_dir, filename='PCRPrediction.csv'):
    """
    | Save model predictions to CSV file.
    :param predictions: Model predictions
    :param patient_ids: Patient IDs
    :param results_dir: Directory to save results
    :param filename: Output filename
    | Authors: Benjamin Adonis
    """
    predictions_df = pd.DataFrame({
        'ID': patient_ids,
        'Prediction': predictions.astype(int)
    })
    output_path = results_dir / filename
    predictions_df.to_csv(output_path, index=False)
    logging.info(f"Predictions saved to {output_path}")

def main():
    """
    | Main execution pipeline for PCR prediction
    | Authors: Benjamin Adonis
    """
    results_dir = None
    timestamp = None

    try:
        # Setup environment based on training mode
        timestamp, results_dir = setup_logging(Constants.TRAINING_MODE)
        logging.info(f"Starting PCR prediction pipeline in {Constants.TRAINING_MODE} mode")

        # Initialise components
        visualiser = DataVisualiser(results_dir)
        preprocessor = Preprocessor()
        classifier = Classifier(visualiser=visualiser)

        # Load and validate data
        logging.info("Loading training data...")
        data = preprocessor.load_data(Constants.FILE_PATHS['train_data'])

        errors, warnings = validate_dataset(data)
        if errors:
            for error in errors:
                logging.error(f"Validation Error: {error}")
            raise ValueError("Dataset validation failed")
        for warning in warnings:
            logging.warning(f"Validation Warning: {warning}")

        # Preprocess data
        logging.info("\nStarting data preprocessing...")
        processed_data = preprocessor.preprocess(data, target_column=Constants.PCR_TARGET)

        # Split data
        logging.info("\nSplitting data...")
        y = processed_data[Constants.PCR_TARGET].dropna()
        X = processed_data.loc[y.index].drop(Constants.PCR_TARGET, axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=Constants.TEST_SIZE,
            stratify=y,
            random_state=Constants.RANDOM_SEED
        )

        # Train and evaluate based on mode
        logging.info(f"\nStarting model training in {Constants.TRAINING_MODE} mode...")
        results = classifier.train_and_evaluate(X_train, X_test, y_train, y_test)

        if not results:
            raise ValueError("No models were successfully trained and evaluated")

        # Save run metadata
        with open(results_dir / 'run_metadata.txt', 'w') as f:
            f.write(f"Run timestamp: {timestamp}\n")
            f.write(f"Training mode: {Constants.TRAINING_MODE}\n")
            f.write(f"Dataset shape: {data.shape}\n")
            f.write(f"Preprocessing steps: {preprocessor.get_preprocessing_steps()}\n")
            f.write("\nResults summary:\n")

            # Now we can use the same handling for all modes
            for model_name, model_results in results.items():
                f.write(f"\n{model_name}:\n")
                if 'metrics' in model_results:
                    for metric, value in model_results['metrics'].items():
                        f.write(f"  {metric}: {value:.4f}\n")
                if 'best_params' in model_results:
                    f.write("\nBest parameters:\n")
                    for param, value in model_results['best_params'].items():
                        f.write(f"  {param}: {value}\n")

                # Add extra information for best_train_model mode
                if Constants.TRAINING_MODE == 'best_train_model':
                    f.write("\nPerformance Comparison:\n")
                    previous_performance = Constants.BEST_MODEL['hp_performance']
                    current_performance = model_results['metrics']['balanced_accuracy']
                    improvement = ((current_performance - previous_performance) / previous_performance) * 100

                    f.write(f"Previous HP tuning balanced accuracy: {previous_performance:.4f}\n")
                    f.write(f"Current deep tuning balanced accuracy: {current_performance:.4f}\n")
                    f.write(f"Improvement: {improvement:+.2f}%\n")

                    f.write("\nTraining Configuration:\n")
                    f.write("Parameter grids tested:\n")
                    for i, grid in enumerate(Constants.DEEP_PARAM_GRIDS['logistic_regression'], 1):
                        f.write(f"\nGrid {i}:\n")
                        for param, values in grid.items():
                            f.write(f"  {param}: {values}\n")

        # Save predictions if needed
        if Constants.TRAINING_MODE in ['best_train_model', 'best_saved_model']:
            for model_name, model_results in results.items():
                if 'predictions' in model_results:
                    save_predictions(
                        model_results['predictions'],
                        X_test.index,
                        results_dir,
                        'PCRPrediction.csv'
                    )

        logging.info("Pipeline completed successfully!")

    except Exception as e:
        logging.error(f"Pipeline failed with error: {str(e)}")
        if results_dir:
            with open(results_dir / 'error_report.txt', 'w') as f:
                f.write(f"Pipeline failed at: {datetime.now()}\n")
                f.write(f"Error: {str(e)}\n")
        raise

    FinalTestRFS.runFinalTestRFS()

if __name__ == '__main__':
    main()


