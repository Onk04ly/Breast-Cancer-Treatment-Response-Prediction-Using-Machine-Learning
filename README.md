# Breast Cancer Treatment Response Prediction

## Introduction

This machine learning project addresses a critical challenge in breast cancer treatment: predicting patient responses to chemotherapy. The system provides two key predictions: the likelihood of achieving pathological complete response (PCR) and the expected relapse-free survival (RFS) time. These predictions are generated through analysis of both clinical indicators and radiomics features derived from magnetic resonance imaging (MRI) scans.

## System Requirements and Installation

This system was developed in Python 3.11.10.

Install the required dependencies:
```
pip install -r requirements.txt
```

The installation process will configure all necessary packages, including pandas, numpy, scikit-learn, imbalanced-learn, xgboost, lightgbm, catboost, matplotlib, seaborn, and joblib.

## Project Architecture

The system architecture comprises several interconnected modules, each serving a specific function in the prediction pipeline:

The root directory contains the core implementation files:
```
Assignment_2/
│
├── analysis_utils.py      # Core analytical functions
├── FinalTestPCR.py      # PCR prediction implementation
├── constants.py          # System configuration
├── main.py              # Primary execution pipeline
├── preprocessing.py     # Data preparation pipeline
├── regression.py       # RFS prediction implementation
├── visualiser.py      # Results visualisation
│
├── Datasets/  # Used datasets
├── results/           # Output directory
└── trained_models/    # Model storage
```

## Implementation Process

The implementation begins with data preparation. Your dataset files should be placed in the project's root directory. The system configuration file, `constants.py`, must be updated to reflect your data file locations:

```python
FILE_PATHS = {
    'train_data': 'TrainDataset2024.csv',
    'test_data': 'TestDatasetExample.csv'
}
```

The system offers four distinct operational modes, each serving different analytical purposes:

The first mode, `base_train_models`, conducts initial training with default parameters. The second, `hp_train_models`, performs hyperparameter optimisation on the most promising models. The third mode, `best_train_model`, executes deep parameter tuning on the optimal model. The fourth mode, `best_saved_model`, utilises a previously trained model for predictions.

To select an operational mode for classification models, modify the following parameter in `constants.py`:
```python
TRAINING_MODE = 'base_train_models'
```

## Execution and Output Generation

To execute the prediction pipeline, run:
```bash
python main.py
```

The system performs comprehensive data processing, model training, and evaluation automatically. For test set predictions, ensure your test data maintains the same format as the training set. The system generates two primary output files: `PCRPrediction.csv` for classification results and `RFSPrediction.csv` for survival predictions.

## Performance Assessment

The system employs rigorous evaluation metrics for model assessment. For PCR classification, we utilise balanced accuracy as the primary metric, supplemented by precision, recall, F1-score, and ROC-AUC measurements. The RFS regression performance is evaluated using Mean Absolute Error.

All evaluation results, including detailed performance visualisations and training logs, are automatically organised by timestamp within the results directory. This systematic organisation ensures complete traceability of all analytical outputs.

## Support and Documentation

This implementation forms part of the COMP3009/COMP4139 Machine Learning coursework. Detailed implementation specifications can be found within the inline documentation of each module.

The system has been designed with comprehensive error handling and logging capabilities to facilitate troubleshooting and ensure robust operation in various scenarios. Each module contains extensive documentation explaining its functionality and integration within the broader system architecture.