from scipy.stats import uniform
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class Constants:
    """
    | Centralised constants for the project
    | Authors: Benjamin Adonis
    """
    # === Debugging and Runtime Settings === #
    RANDOM_SEED = 42  # For reproducibility across models and experiments
    LOGGING_LEVEL = 'INFO'  # NOTSET / DEBUG / INFO / WARNING / ERROR / CRITICAL
    N_JOBS = -1  # Number of CPU cores to use (-1 for all available)

    # === Model and Evaluation Settings === #
    TEST_SIZE = 0.2  # Proportion of dataset to use for testing

    # === Feature Engineering Settings === #
    PCA_VARIANCE_THRESHOLD = 0.95  # Threshold for retained variance in PCA
    CORRELATION_THRESHOLD = 0.9  # Threshold for removing highly correlated features

    # === Target Column Names === #
    PCR_TARGET = 'pCR (outcome)'
    RFS_TARGET = 'RelapseFreeSurvival (outcome)'

    # === Model Names === #
    MODEL_NAMES = [
        'logistic_regression',
        'random_forest',
        'gradient_boosting',
        'hist_gradient_boosting',
        'lightgbm',
        'xgboost',
        'catboost',
        'svm',
        'neural_network'
    ]

    # === Hyperparameter Search Ranges === #
    # Random Forest
    RF_PARAMS = {
        'classifier__n_estimators': (100, 500),
        'classifier__max_depth': [None] + list(range(5, 20)),
        'classifier__min_samples_split': (2, 20),
        'classifier__max_features': ['sqrt', 'log2']
    }

    # LightGBM
    LGBM_PARAMS = {
        'classifier__n_estimators': (100, 500),
        'classifier__learning_rate': (0.01, 0.3),
        'classifier__num_leaves': (20, 100),
        'classifier__max_depth': (-1, 15),  # -1 means no limit
        'classifier__min_child_samples': (1, 50)
    }

    # CatBoost
    CATBOOST_PARAMS = {
        'classifier__iterations': (100, 500),
        'classifier__learning_rate': (0.01, 0.3),
        'classifier__depth': (4, 10),
        'classifier__l2_leaf_reg': (1, 10)
    }

    # SVM
    SVM_PARAMS = {
        'classifier__C': uniform(0.001, 100),
        'classifier__gamma': uniform(0.0001, 1.0),
        'classifier__kernel': ['rbf', 'poly', 'sigmoid'],
        'classifier__degree': [2, 3, 4],
        'classifier__coef0': uniform(-1, 1),
        'classifier__tol': uniform(1e-4, 1e-3),
        'classifier__class_weight': ['balanced', None],
        'classifier__shrinking': [True, False]
    }

    # === Training Mode Settings === #
    TRAINING_MODE = 'best_saved_model'  # Options: base_train_models, hp_train_models, best_train_model, best_saved_model

    # Base training model settings
    BASE_MODEL_NAMES = [
        'logistic_regression',
        'random_forest',
        'gradient_boosting',
        'hist_gradient_boosting',
        'lightgbm',
        'xgboost',
        'catboost',
        'svm',
        'neural_network'
    ]

    # === Hyperparameter Grids for Top Models === #
    PARAM_GRIDS = {
        'logistic_regression': {
            # Balance between sensitivity and specificity
            'classifier__C': [0.007, 0.01, 0.015],  # Slightly higher than previous to reduce overfitting
            'classifier__penalty': ['l2'],
            'classifier__solver': ['liblinear'],
            'classifier__class_weight': ['balanced', {0: 1.5, 1: 2}],  # More nuanced weight balance
            'classifier__fit_intercept': [True],  # Keep True as it performed better
            'classifier__tol': [1e-4, 1e-5]  # Add tolerance parameter
        },
        'svm': {
            # Return to more conservative values
            'classifier__C': [0.1, 0.5, 1],  # Lower C values to reduce overfitting
            'classifier__gamma': [0.0001, 0.0005, 0.001],  # Lower gamma values
            'classifier__kernel': ['rbf'],
            'classifier__class_weight': ['balanced', {0: 1.5, 1: 2}],  # More nuanced weights
            'classifier__max_iter': [5000],
            'classifier__tol': [1e-4]
        },
        'lightgbm': {
            # Build on improvements
            'classifier__n_estimators': [750, 1000],  # More trees for stability
            'classifier__learning_rate': [0.03, 0.05],  # Focused learning rates
            'classifier__max_depth': [4, 5],  # Slightly increase depth
            'classifier__num_leaves': [23, 31],  # Proportional to depth
            'classifier__min_child_samples': [15, 20],
            'classifier__subsample': [0.8, 0.9],
            'classifier__colsample_bytree': [0.9],  # Keep high feature sampling
            'classifier__reg_alpha': [0.3, 0.5],  # Slightly reduce L1
            'classifier__reg_lambda': [0.1, 0.3],  # Slightly increase L2
            'classifier__scale_pos_weight': [2.5, 3, 3.5],  # Fine-tune class balance
            'classifier__boosting_type': ['gbdt', 'dart']  # Try DART boosting
        }
    }

    # === Deep Hyperparameter Grids for Best Model === #
    DEEP_PARAM_GRIDS = {
        'logistic_regression': [
            # Group 1: l2 penalty - Conservative baseline
            {
                'classifier__C': [0.01, 0.012, 0.015],
                'classifier__penalty': ['l2'],
                'classifier__solver': ['liblinear'],
                'classifier__class_weight': ['balanced'],
                'classifier__fit_intercept': [True],
                'classifier__tol': [1e-4]
            },
            # Group 2: elasticnet with moderate l1 - Balanced feature selection
            {
                'classifier__C': [0.01, 0.012, 0.015],
                'classifier__penalty': ['elasticnet'],
                'classifier__solver': ['saga'],
                'classifier__class_weight': ['balanced'],
                'classifier__fit_intercept': [True],
                'classifier__tol': [1e-4],
                'classifier__l1_ratio': [0.15, 0.2, 0.25]
            },
            # Group 3: l2 with scaled class weights - Handle imbalance
            {
                'classifier__C': [0.01, 0.012, 0.015],
                'classifier__penalty': ['l2'],
                'classifier__solver': ['liblinear'],
                'classifier__class_weight': ['balanced', {0: 1.2, 1: 1.8}],  # Conservative custom weight
                'classifier__fit_intercept': [True],
                'classifier__tol': [1e-4]
            },
            # Group 4: elasticnet with stronger regularisation - More conservative
            {
                'classifier__C': [0.005, 0.007, 0.01],
                'classifier__penalty': ['elasticnet'],
                'classifier__solver': ['saga'],
                'classifier__class_weight': ['balanced'],
                'classifier__fit_intercept': [True],
                'classifier__tol': [1e-4],
                'classifier__l1_ratio': [0.2]
            },
            # Group 5: l2 with stronger regularization - Handle noise
            {
                'classifier__C': [0.005, 0.007, 0.01],
                'classifier__penalty': ['l2'],
                'classifier__solver': ['liblinear'],
                'classifier__class_weight': ['balanced'],
                'classifier__fit_intercept': [True],
                'classifier__tol': [1e-4]
            }
        ]
    }

    # Top performing models (to be filled based on base model results)
    TOP_MODELS = {
        'logistic_regression': {
            'base_performance': 0.6115,
            'sampling_method': 'smote',
            'model': LogisticRegression(
                class_weight='balanced',
                max_iter=3000,
                random_state=RANDOM_SEED,
                solver='liblinear'
            )
        },
        'svm': {
            'base_performance': 0.6091,
            'sampling_method': 'smote',
            'model': SVC(
                probability=True,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                cache_size=1000,
                max_iter=5000,
                tol=1e-4
            )
        },
        'lightgbm': {
            'base_performance': 0.6039,
            'sampling_method': 'smote',
            'model': lgb.LGBMClassifier(
                n_estimators=1000,
                class_weight='balanced',
                random_state=RANDOM_SEED,
                n_jobs=N_JOBS,
                verbose=-1
            )
        }
    }

    # Best performing model (to be filled based on HP tuning results)
    BEST_MODEL = {
        'name': 'logistic_regression',
        'sampling_method': 'smote',
        'model': LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_SEED,
        ),
        'hp_performance': 0.6380  # Based on previous HP tuning results
    }

    # Model save/load paths
    MODEL_PATHS = {
        'base_results': 'results/base_models_results/',
        'hp_results': 'results/hp_models_results/',
        'best_results': 'results/best_model_results/',
        'trained_models': 'trained_models/'
    }

    # === File Paths === #
    FILE_PATHS = {
        'train_data': 'Datasets/TrainDataset2024.csv',
        'test_data': 'Datasets/TestDatasetExample.csv',
        'results_dir': 'results/',
        'logs_dir': 'logs/',
        'visualisations_dir': 'visualisations/',
        'models_dir': 'models/'
    }

    # === Required Features === #
    MANDATORY_FEATURES = [
        'ER',
        'HER2',
        'Gene',
        'PgR',
        'Age',
        'TumourStage',
        'LNStatus',
        'ChemoGrade'
    ]

    # === Feature Types === #
    CATEGORICAL_FEATURES = [
        'ER',
        'HER2',
        'Gene',
        'PgR',
        'TumourStage',
        'LNStatus',
        'ChemoGrade'
    ]

    NUMERICAL_FEATURES = [
        'Age'
    ]

    # === Validation Thresholds === #
    MIN_SAMPLES_PER_CLASS = 10

    @classmethod
    def get_model_params(cls, model_name):
        """Get hyperparameter search space for a specific model"""
        param_mapping = {
            'random_forest': cls.RF_PARAMS,
            'lightgbm': cls.LGBM_PARAMS,
            'catboost': cls.CATBOOST_PARAMS,
            'svm': cls.SVM_PARAMS
        }
        return param_mapping.get(model_name, {})