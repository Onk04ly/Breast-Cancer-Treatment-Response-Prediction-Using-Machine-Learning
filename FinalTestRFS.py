### Relapse-Free Survival Regression ###

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


def load_data(file_path):
    """Load dataset from a specified CSV file path."""
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully with shape:", data.shape)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None


def handle_missing_values(data):
    """Replace 999 with np.nan for missing values."""
    return data.replace(999, np.nan)


def handle_outliers(data):
    """Detect and handle outliers using IQR."""
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    for column in numeric_columns:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
    return data


def normalize_features(data):
    """Normalize numerical features using Min-Max scaling."""
    scaler = MinMaxScaler()
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    return data


def select_mandatory_features(data):
    """Select mandatory features that should be retained in all models."""
    mandatory_features = ['ER', 'HER2', 'Gene', 'PgR', 'Age', 'TumourStage', 'LNStatus', 'ChemoGrade']
    return data[mandatory_features]


def apply_pca_to_mri_features(data):
    """Apply PCA to reduce MRI features to explain 95% variance."""
    mri_features = [col for col in data.columns if col.startswith('original_')]
    scaler = StandardScaler()
    mri_data_scaled = scaler.fit_transform(data[mri_features])
    pca = PCA(n_components=0.95)
    principal_components = pca.fit_transform(mri_data_scaled)
    n_components = pca.n_components_
    print(f"Number of PCA components retained: {n_components}")
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i + 1}' for i in range(n_components)])
    return pca_df


def correlation_analysis(data, threshold=0.9):
    """Identify and remove highly correlated features."""
    correlation_matrix = data.corr().abs()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    return data.drop(columns=to_drop)


def preprocess_and_select_features(data):
    """Preprocess the dataset and select features."""
    data = handle_missing_values(data)
    data = handle_outliers(data)
    data = normalize_features(data)
    mandatory_features = select_mandatory_features(data)
    pca_features = apply_pca_to_mri_features(data)
    selected_features = pd.concat([mandatory_features, pca_features], axis=1)
    uncorrelated_features = correlation_analysis(selected_features)
    return uncorrelated_features



class Regressor:
    def __init__(self, models, param_distributions):
        """
        Initialize the Regressor class.

        :param models: Dictionary of model names and their corresponding estimators.
        :param param_distributions: Dictionary of model names and their hyperparameter search spaces.
        """
        self.models = models
        self.param_distributions = param_distributions
        self.best_model = None

    def train(self, X, y, validation_size=0.2, n_iter=50, random_state=42):
        """
        Train the models with hyperparameter tuning and find the best model.

        :param X: Training features.
        :param y: Training target.
        :param validation_size: Proportion of training data to use as the validation set.
        :param n_iter: Number of iterations for RandomizedSearchCV.
        :param cv: Number of cross-validation folds.
        :param random_state: Random seed.
        """
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_size, random_state=random_state
        )

        best_score = float('inf')

        for model_name, model in self.models.items():
            print(f"Training {model_name}")

            # Perform RandomizedSearchCV
            searchCV = RandomizedSearchCV(
                estimator=model,
                param_distributions=self.param_distributions[model_name],
                n_iter=n_iter,
                scoring='neg_mean_absolute_error',
                cv=KFold(n_splits=5, shuffle=True, random_state=42),
                random_state=random_state,
                n_jobs=-1
            )

            searchCV.fit(X_val, y_val)

            # Get the best estimator
            tuned_model = searchCV.best_estimator_

            # Fit the tuned model on the training data
            tuned_model.fit(X_train, y_train)

            # Evaluate on the validation set
            predictions = tuned_model.predict(X_val)
            mae = mean_absolute_error(y_val, predictions)
            r2 = r2_score(y_val, predictions)
            mse = mean_squared_error(y_val, predictions)

            print(f"{model_name} MAE: {mae}")
            print(f"{model_name} r2: {r2}")
            print(f"{model_name} MSE: {mse}")

            # Update the best model if this one is better
            if mae < best_score:
                best_score = mae
                self.best_model = tuned_model

        print(f"Best model: {self.best_model}")

    def predict(self, X):
        """
        Predictions using the best model.

        :param- X: Input test features.
        :return: Best Model predictions.
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet!")
        return self.best_model.predict(X)

    def save_predictions(self, predictions, ids, output_file):
        """
        Save predictions to a CSV file.

        :param- predictions: Best model's predictions.
        :param- output_file: Path to the CSV file where we want to save the output.
        """
        pd.DataFrame({
            'id': ids,
            'predictions': predictions
            }).to_csv(output_file, index=False)


"""Define models and their hyperparameter distributions"""
models = {
    'Ridge': Ridge(),
    'SVR': SVR(),
    'RandomForest': RandomForestRegressor(random_state=42),
    'MLPRegressor': MLPRegressor(random_state=42)
}

param_distributions = {
    'Ridge': {
        'alpha': [0.01, 0.1, 1, 10, 100]
    },
    'SVR': {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.2, 0.5],
        'kernel': ['poly', 'rbf', 'sigmoid'],
        'gamma': ['auto', 'scale', 0.01, 0.1, 1]
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 20, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    },
    'MLPRegressor': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100,100,50)],
        'activation': ['relu', 'tanh', 'logistic'],
        'alpha': [0.0001, 0.001, 0.01, 0.1],
        'solver': ['adam', 'sgd'],
        'learning_rate': ['constant', 'adaptive']
    }
}


def runFinalTestRFS():
    """Loading the training dataset as pandas dataframes"""
    train_data = load_data('./Datasets/TrainDataset2024.csv')

    """Storing the id of all the entries"""
    ids = train_data['ID']

    """Creating target variable"""
    target = train_data['RelapseFreeSurvival (outcome)']

    """Removing the target columns and ID columns before feature selection"""
    train_data_new = train_data.drop(columns=['RelapseFreeSurvival (outcome)','pCR (outcome)', 'ID'])

    """Preprocess the dataset and select features."""
    data = handle_missing_values(train_data_new)
    data = handle_outliers(data)
    data = normalize_features(data)
    mandatory_features = select_mandatory_features(data)

    """Apply PCA to reduce MRI features to explain 95% variance."""
    mri_features = [col for col in data.columns if col.startswith('original_')]
    scaler = StandardScaler()
    mri_data_scaled = scaler.fit_transform(data[mri_features])
    pca = PCA(n_components=0.95)
    principal_components_trained = pca.fit(mri_data_scaled)
    principal_components = pca.transform(mri_data_scaled)
    n_components = pca.n_components_
    print(f"Number of PCA components retained: {n_components}")
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i + 1}' for i in range(n_components)])

    """Combining the mandatory columns and columns after feature selection"""
    selected_features = pd.concat([mandatory_features, pca_df], axis=1)

    """New training dataset after preprocessing"""
    X_train = selected_features
    y_train = target

    """Adding ID column back to the dataset to keep of track what IDs are removed in the next step"""
    X_train.insert(0, 'ID', ids)

    """Remove rows where the target variable is NaN"""
    X_train_filtered = X_train.dropna()
    y_train_filtered = y_train.iloc[X_train_filtered.index]

    """Split the data into training and testing sets"""
    X_train, X_test, y_train, y_test = train_test_split(X_train_filtered, y_train_filtered, test_size=0.3, random_state=42)

    """keeping track of ID of training and test dataset"""
    ids_train = X_train['ID']
    ids_test = X_test['ID']

    """droping ID column because we dont want to use it as a feature"""
    X_train_no_id = X_train.drop(columns=['ID'])
    X_test_no_id = X_test.drop(columns=['ID'])

    regressor = Regressor(models=models, param_distributions=param_distributions)
    regressor.train(X=X_train_no_id, y=y_train)

    """Make predictions using the best model"""
    predictions = regressor.predict(X_test_no_id)
    mae = mean_absolute_error(y_test, predictions)
    print(f"MAE: {mae}")

    """Testing the test set"""
    """Loading the test dataset as pandas dataframes"""
    test_data = pd.read_excel('./Downloads/FinalTestDataset2024.csv')

    ids_test_new = test_data['ID']
    test_data_new = test_data.drop(columns=['ID'])

    """Preprocess the test dataset and select features."""
    data_test = handle_missing_values(test_data_new)
    data_test = handle_outliers(data_test)
    data_test = normalize_features(data_test)
    mandatory_features_test = select_mandatory_features(data_test)

    """Apply PCA to reduce MRI features to explain 95% variance."""
    mri_features_test = [col for col in data_test.columns if col.startswith('original_')]
    scaler = StandardScaler()
    mri_data_scaled_test = scaler.fit_transform(data_test[mri_features_test])
    principal_components_test = pca.transform(mri_data_scaled_test)
    n_components_test = pca.n_components_
    print(f"Number of PCA components retained: {n_components_test}")
    pca_df_test = pd.DataFrame(data=principal_components_test, columns=[f'PC{i + 1}' for i in range(n_components_test)])

    """Combining the mandatory columns and columns after feature selection"""
    selected_features_test = pd.concat([mandatory_features_test, pca_df_test], axis=1)

    """New test dataset after preprocessing"""
    X_test_new = selected_features_test


    """Adding ID column back to the dataset to keep of track what IDs are removed in the next step"""
    X_test_new.insert(0, 'ID', ids_test_new)

    """Remove rows where the target variable is NaN"""
    X_test_new_filtered = X_test_new.dropna()

    """keeping track of ID of test dataset"""
    ids_test_new = X_test_new_filtered['ID']

    """droping ID column because we dont want to use it as a feature"""
    X_test_new_no_id = X_test_new_filtered.drop(columns=['ID'])

    """Making new predictions using the best model"""
    predictions_new = regressor.predict(X_test_new_no_id)

    """Saving predictions to a file"""
    regressor.save_predictions(predictions_new, ids_test_new, "./Datasets/FinalTestDataset2024.xls")