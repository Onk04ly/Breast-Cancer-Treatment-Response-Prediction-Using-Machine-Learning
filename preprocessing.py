### Data Preprocessing ###

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from constants import Constants


class Preprocessor:
    def __init__(self):
        """
        | Initialises the Preprocessor class for handling data loading, cleaning, transformation, and feature selection.
        | Authors: Benjamin Adonis
        """
        self.scaler = MinMaxScaler()
        self.pca = PCA(n_components=Constants.PCA_VARIANCE_THRESHOLD)
        self.random_state = Constants.RANDOM_SEED
        self.preprocessing_steps = []

    def _log_step(self, step_name, details=None):
        """
        | Log a preprocessing step with optional details
        :param step_name: Name of the preprocessing step
        :param details: Optional details about the step
        | Authors: Benjamin Adonis
        """
        step_info = {'step': step_name}
        if details:
            step_info['details'] = details
        self.preprocessing_steps.append(step_info)

    def load_data(self, filepath):
        """
        | Load dataset from the specified CSV file path.
        :param filepath: Path to the CSV file
        | Authors: Onkar Thete, edited by Benjamin Adonis
        """
        try:
            # Use pandas read_excel for .xlsx files
            if filepath.endswith('.xlsx'):
                data = pd.read_excel(filepath)
            # Use read_csv for .csv files
            elif filepath.endswith('.csv'):
                data = pd.read_csv(filepath)
            else:
                raise ValueError(f"Unsupported file type: {filepath}")

            self._log_step('data_loading', f'Loaded data with shape: {data.shape}')
            print(f"Data loaded successfully with shape: {data.shape}")
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None
        except Exception as e:
            print(f"Error loading file {filepath}: {e}")
            return None

    def handle_missing_values(self, data):
        """
        | Replace placeholders for missing values (e.g., 999) with NaN.
        :param data: Input data with missing values
        | Authors: Onkar Thete, edited by Benjamin Adonis
        """
        data = data.copy()
        initial_missing = data.isna().sum().sum()
        data = data.replace(999, np.nan)
        # Handle numeric columns
        numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_columns:
            if data[col].isnull().any():
                median_val = data[col].median()
                data[col] = data[col].fillna(median_val)
        final_missing = data.isna().sum().sum()
        self._log_step('missing_values_handling',
                       f'Replaced 999 with NaN and filled numeric missings. '
                       f'Initial missing: {initial_missing}, Final missing: {final_missing}')
        return data

    def handle_outliers(self, data):
        """
        | Handle outliers in numeric columns using the IQR method.
        :param data: Input data with potential outliers
        | Authors: Onkar Thete, edited by Benjamin Adonis
        """
        numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
        outliers_removed = 0

        for column in numeric_columns:
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers_before = ((data[column] < lower_bound) | (data[column] > upper_bound)).sum()
            data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
            outliers_removed += outliers_before
        # Benjamin Adonis extension for logging
        self._log_step('outlier_handling', f'Removed {outliers_removed} outliers using IQR method')
        return data

    def normalise_features(self, data):
        """
        | Normalise numerical features using Min-Max scaling.
        :param data: Input data with numerical features
        | Authors: Onkar Thete, edited by Benjamin Adonis
        """
        numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
        data[numeric_columns] = self.scaler.fit_transform(data[numeric_columns])
        # Benjamin Adonis extension for logging
        self._log_step('normalization', f'Normalized {len(numeric_columns)} numeric features')
        return data

    def select_mandatory_features(self, data):
        """
        | Retain only mandatory features specified in Constants.
        :param data: Input data with all features
        | Authors: Onkar Thete, edited by Benjamin Adonis
        """
        # Check if all mandatory features exist in the dataset
        missing_features = set(Constants.MANDATORY_FEATURES) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing mandatory features: {missing_features}")
        # Select only the mandatory features
        mandatory_data = data[Constants.MANDATORY_FEATURES].copy()
        self._log_step('mandatory_feature_selection',
                       f'Selected {len(Constants.MANDATORY_FEATURES)} mandatory features')
        return mandatory_data

    def apply_pca_to_mri_features(self, data):
        """
        | Apply PCA to reduce MRI features while retaining the specified variance.
        :param data: Input data with MRI features
        | Authors: Onkar Thete, edited by Benjamin Adonis
        """
        mri_features = [col for col in data.columns if col.startswith('original_')]
        if not mri_features:
            self._log_step('pca_transformation', 'No MRI features found to apply PCA')
            return pd.DataFrame(index=data.index)
        # Standardise MRI features before PCA
        scaler = StandardScaler()
        mri_data_scaled = scaler.fit_transform(data[mri_features])
        # Apply PCA
        principal_components = self.pca.fit_transform(mri_data_scaled)
        n_components = self.pca.n_components_
        explained_variance = np.sum(self.pca.explained_variance_ratio_) * 100
        self._log_step('pca_transformation',
                       f'Reduced {len(mri_features)} features to {n_components} components, '
                       f'explaining {explained_variance:.2f}% of variance')
        # Create DataFrame with PCA components
        pca_df = pd.DataFrame(
            principal_components,
            columns=[f'PC{i + 1}' for i in range(n_components)],
            index=data.index
        )
        return pca_df

    def correlation_analysis(self, data, threshold=Constants.CORRELATION_THRESHOLD):
        """
        | Remove highly correlated features based on the specified threshold.
        :param data: Input data with features
        :param threshold: Threshold for removing correlated features
        | Authors: Onkar Thete, edited by Benjamin Adonis
        """
        if data.empty:
            return data

        correlation_matrix = data.corr().abs()
        upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        # Benjamin Adonis extension for logging
        self._log_step('correlation_analysis',
                       f'Removed {len(to_drop)} features with correlation > {threshold}')
        return data.drop(columns=to_drop)

    def get_preprocessing_steps(self):
        """
        | Get a formatted string of all preprocessing steps performed
        :return: String containing preprocessing steps
        | Authors: Benjamin Adonis
        """
        if not self.preprocessing_steps:
            return "No preprocessing steps recorded"
        formatted_steps = []
        for i, step in enumerate(self.preprocessing_steps, 1):
            step_str = f"{i}. {step['step']}"
            if 'details' in step:
                step_str += f": {step['details']}"
            formatted_steps.append(step_str)
        return "\n".join(formatted_steps)

    def preprocess(self, data, target_column):
        """
        | Perform the full preprocessing pipeline.
        :param data: Input data to preprocess
        | Authors: Onkar Thete, edited by Benjamin Adonis
        """
        # Reset preprocessing steps for new run
        self.preprocessing_steps = []

        # Make a copy of the input data
        data = data.copy()
        # Clean target column
        data[target_column] = data[target_column].replace(999, np.nan)
        target = data[target_column].copy()
        # Apply preprocessing steps
        data = self.handle_missing_values(data)
        data = self.handle_outliers(data)
        data = self.normalise_features(data)
        # Extract features
        mandatory_features = self.select_mandatory_features(data)
        pca_features = self.apply_pca_to_mri_features(data)
        # Combine features if we have PCA components
        if not pca_features.empty:
            combined_features = pd.concat([mandatory_features, pca_features], axis=1)
        else:
            combined_features = mandatory_features
        # Remove highly correlated features
        final_features = self.correlation_analysis(combined_features)
        # Add target column back
        final_features[target_column] = target

        return final_features

    def split_data(self, data, target_column):
        """
        | Split the dataset into training and testing sets.
        :param data: Input data with features and target column
        :param target_column: Name of the target column
        | Authors: Onkar Thete, edited by Benjamin Adonis
        """
        y = data[target_column].dropna()
        X = data.loc[y.index].drop(target_column, axis=1)

        if X.isnull().any().any():
            raise ValueError("NaN values detected in the feature set. Please preprocess the data accordingly.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )

        smote = SMOTE(random_state=self.random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        return X_train, X_test, y_train, y_test
