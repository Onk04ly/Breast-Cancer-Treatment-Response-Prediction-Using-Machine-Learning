import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split


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


def random_forest_feature_importance(X, y):
    """Use Random Forest to determine feature importance."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    selector = SelectFromModel(rf, prefit=True)
    feature_idx = selector.get_support()
    feature_names = X.columns[feature_idx]
    return feature_names


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


def main():
    # Load the data
    data = load_data('Train_Data.csv')

    if data is not None:
        # Preprocess and select features
        X = preprocess_and_select_features(data)
        y = data['pCR (outcome)']

        # Remove rows where the target variable is NaN
        y_filtered = y.dropna()
        X_filtered = X.iloc[y_filtered.index]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, stratify=y_filtered,
                                                            random_state=42)

        print("Final selected features:", X.columns.tolist())
        print("Preprocessed features shape:", X.shape)
        print("Training set shape:", X_train.shape)
        print("Testing set shape:", X_test.shape)


if __name__ == "__main__":
    main()
