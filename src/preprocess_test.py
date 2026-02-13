import pandas as pd
import numpy as np
import os
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import __version__ as skl_version
from packaging import version


def preprocess_train_and_save(
    train_data_path="data/raw/train.csv",
    output_preprocessor_path="data/processed/train_preprocessor.joblib",
    output_data_path="data/processed/train_processed.pkl"
):
    """
    Preprocess training data and save both the fitted preprocessor and processed dataset.

    Steps:
        - Load raw training data
        - Split features and targets
        - Identify numerical and categorical columns
        - Construct preprocessing pipelines for both types
        - Fit and transform features
        - Save the preprocessor and processed features + targets

    Args:
        train_data_path (str): Path to the raw training CSV file.
        output_preprocessor_path (str): Path to save the fitted preprocessor object (joblib).
        output_data_path (str): Path to save the processed features and targets (joblib).
    """
    print(f" Reading training data from {train_data_path}")
    df = pd.read_csv(train_data_path)
    df.columns = df.columns.str.strip()  # Remove any whitespace from column names

    # Split into features and multi-output targets
    X = df.iloc[:, :55]
    y = df.iloc[:, 55:]

    # Identify numerical and categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Numerical preprocessing pipeline: impute missing values and scale
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing pipeline: impute missing values and one-hot encode
    if version.parse(skl_version) >= version.parse("1.2"):
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", encoder)
    ])

    # Combine numerical and categorical pipelines
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    # Fit and transform the features
    print(" Fitting preprocessor...")
    X_processed = preprocessor.fit_transform(X)

    # Save the fitted preprocessor
    print(f" Saving preprocessor to {output_preprocessor_path}")
    os.makedirs(os.path.dirname(output_preprocessor_path), exist_ok=True)
    joblib.dump(preprocessor, output_preprocessor_path)

    # Save processed features and targets
    print(f" Saving processed X and y to {output_data_path}")
    os.makedirs(os.path.dirname(output_data_path), exist_ok=True)
    joblib.dump((X_processed, y), output_data_path)

    print(" Training preprocessing complete.")


if __name__ == "__main__":
    preprocess_train_and_save()
