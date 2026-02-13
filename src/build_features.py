import pandas as pd
import numpy as np
import os

def load_raw_data(train_path='data/raw/train.csv', test_path='data/raw/test.csv'):
    """
    Load raw training and test datasets from CSV files.

    Args:
        train_path (str): Path to the raw training CSV file.
        test_path (str): Path to the raw test CSV file.

    Returns:
        tuple: (train_df, test_df) as pandas DataFrames.
    """
    print(f"Loading train data from {train_path}")
    train_df = pd.read_csv(train_path)  # Read training data
    print(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)    # Read test data
    return train_df, test_df


def engineer_features(df):
    """
    Perform feature engineering on a DataFrame.

    Generates additional features such as row-wise statistics,
    pairwise ratios, and interaction terms.

    Args:
        df (pd.DataFrame): Input DataFrame (raw train or test).

    Returns:
        pd.DataFrame: Feature matrix with new engineered features.
    """
    print("Engineering new features...")

    # Extract input features only (first 55 columns), skipping targets if present
    X = df.iloc[:, :55].copy()

    # Feature 1: Row-wise statistics
    X["row_sum"] = X.sum(axis=1)      # Sum across all columns for each row
    X["row_mean"] = X.mean(axis=1)    # Mean across all columns for each row
    X["row_std"] = X.std(axis=1)      # Standard deviation across all columns for each row

    # Feature 2: Pairwise ratio features between selected columns
    # Avoid division by zero by selecting columns where all values are non-zero
    safe_cols = [col for col in X.columns if (X[col] != 0).all()]
    if len(safe_cols) >= 2:
        X["ratio_1_2"] = X[safe_cols[0]] / X[safe_cols[1]]  # Ratio of first two safe columns

    # Feature 3: Interaction term (product of first two columns)
    X["interaction_1_2"] = X.iloc[:, 0] * X.iloc[:, 1]

    print(f"Feature matrix now has {X.shape[1]} features.")
    return X


def save_engineered_data(X_train, y_train, X_test, output_dir='data/processed/'):
    """
    Save the engineered training and test datasets to disk.

    Args:
        X_train (pd.DataFrame): Engineered training feature matrix.
        y_train (pd.DataFrame): Original training targets.
        X_test (pd.DataFrame): Engineered test feature matrix.
        output_dir (str): Directory to save processed CSV files.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Combine engineered features with target columns for training
    full_train = pd.concat([X_train, y_train], axis=1)
    train_path = os.path.join(output_dir, 'train_features.csv')
    test_path = os.path.join(output_dir, 'test_features.csv')

    # Save training and test datasets
    print(f"Saving engineered training data to {train_path}")
    full_train.to_csv(train_path, index=False)

    print(f"Saving engineered test data to {test_path}")
    X_test.to_csv(test_path, index=False)


def main():
    """
    Main pipeline for loading raw data, performing feature engineering,
    and saving the processed datasets.
    """
    # Load raw CSV files
    train_df, test_df = load_raw_data('data/raw/train.csv', 'data/raw/test.csv')

    # Split target columns from training data
    X_train_raw = train_df.iloc[:, :55]  # First 55 columns are features
    y_train = train_df.iloc[:, 55:]      # Remaining columns are targets

    # Perform feature engineering
    X_train_feat = engineer_features(train_df)  # Training features
    X_test_feat = engineer_features(test_df)    # Test features

    # Save engineered features to disk
    save_engineered_data(X_train_feat, y_train, X_test_feat, output_dir='data/processed/')

    print(" Feature building complete.")


if __name__ == "__main__":
    main()
