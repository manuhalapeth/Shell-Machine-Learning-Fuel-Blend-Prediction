import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


def load_data(path='data/processed/train_processed.pkl'):
    """
    Load processed training data from a joblib file.

    Args:
        path (str): Path to the processed training data file.

    Returns:
        tuple: 
            X (np.ndarray or pd.DataFrame) - Feature matrix
            y (np.ndarray) - Target values
            target_names (list of str) - List of target column names
    """
    print(f"Loading processed training data from {path}")
    X, y_df = joblib.load(path)

    if isinstance(y_df, pd.DataFrame):
        y = y_df.values
        target_names = y_df.columns.tolist()
    else:
        y = y_df
        target_names = [f"property_{i+1}" for i in range(y.shape[1])]

    return X, y, target_names


def evaluate_single_model(model_path='models/model_all_targets.pkl', X=None, y=None):
    """
    Evaluate a single multi-output regression model across all targets.

    Performs a train/validation split, fits the model, generates predictions,
    and computes MAPE for each target and overall.

    Args:
        model_path (str): Path to the trained multi-output model.
        X (np.ndarray or pd.DataFrame): Feature matrix.
        y (np.ndarray): Target matrix.
    """
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    print("Splitting data for evaluation...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Fitting model (if not already fitted)...")
    model.fit(X_train, y_train)

    print("Generating predictions...")
    y_pred = model.predict(X_val)

    print("\n Evaluation (Single MultiOutput Model)")
    mape_scores = []
    for i in range(y.shape[1]):
        mape = mean_absolute_percentage_error(y_val[:, i], y_pred[:, i])
        print(f" - MAPE for property_{i+1}: {mape:.4f}")
        mape_scores.append(mape)
    print(f"  Overall MAPE: {np.mean(mape_scores):.4f}")


def evaluate_multiple_models(model_dir='models/', X=None, y=None, target_names=None):
    """
    Evaluate individual models for each target property.

    Each target has its own model. Performs train/validation split for each target,
    fits the model, generates predictions, and computes MAPE.

    Args:
        model_dir (str): Directory containing the individual target models.
        X (np.ndarray or pd.DataFrame): Feature matrix.
        y (np.ndarray): Target matrix.
        target_names (list of str): List of target column names.
    """
    print(f"Evaluating 10 individual models from {model_dir}")
    mape_scores = []

    for i in range(y.shape[1]):
        model_path = os.path.join(model_dir, f"model_property{i+1}.pkl")
        print(f"\n🔍 Evaluating {target_names[i]} from {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing: {model_path}")

        model = joblib.load(model_path)

        # Train/validation split for each target separately
        X_train, X_val, y_train, y_val = train_test_split(
            X, y[:, i], test_size=0.2, random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        mape = mean_absolute_percentage_error(y_val, y_pred)
        print(f" - MAPE: {mape:.4f}")
        mape_scores.append(mape)

    print(f"\n Overall MAPE across 10 models: {np.mean(mape_scores):.4f}")


def main():
    """
    Main evaluation pipeline:
    - Loads processed data
    - Evaluates a single multi-output model
    - Evaluates 10 individual target models
    """
    X, y, target_names = load_data('data/processed/train_processed.pkl')

    print("\n====== Evaluating Single Model ======")
    evaluate_single_model('models/model_all_targets.pkl', X, y)

    print("\n====== Evaluating 10 Individual Models ======")
    evaluate_multiple_models('models/', X, y, target_names)


if __name__ == "__main__":
    main()
