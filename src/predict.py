import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime


def load_test_data(path='data/processed/test_processed.pkl'):
    """
    Load preprocessed test dataset from a joblib file.

    Args:
        path (str): Path to the preprocessed test data file.

    Returns:
        np.ndarray or pd.DataFrame: Test feature matrix.
    """
    print(f" Loading preprocessed test data from: {path}")
    return joblib.load(path)


def load_or_create_submission_template(X_test, path='data/raw/sample_submission.csv'):
    """
    Load a submission template CSV if it exists, otherwise create a dummy template.

    Args:
        X_test (np.ndarray or pd.DataFrame): Test features, used to determine number of rows.
        path (str): Path to the submission template CSV.

    Returns:
        pd.DataFrame: Submission template with an 'id' column and 10 target property columns.
    """
    if os.path.exists(path):
        print(f" Loading submission template from: {path}")
        return pd.read_csv(path)
    
    print(f" Submission template not found at {path}. Creating one...")
    submission = pd.DataFrame()
    submission['id'] = np.arange(1, X_test.shape[0] + 1)  # Create ID column
    for i in range(10):
        submission[f'property_{i+1}'] = 0.0  # Initialize targets to 0.0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    submission.to_csv(path, index=False)
    print(f" Dummy submission template saved to: {path}")
    return submission


def load_models(model_dir='models/', n_targets=10):
    """
    Load individual models for each target property.

    Args:
        model_dir (str): Directory containing the model files.
        n_targets (int): Number of target properties/models to load.

    Returns:
        list: List of loaded models.
    
    Raises:
        FileNotFoundError: If any model file is missing.
    """
    print(f" Loading {n_targets} models from: {model_dir}")
    models = []
    for i in range(1, n_targets + 1):
        model_path = os.path.join(model_dir, f"model_property{i}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f" Model not found: {model_path}")
        models.append(joblib.load(model_path))
    return models


def predict_all(models, X_test):
    """
    Generate predictions for all target properties using individual models.

    Args:
        models (list): List of trained models for each property.
        X_test (np.ndarray or pd.DataFrame): Test feature matrix.

    Returns:
        np.ndarray: Predictions for all properties, shape (n_samples, n_targets)
    """
    print(" Predicting all target properties...")
    preds = []
    for i, model in enumerate(models):
        print(f"   └─ Predicting property_{i+1}")
        pred = model.predict(X_test).reshape(-1, 1)  # Ensure column vector
        preds.append(pred)
    return np.hstack(preds)  # Combine predictions horizontally


def save_submission(preds, template_df, output_dir='submissions/'):
    """
    Save predictions to a CSV file using a submission template.

    Args:
        preds (np.ndarray): Prediction array of shape (n_samples, 10).
        template_df (pd.DataFrame): Submission template with 'id' column.
        output_dir (str): Directory to save the submission file.

    Raises:
        ValueError: If predictions do not have 10 columns.
    """
    if preds.shape[1] != 10:
        raise ValueError("Prediction output must have 10 columns.")

    submission = template_df.copy()
    submission.iloc[:, 1:] = preds  # Fill target columns, keep ID intact

    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"submission_{timestamp}.csv"
    output_path = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f" Submission saved: {output_path}")


def main():
    """
    Main pipeline for generating a submission:
    - Load preprocessed test data
    - Load trained models
    - Predict all target properties
    - Load or create submission template
    - Save predictions to CSV
    """
    X_test = load_test_data('data/processed/test_processed.pkl')
    models = load_models('models/', n_targets=10)
    preds = predict_all(models, X_test)
    template_df = load_or_create_submission_template(X_test, path='data/raw/sample_submission.csv')
    save_submission(preds, template_df)


if __name__ == "__main__":
    main()
