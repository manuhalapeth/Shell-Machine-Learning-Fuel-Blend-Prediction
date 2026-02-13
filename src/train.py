import os
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error


def load_processed_data(path='data/processed/train_processed.pkl'):
    """
    Load preprocessed training data (features and targets) from a joblib file.

    Args:
        path (str): Path to the processed training data file.

    Returns:
        tuple:
            X (np.ndarray): Processed feature matrix.
            y (np.ndarray): Target values for multiple outputs.
    """
    print(f"Loading preprocessed training data from {path}")
    X, y = joblib.load(path)
    return X, y


def train_and_evaluate(X, y, model_output_path='models/model_all_targets.pkl', evaluate=True):
    """
    Train a multi-output regression model and optionally evaluate its performance.

    Steps:
        - Initialize MultiOutputRegressor with HistGradientBoostingRegressor
        - If evaluation is enabled:
            - Split data into train and validation sets
            - Fit model on training set
            - Predict on validation set and compute MAPE
        - Otherwise, train on the full dataset
        - Save the trained model to disk

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target matrix.
        model_output_path (str): Path to save the trained model.
        evaluate (bool): Whether to perform train/validation evaluation.
    """
    print("Initializing model...")
    model = MultiOutputRegressor(HistGradientBoostingRegressor(random_state=42))

    if evaluate:
        print("Splitting into train/validation sets for evaluation...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Compute Mean Absolute Percentage Error
        mape = mean_absolute_percentage_error(y_val, y_pred)
        print(f"Validation MAPE: {mape:.4f}")
    else:
        print("Training on full dataset...")
        model.fit(X, y)

    # Save trained model
    print(f"Saving trained model to {model_output_path}")
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    joblib.dump(model, model_output_path)
    print("Training complete.")


def main():
    """
    Main function to load processed data and train the multi-output regression model.
    """
    X, y = load_processed_data('data/processed/train_processed.pkl')
    train_and_evaluate(X, y, model_output_path='models/model_all_targets.pkl', evaluate=True)


if __name__ == "__main__":
    main()
