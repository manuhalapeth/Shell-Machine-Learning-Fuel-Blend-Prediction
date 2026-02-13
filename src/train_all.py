import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

def load_processed_data(path='data/processed/train_processed.pkl'):
    print(f"Loading preprocessed training data from {path}")
    X, y = joblib.load(path)
    return X, y

def train_one_model(X, y, target_index, output_dir='models/', evaluate=True):
    target_name = f"property_{target_index + 1}"
    print(f"\n[INFO] Training model for {target_name}...")

    model = HistGradientBoostingRegressor(random_state=42)

    if evaluate:
        print(" - Performing train/val split for evaluation")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y[:, target_index], test_size=0.2, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mape = mean_absolute_percentage_error(y_val, y_pred)
        print(f" - Validation MAPE for {target_name}: {mape:.4f}")
    else:
        print(" - Training on full dataset")
        model.fit(X, y[:, target_index])

    # Save the model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"model_property{target_index + 1}.pkl")
    joblib.dump(model, model_path)
    print(f" - Saved model to {model_path}")

def train_all_models(evaluate=True):
    X, y_df = load_processed_data('data/processed/train_processed.pkl')
    
    # Convert y to numpy array for indexing
    if isinstance(y_df, pd.DataFrame):
        y = y_df.values
    else:
        y = y_df  # already numpy

    n_targets = y.shape[1]
    print(f"\n[INFO] Detected {n_targets} target properties.")
    
    for i in range(n_targets):
        train_one_model(X, y, target_index=i, evaluate=evaluate)

    print("\nDone training all models.")

if __name__ == "__main__":
    train_all_models(evaluate=True)
