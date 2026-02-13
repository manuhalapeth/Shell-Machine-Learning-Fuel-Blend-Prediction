import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error


def preprocess_and_save(
    train_path='data/raw/train.csv',
    output_path='data/processed/train_processed.pkl',
    preprocessor_path='data/processed/train_preprocessor.joblib',
    target_cols=[f'BlendProperty{i}' for i in range(1, 11)],
    evaluate=True
):
    """
    Preprocess numeric features from the training dataset, optionally evaluate
    a baseline model, and save processed data and the fitted preprocessor.

    Steps:
        - Load raw training CSV
        - Select numeric input features and multi-output targets
        - Construct preprocessing pipeline (imputation + scaling)
        - Optionally evaluate a baseline multi-output gradient boosting model
        - Transform full dataset
        - Save processed features and targets
        - Save fitted preprocessor

    Args:
        train_path (str): Path to raw training CSV.
        output_path (str): Path to save processed features and targets (joblib).
        preprocessor_path (str): Path to save fitted preprocessor (joblib).
        target_cols (list of str): List of target column names.
        evaluate (bool): Whether to perform a baseline model evaluation.
    """
    print(f" Reading training data from {train_path}")
    df = pd.read_csv(train_path)

    # Select numeric input features and target columns
    X = df.iloc[:, :55]
    y = df[target_cols]
    numeric_features = X.columns.tolist()

    # Preprocessing pipeline for numeric features: mean imputation + standard scaling
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features)
    ])

    # Fit the preprocessor on the full feature matrix
    print("⚙️ Fitting preprocessor...")
    preprocessor.fit(X)

    # Optional baseline evaluation using multi-output HistGradientBoosting
    if evaluate:
        print(" Evaluating baseline model...")
        from sklearn.multioutput import MultiOutputRegressor

        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Pipeline: preprocessor + multi-output regressor
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', MultiOutputRegressor(HistGradientBoostingRegressor(random_state=42)))
        ])

        # Fit model and compute validation MAPE
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mape_score = mean_absolute_percentage_error(y_val, y_pred)
        print(f" Validation MAPE: {mape_score:.4f}")

    # Transform the full dataset using fitted preprocessor
    print(" Transforming full data and saving...")
    X_processed = preprocessor.transform(X)

    # Save processed features and target values
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump((X_processed, y.values), output_path)  # Ensure y is 2D NumPy array

    # Save fitted preprocessor
    joblib.dump(preprocessor, preprocessor_path)
    print(" Preprocessing complete.")


if __name__ == "__main__":
    preprocess_and_save()
