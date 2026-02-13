import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error


class BaseModel:
    """
    BaseModel is a general-purpose regression model wrapper that supports both
    single-output and multi-output regression using scikit-learn models.
    
    Currently supports:
        - HistGradientBoostingRegressor (scikit-learn)
    
    Features:
        - Train with optional validation split and evaluation
        - Save and load trained model
        - Predict and evaluate using MAPE
    """
    
    def __init__(
        self,
        model_type='HistGradientBoosting',
        model_params=None,
        multi_output=True,
        random_state=42
    ):
        """
        Initialize the BaseModel.
        
        Args:
            model_type (str): Type of model to use ('HistGradientBoosting' supported).
            model_params (dict): Optional dictionary of model hyperparameters.
            multi_output (bool): Whether to wrap the model for multi-output regression.
            random_state (int): Random seed for reproducibility.
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.multi_output = multi_output
        self.random_state = random_state
        self.model = self._init_model()

    def _init_model(self):
        """
        Initialize the chosen scikit-learn model and optionally wrap for multi-output.
        
        Returns:
            model (sklearn estimator): Configured scikit-learn model.
        """
        if self.model_type == 'HistGradientBoosting':
            base = HistGradientBoostingRegressor(random_state=self.random_state, **self.model_params)
        else:
            raise NotImplementedError(f"Model type '{self.model_type}' is not supported yet.")

        if self.multi_output:
            return MultiOutputRegressor(base)
        return base

    def load_data(self, path='data/processed/train_processed.pkl'):
        """
        Load preprocessed training data from a joblib file.
        
        Args:
            path (str): Path to the processed training data file.
            
        Returns:
            X (pd.DataFrame / np.ndarray): Feature matrix.
            y (pd.DataFrame / np.ndarray): Target matrix.
        """
        print(f" Loading training data from: {path}")
        X, y = joblib.load(path)
        return X, y

    def train(self, X, y, evaluate=True, val_split=0.2):
        """
        Train the model on the provided dataset.
        
        Optionally performs a train/validation split to compute validation MAPE.
        
        Args:
            X (array-like): Features.
            y (array-like): Target variables.
            evaluate (bool): If True, performs train/validation split.
            val_split (float): Fraction of data to use as validation.
            
        Returns:
            float or None: Validation MAPE if evaluate=True, else None.
        """
        if evaluate:
            print(" Performing train/val split to evaluate performance...")
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=self.random_state)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
            mape = mean_absolute_percentage_error(y_val, y_pred)
            print(f" Validation MAPE: {mape:.4f}")
            return mape
        else:
            print(" Training on full dataset...")
            self.model.fit(X, y)
            return None

    def save(self, path):
        """
        Save the trained model to disk using joblib.
        
        Args:
            path (str): File path to save the model.
        """
        print(f" Saving model to: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path):
        print(f" Loading model from: {path}")
        self.model = joblib.load(path)

    def predict(self, X):
        """
        Generate predictions using the trained model.
        
        Args:
            X (array-like): Feature matrix for prediction.
            
        Returns:
            np.ndarray: Predicted target values.
        """
        print(f" Predicting on input of shape {X.shape}")
        return self.model.predict(X)

    def evaluate(self, X, y_true):
        """
        Evaluate the model on a test/validation set using MAPE.
        
        Args:
            X (array-like): Feature matrix.
            y_true (array-like): True target values.
            
        Returns:
            float: Mean Absolute Percentage Error (MAPE)
        """
        print(" Evaluating model...")
        y_pred = self.model.predict(X)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        print(f" MAPE: {mape:.4f}")
        return mape


if __name__ == "__main__":
    # Example run for model training (testing/debugging)
    model = BaseModel() # Initialize the BaseModel with default settings
    X, y = model.load_data('data/processed/train_processed.pkl') #Load training data 
    model.train(X, y, evaluate=True) # Train the model with validation split and print validation MAPE
    model.save('models/base_model_debug.pkl') # Save the trained model to disk
