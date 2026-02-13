import joblib
import numpy as np
import sys

file_path = 'data/processed/train_processed.pkl'

try:
    data = joblib.load(file_path)
    print(" Loaded successfully")
except Exception as e:
    print(" Failed to load")
    print(f"Error: {e}")
    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
