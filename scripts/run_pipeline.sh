#!/bin/bash

set -e  # Exit on error

echo " Step 1: Feature Engineering"
python src/build_features.py

echo " Step 2: Preprocessing Training Data"
python src/preprocess.py

echo " Step 3: Preprocessing Test Data"
python src/preprocess_test.py

echo " Step 4: Training Models (All 10 Properties)"
python src/train_all.py

echo " Step 5: Generating Predictions for Submission"
python src/predict.py

echo " Pipeline completed successfully."
