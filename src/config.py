import os
from datetime import datetime

# =========================
# General Paths
# =========================

# Base directory of the project (three levels up from this script)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, "data")                 # Main data folder
RAW_DIR = os.path.join(DATA_DIR, "raw")                  # Raw/unprocessed data
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")      # Processed/engineered data

# Other project directories
MODEL_DIR = os.path.join(BASE_DIR, "models")             # Directory to save trained models
SUBMISSION_DIR = os.path.join(BASE_DIR, "submissions")   # Directory for submission CSVs
LOG_DIR = os.path.join(BASE_DIR, "experiments", "logs")  # Directory for experiment logs

# Specific file paths
TRAIN_PATH = os.path.join(RAW_DIR, "train.csv")                      # Raw training data
TEST_PATH = os.path.join(RAW_DIR, "test.csv")                        # Raw test data
TRAIN_PROCESSED_PATH = os.path.join(PROCESSED_DIR, "train_processed.pkl")  # Preprocessed train data
TEST_PROCESSED_PATH = os.path.join(PROCESSED_DIR, "test_processed.pkl")    # Preprocessed test data
PREPROCESSOR_PATH = os.path.join(PROCESSED_DIR, "train_preprocessor.pkl")  # Preprocessor object
SUBMISSION_TEMPLATE = os.path.join(RAW_DIR, "sample_submission.csv")       # Sample submission template


# =========================
# Model Configurations
# =========================

RANDOM_STATE = 42       # Fixed seed for reproducibility
N_TARGETS = 10          # Number of target properties
N_SPLITS = 5            # Number of splits for cross-validation or KFold

# Default model type and hyperparameters
DEFAULT_MODEL_TYPE = "HistGradientBoosting"
DEFAULT_MODEL_PARAMS = {
    "learning_rate": 0.1,        # Step size at each iteration
    "max_iter": 300,             # Maximum number of boosting iterations
    "max_depth": None,           # Maximum depth of each tree (None = unlimited)
    "l2_regularization": 0.0,   # L2 regularization term
    "early_stopping": True       # Stop early if validation score does not improve
}


# =========================
# Output File Naming Helpers
# =========================

def get_timestamp():
    """
    Generate a string timestamp in the format YYYYMMDD_HHMMSS.

    Returns:
        str: Current timestamp as a string.
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_submission_path():
    """
    Generate a file path for saving a submission CSV, including a timestamp.

    Returns:
        str: Full path for the submission file.
    """
    return os.path.join(SUBMISSION_DIR, f"submission_{get_timestamp()}.csv")


def get_model_path(property_index):
    """
    Generate a file path for saving a model for a specific target property.

    Args:
        property_index (int): Index of the target property (1-based).

    Returns:
        str: Full path for the model file.
    """
    return os.path.join(MODEL_DIR, f"model_property{property_index}.pkl")


def get_log_path():
    """
    Generate a file path for saving a log of the current experiment run.

    Returns:
        str: Full path for the log file.
    """
    return os.path.join(LOG_DIR, f"run_{get_timestamp()}.json")


# =========================
# Target Properties
# =========================

# List of target property column names
TARGET_NAMES = [
    "property_1", "property_2", "property_3", "property_4", "property_5",
    "property_6", "property_7", "property_8", "property_9", "property_10"
]
