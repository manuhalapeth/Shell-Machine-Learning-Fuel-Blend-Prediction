import os
import json
from datetime import datetime
from typing import Dict, Any


def get_log_dir():
    return os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")), "experiments", "logs")


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_log(run_data: Dict[str, Any], filename: str = None):
    """
    Save a dictionary of run metadata/config/metrics to a .json log file.
    
    Args:
        run_data (dict): All relevant info (params, scores, model paths, etc.)
        filename (str): Optional. If None, use timestamp-based default.
    """
    log_dir = get_log_dir()
    os.makedirs(log_dir, exist_ok=True)

    if filename is None:
        filename = f"run_{get_timestamp()}.json"
    
    log_path = os.path.join(log_dir, filename)

    with open(log_path, 'w') as f:
        json.dump(run_data, f, indent=4)

    print(f" Log saved to: {log_path}")


def log_example():
    # Example use case
    example_log = {
        "timestamp": get_timestamp(),
        "model": "HistGradientBoostingRegressor",
        "data_used": "data/processed/train_processed.pkl",
        "n_targets": 10,
        "params": {
            "learning_rate": 0.1,
            "max_iter": 300
        },
        "scores": {
            "val_mape": 0.0543
        },
        "model_files": [
            "models/model_property1.pkl",
            "models/model_property2.pkl",
            # ...
        ]
    }
    save_log(example_log)


if __name__ == "__main__":
    log_example()
