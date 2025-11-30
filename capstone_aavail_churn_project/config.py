import os
from dataclasses import dataclass

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

@dataclass
class Config:
    data_path: str = os.path.join(BASE_DIR, "data", "sample_aavail_churn.csv")
    model_path: str = os.path.join(BASE_DIR, "models", "model.joblib")
    log_path: str = os.path.join(BASE_DIR, "logs", "app.log")

config = Config()
