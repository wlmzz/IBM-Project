import os
from src.train_model import train_and_evaluate
from src.config import config

def test_training_produces_model_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "model_path", str(tmp_path / "model.joblib"))
    res = train_and_evaluate()
    assert os.path.exists(config.model_path)
    assert res["best_auc"] > 0.5
