import os
from src.logging_utils import logger
from src.config import config

def test_logging_creates_file():
    logger.info("Test log message from unit test")
    assert os.path.exists(config.log_path)
    with open(config.log_path, "r", encoding="utf-8") as f:
        txt = f.read()
    assert "Test log message from unit test" in txt
