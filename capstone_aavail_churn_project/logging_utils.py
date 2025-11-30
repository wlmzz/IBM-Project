import logging
import os
from .config import config

def setup_logging():
    os.makedirs(os.path.dirname(config.log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(config.log_path),
            logging.StreamHandler()
        ],
    )
    return logging.getLogger("aavail")

logger = setup_logging()
