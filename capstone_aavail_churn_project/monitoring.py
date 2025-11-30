from time import time
from .logging_utils import logger

class SimpleMonitor:
    def __init__(self):
        self.request_count = 0

    def log_request(self, endpoint: str, duration: float):
        self.request_count += 1
        logger.info(f"endpoint={endpoint} duration_ms={duration*1000:.1f} count={self.request_count}")

monitor = SimpleMonitor()

def monitored(endpoint_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time() - start
                monitor.log_request(endpoint_name, duration)
        return wrapper
    return decorator
