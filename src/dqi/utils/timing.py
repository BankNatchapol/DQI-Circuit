import time
import logging
from functools import wraps

logger = logging.getLogger("dqi")

def time_logger(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.6f}s")
        return result, elapsed
    return wrapper
