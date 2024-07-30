from time import time
from typing import Dict
from logging_config import get_logger
import functools
import numpy as np

logger = get_logger(__name__)

def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        elapsed_time = end_time - start_time
        logger.info(f"Function {func.__name__} executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper

def normalize_features(X: np.ndarray, mean: np.ndarray = None, stddev: np.ndarray = None) -> Dict:
    """
    Subtract each column by the mean and divide by standard deviation. Store the mean
    and standard deviation values. They will be used in the predict method.
    """
    if mean is None and stddev is None: 
        mean = np.mean(X, axis=0)
        stddev = np.std(X, axis=0)
        normalized = (X - mean) / stddev
    elif mean is not None and stddev is not None:
        normalized = (X - mean) / stddev
    else:
        logger.error("Either have to specify both mean and stddev or not specify both.")
        return {}
    return {"normalized": normalized, "mean": mean, "stddev": stddev}