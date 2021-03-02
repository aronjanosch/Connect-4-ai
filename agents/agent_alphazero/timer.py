import functools
import sys
import time
from typing import TextIO

logging_destination: TextIO = sys.stdout


def log_to(destination: TextIO):
    global logging_destination
    logging_destination = destination


# Used to print timing information for a given function
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        logging_destination.write(f"[TIMER] {func.__name__!r}: {run_time:.4f} seconds\n")
        return value

    return wrapper
