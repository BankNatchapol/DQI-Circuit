import time

def time_logger(func):
    """
    A decorator to log the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.6f} seconds.")
        total_time = end_time - start_time
        return result, total_time 
    return wrapper

def expected_number_of_satisfied_constraints(x):
    pass