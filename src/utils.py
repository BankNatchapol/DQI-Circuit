import time
from itertools import product
import qiskit

def time_logger(func):
    """
    A decorator to log the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        # print(f"Function '{func.__name__}' executed in {end_time - start_time:.6f} seconds.")
        total_time = end_time - start_time
        return result, total_time 
    return wrapper

def expected_number_of_satisfied_constraints(x):
    pass

def binary_combinations(n):
    # Generate all combinations of binary of length n
    return [''.join(map(str, bits)) for bits in product([0, 1], repeat=n)]

def choose(n, k):
    if k == 0:
        return 1
    return (n * choose(n - 1, k - 1)) // k
