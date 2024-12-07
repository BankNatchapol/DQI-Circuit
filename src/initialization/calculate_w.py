import numpy as np
from scipy.sparse.linalg import eigs
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import sys
sys.path.append(r".\src")
from utils import time_logger

@time_logger
def calculate_w_dense(A):
    """
    Calculate w for dense matrix.
    """
    # Use subset_by_index to get the largest eigenvalue and eigenvector
    largest_eigenvalue, largest_eigenvector = eigh(
        A, subset_by_index=[A.shape[0] - 1, A.shape[0] - 1]
    )
    w = largest_eigenvector.flatten()
    w /= np.linalg.norm(w)
    return w


@time_logger
def calculate_w_sparse(A):
    """
    Calculate w for sparse matrix.
    """
    largest_eigenvalue, largest_eigenvector = eigs(A, k=1, which='LM')
    w = largest_eigenvector.flatten().real
    w /= np.linalg.norm(w)
    return w

if __name__ == "__main__":
    
    from construct_A import original_construct_A_matrix, construct_A_matrix

    # Benchmarking
    matrix_sizes = [10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000] # , 25000, 50000, 75000, 100000]
    dense_construction_times = []
    sparse_construction_times = []
    dense_w_times = []
    sparse_w_times = []

    m, p, r = 100000, 2, 1  # Fixed parameters

    for ell in matrix_sizes:
        print(f"Processing matrix size ell={ell}...")

        # Dense matrix
        A_dense, dense_construction_time = original_construct_A_matrix(m, ell, p, r)
        _, dense_w_time = calculate_w_dense(A_dense)

        # Sparse matrix
        A_sparse, sparse_construction_time = construct_A_matrix(m, ell, p, r)
        _, sparse_w_time = calculate_w_sparse(A_sparse)

        # Record times
        dense_construction_times.append(dense_construction_time)
        sparse_construction_times.append(sparse_construction_time)
        dense_w_times.append(dense_w_time)
        sparse_w_times.append(sparse_w_time)

    # Plotting results
    plt.figure(figsize=(12, 7))
    plt.plot(matrix_sizes, dense_w_times, label="Dense w Calculation", marker='o')
    plt.plot(matrix_sizes, sparse_w_times, label="Sparse w Calculation", marker='o')
    plt.xlabel("Matrix Size (ℓ + 1)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Benchmark: w Calculation (Dense vs Sparse)")
    plt.legend()
    plt.grid(True)
    plt.savefig("assets/execution_time_w_calculation.png", dpi=300)
    plt.show()