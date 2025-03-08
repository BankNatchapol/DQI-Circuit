#!/usr/bin/env python3
import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
import sys
sys.path.append(r".\src")
from utils import time_logger

@time_logger
def original_construct_A_matrix(m, ell, p, r):
    """
    Construct the A matrix using a naive dense approach.
    
    Parameters:
        m (int): Parameter m.
        ell (int): Determines the size of the matrix (matrix is (ell+1) x (ell+1)).
        p (int): Parameter p.
        r (int): Parameter r.
    
    Returns:
        np.ndarray: Dense A matrix.
    
    Raises:
        ValueError: If m < ell.
    """
    if m < ell:
        raise ValueError(f"Invalid input: m ({m}) must be >= ell ({ell}).")
    
    d = (p - 2 * r) / np.sqrt(r * (p - r))
    A = np.zeros((ell + 1, ell + 1))
    for i in range(ell + 1):
        if i < ell:
            k = i + 1
            a_k = np.sqrt(k * (m - k + 1))
            A[i, i + 1] = a_k
            A[i + 1, i] = a_k
        A[i, i] = i * d
    return A

@time_logger
def construct_A_matrix(m, ell, p, r):
    """
    Construct the A matrix using an optimized sparse approach.
    
    Parameters:
        m (int): Parameter m.
        ell (int): Determines the size of the matrix (matrix is (ell+1) x (ell+1)).
        p (int): Parameter p.
        r (int): Parameter r.
    
    Returns:
        scipy.sparse.csr_matrix: Sparse A matrix in CSR format.
    
    Raises:
        ValueError: If m < ell.
    """
    if m < ell:
        raise ValueError(f"Invalid input: m ({m}) must be >= ell ({ell}).")
    
    d = (p - 2 * r) / np.sqrt(r * (p - r))
    diag = np.arange(ell + 1) * d
    k_values = np.arange(1, ell + 1)
    off_diag = np.sqrt(k_values * (m - k_values + 1))
    A_sparse = diags(
        diagonals=[diag, off_diag, off_diag],
        offsets=[0, 1, -1],
        shape=(ell + 1, ell + 1),
        format="csr"
    )
    return A_sparse

if __name__ == "__main__":
    # List of matrix sizes (for ell)
    matrix_sizes = [10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000, 50000, 75000, 100000]
    original_times = []
    optimized_times = []
    results_match = []  # For checking dense vs. sparse results (only for small ell)

    # Fixed parameters for the tests
    m, p, r = 100000, 2, 1

    for ell in matrix_sizes:
        if m < ell:
            print(f"Skipping ell={ell} since m ({m}) < ell ({ell}).")
            continue

        result_a, original_time = original_construct_A_matrix(m, ell, p, r)
        result_b, optimized_time = construct_A_matrix(m, ell, p, r)
        
        original_times.append(original_time)
        optimized_times.append(optimized_time)

        # Compare results only for smaller matrices to avoid performance issues
        if ell <= 1000:
            result_b_dense = result_b.toarray()
            match = np.allclose(result_a, result_b_dense)
            results_match.append(match)

    all_match = all(results_match)
    print("All results match for small ell values:", all_match)

    # Plot execution time vs matrix size
    plt.figure(figsize=(10, 6))
    sizes = matrix_sizes[:len(original_times)]
    plt.plot(sizes, original_times, label="Original Function", marker='o')
    plt.plot(sizes, optimized_times, label="Optimized Function", marker='o')
    plt.xlabel("Matrix Size (ℓ + 1)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Benchmark: Execution Time vs Matrix Size")
    plt.legend()
    plt.grid(True)

    plot_filename = "assets/execution_time_A_construction.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved as {plot_filename}")
    plt.show()
