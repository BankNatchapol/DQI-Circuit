import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt
import sys
sys.path.append(r".\src")
from utils import time_logger


# Define the original and optimized functions
@time_logger
def original_construct_A_matrix(m, ell, p, r):
    if m < ell:
        raise ValueError(f"Invalid input: m ({m}) must be greater or equal than ell ({ell}).")
    
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
    """Optimized version"""
    if m < ell:
        raise ValueError(f"Invalid input: m ({m}) must be greater or equal than ell ({ell}).")
    
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
    # Benchmarking
    matrix_sizes = [10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 25000, 50000, 75000, 100000]
    original_times = []
    optimized_times = []
    results_match = []  # To track if the results match

    m, p, r = 100000, 2, 1  # Fixed parameters

    for ell in matrix_sizes:
        if m < ell:
            print(f"Skipping ell={ell} since m ({m}) < ell ({ell}).")
            continue

        result_a, original_time = original_construct_A_matrix(m, ell, p, r)
        result_b, optimized_time = construct_A_matrix(m, ell, p, r)
        
        original_times.append(original_time)
        optimized_times.append(optimized_time)

        # Compare results
        if ell > 1000:
            # it's gonna take to long
            continue
        result_b_dense = result_b.toarray()  # Convert sparse matrix to dense
        match = np.allclose(result_a, result_b_dense)
        results_match.append(match)

    # Check if all results match
    all_match = all(results_match)
    print("All results match:", all_match)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(matrix_sizes[:len(original_times)], original_times, label="Original Function", marker='o')
    plt.plot(matrix_sizes[:len(optimized_times)], optimized_times, label="Optimized Function", marker='o')
    plt.xlabel("Matrix Size (ℓ + 1)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Benchmark: Execution Time vs Matrix Size")
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plot_filename = "assets/execution_time_A_construction.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved as {plot_filename}")

    # Show the plot
    plt.show()
