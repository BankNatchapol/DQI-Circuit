import numpy as np
from scipy.sparse.linalg import eigs, eigsh
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import sys
import math

sys.path.append(r".\src")
from utils import time_logger


@time_logger
def calculate_w_dense(A):
    """
    Calculate w for dense matrix.
    Uses subset_by_index to obtain the largest eigenvalue and corresponding eigenvector.
    """
    largest_eigenvalue, largest_eigenvector = eigh(
        A, subset_by_index=[A.shape[0] - 1, A.shape[0] - 1]
    )
    w = largest_eigenvector.flatten()
    w /= np.linalg.norm(w)
    return w


@time_logger
def calculate_w(A):
    """
    Calculate w for sparse matrix.
    Uses eigs to get the largest eigenvalue and corresponding eigenvector,
    then pads the result to the next power of 2.
    """
    largest_eigenvalue, largest_eigenvector = eigs(A, k=1, which='LM')
    w = largest_eigenvector.flatten().real
    w /= np.linalg.norm(w)

    original_size = w.shape[0]
    power = math.ceil(math.log2(original_size))
    target_size = 2 ** power
    padded_w = np.zeros(target_size, dtype=w.dtype)
    padded_w[:original_size] = w

    return padded_w


@time_logger
def get_optimal_w(m, n, l, p, r):
    """
    Compute the optimal w vector (w_k) for a max-xor SAT problem using sparse linear algebra.
    
    Parameters:
        m (int): Number of rows (used in off-diagonal calculation).
        n (int): Number of columns (unused in current formulation).
        l (int): Determines the size of the A matrix (l+1 diagonal entries).
    
    Returns:
        np.ndarray: The padded, normalized principal eigenvector of the constructed sparse A matrix.
    """

    d = (p - 2 * r) / np.sqrt(r * (p - r))

    # Build A matrix (sparse tridiagonal)
    diag = np.arange(l + 1) * d
    off_diag = np.array([np.sqrt(i * (m - i + 1)) for i in range(l)])
    
    # Construct the sparse matrix using scipy.sparse.diags
    from scipy.sparse import diags
    offsets = [-1, 0, 1]
    data = [off_diag, diag, off_diag]
    A = diags(data, offsets, format='csr')

    # Compute the principal eigenvector using a sparse eigensolver for symmetric matrices
    eigenvalues, eigenvector = eigsh(A, k=1, which='LA')  # 'LA' finds the largest algebraic eigenvalue
    principal_vector = eigenvector.flatten()
    optimal_w = principal_vector / np.linalg.norm(principal_vector)

    # Pad the vector to the next power of 2
    original_size = optimal_w.shape[0]
    target_size = 2 ** int(np.ceil(np.log2(original_size)))
    padded_optimal_w = np.pad(optimal_w, (0, target_size - original_size))

    return padded_optimal_w



if __name__ == "__main__":
    from construct_A import original_construct_A_matrix, construct_A_matrix

    # Benchmarking for w calculation using dense, sparse, and optimal (sparse) methods
    matrix_sizes = [10, 25, 50, 75, 100, 250, 500, 750, 1000, 2500, 5000, 7500]
    dense_w_times = []
    sparse_w_times = []
    optimal_w_times = []

    m, p, r = 100000, 2, 1  # Fixed parameters for matrix construction

    for ell in matrix_sizes:
        print(f"Processing matrix size ell={ell}...")

        # Dense matrix: build A_dense and compute w using the dense method.
        A_dense, _ = original_construct_A_matrix(m, ell, p, r)
        _, dense_w_time = calculate_w_dense(A_dense)

        # Sparse matrix: build A_sparse and compute w using the sparse method.
        A_sparse, _ = construct_A_matrix(m, ell, p, r)
        _, sparse_w_time = calculate_w(A_sparse)

        # Optimal w: use the same ell (equivalent to MAX_ERRORS) to build A and compute w.
        _, optimal_w_time = get_optimal_w(m=m, n=m, l=ell, p=p, r=r)

        dense_w_times.append(dense_w_time)
        sparse_w_times.append(sparse_w_time)
        optimal_w_times.append(optimal_w_time)

    # Plotting all three benchmark curves in a single figure for comparison.
    plt.figure(figsize=(12, 7))
    plt.plot(matrix_sizes, dense_w_times, label="Dense w Calculation", marker='o')
    plt.plot(matrix_sizes, sparse_w_times, label="Sparse w Calculation", marker='o')
    plt.plot(matrix_sizes, optimal_w_times, label="Optimal w Calculation (Sparse)", marker='o')
    plt.xlabel("Matrix Size (ℓ + 1)")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Benchmark: Comparison of w Calculation Methods")
    plt.legend()
    plt.grid(True)
    plt.savefig("assets/execution_time_comparison.png", dpi=300)
    plt.show()
