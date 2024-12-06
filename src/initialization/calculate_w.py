import numpy as np
from scipy.sparse.linalg import eigs
from scipy.linalg import eig

def calculate_w(A):
    # Use sparse eigenvalue solver for sparse matrices
    largest_eigenvalue, largest_eigenvector = eigs(A, k=1, which='LM')  # 'LM' -> Largest Magnitude
    w = largest_eigenvector.flatten().real  # Extract the real part of the eigenvector

    # Normalize w
    w /= np.linalg.norm(w)
    return w

# Example usage
if __name__ == "__main__":
    from construct_A import construct_A_matrix
    # Define a sample dense A matrix
    m, ell, p, r = 10, 3, 2, 1

    # Define a sample sparse A matrix
    A_sparse, _ = construct_A_matrix(m, ell, p, r)
    print("Matrix A: \n", A_sparse.toarray())

    # Compute w for the sparse matrix
    w_sparse = calculate_w(A_sparse)
    print("\nNormalized eigenvector w (sparse):", w_sparse)
