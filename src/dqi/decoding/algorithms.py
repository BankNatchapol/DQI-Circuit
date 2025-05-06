from typing import List, Tuple

def gauss_jordan_operations_general(
    matrix: List[List[int]]
) -> Tuple[List[Tuple[str, int, int]], List[List[int]]]:
    """
    Perform Gauss–Jordan elimination mod 2 on an m×(n+1) augmented binary matrix [A|b].
    Records row ops:
      - ('swap', i, j): swap row i with j
      - ('xor', i, j): row j ^= row i

    Args:
        matrix: List of m rows, each n+1 bits (0/1).
    Returns:
        operations: list of (op, src_row, tgt_row)
        rref_matrix: resulting matrix in reduced row echelon form
    """
    # quick exit on empty or invalid matrix
    if not matrix or not matrix[0]:
        return [], []

    m = len(matrix)
    num_cols = len(matrix[0])
    n = num_cols - 1  # last column is RHS

    # work on a copy
    mat = [row.copy() for row in matrix]
    operations: List[Tuple[str, int, int]] = []

    pivot_row = 0
    pivot_col = 0

    while pivot_row < m and pivot_col < n:
        # find pivot in this column at or below pivot_row
        pivot_idx = None
        for r in range(pivot_row, m):
            if mat[r][pivot_col] == 1:
                pivot_idx = r
                break
        if pivot_idx is None:
            pivot_col += 1
            continue

        # swap into pivot position if needed
        if pivot_idx != pivot_row:
            operations.append(('swap', pivot_row, pivot_idx))
            mat[pivot_row], mat[pivot_idx] = mat[pivot_idx], mat[pivot_row]

        # eliminate other rows
        pivot_data = mat[pivot_row]
        for r in range(m):
            if r != pivot_row and mat[r][pivot_col] == 1:
                operations.append(('xor', pivot_row, r))
                # XOR rows from pivot_col onward
                row_r = mat[r]
                row_r[pivot_col:] = [a ^ b for a, b in zip(row_r[pivot_col:], pivot_data[pivot_col:])]

        pivot_row += 1
        pivot_col += 1

    return operations, mat
