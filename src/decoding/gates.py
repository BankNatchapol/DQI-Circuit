import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from typing import List, Tuple

def gauss_jordan_operations_general(
    matrix: List[List[int]]
) -> Tuple[List[Tuple[str, int, int]], List[List[int]]]:
    """
    Perform Gauss–Jordan elimination mod 2 on an m x n augmented matrix and record row operations.

    The input matrix is assumed to be a list of lists of 0s and 1s representing the augmented 
    matrix [A | b]. The function returns a tuple (operations, rref_matrix), where each operation is 
    represented as:
      - ('swap', i, j): swap row i with row j.
      - ('xor', i, j): replace row j with row j XOR row i.

    Args:
        matrix: The augmented matrix (List of List of ints) over GF(2).

    Returns:
        A tuple containing:
          - A list of operations (as tuples of operation type and row indices).
          - The resulting matrix in reduced row echelon form (RREF).
    """
    m: int = len(matrix)
    num_cols: int = len(matrix[0])
    n: int = num_cols - 1  # Assumes the last column is the right-hand side.
    operations: List[Tuple[str, int, int]] = []
    # Create a deep copy of the matrix.
    mat: List[List[int]] = [row[:] for row in matrix]
    pivot_row: int = 0
    pivot_col: int = 0

    while pivot_row < m and pivot_col < n:
        pivot_idx = None
        for r in range(pivot_row, m):
            if mat[r][pivot_col] == 1:
                pivot_idx = r
                break
        if pivot_idx is None:
            pivot_col += 1
            continue
        if pivot_idx != pivot_row:
            operations.append(('swap', pivot_row, pivot_idx))
            mat[pivot_row], mat[pivot_idx] = mat[pivot_idx], mat[pivot_row]
        for r in range(m):
            if r != pivot_row and mat[r][pivot_col] == 1:
                operations.append(('xor', pivot_row, r))
                for c in range(pivot_col, num_cols):
                    mat[r][c] ^= mat[pivot_row][c]
        pivot_row += 1
        pivot_col += 1

    return operations, mat

class GJEGate(Gate):
    """
    Custom gate implementing Gauss–Jordan elimination (mod 2) on a binary matrix.

    This gate computes the Gauss–Jordan elimination operations on an augmented binary 
    matrix [A | b] (with b implicitly included in A) and then applies these row operations 
    on a quantum register representing b.

    Note:
      - The gate acts on m qubits, where m is the number of rows in A.
      - Operations are recorded as swaps and XORs (implemented as CNOTs).
    """
    def __init__(self, A: np.ndarray) -> None:
        """
        Initialize the GJEGate.

        Args:
            A: The binary coefficient matrix (shape m x n) over GF(2).
        """
        A_list: List[List[int]] = A.tolist()  # Convert to a list of lists.
        m: int = len(A_list)
        ops, _ = gauss_jordan_operations_general(A_list)
        super().__init__("GJE", m, [])
        self.operations: List[Tuple[str, int, int]] = ops
        self.A: np.ndarray = A

    def _define(self) -> None:
        """
        Define the gate's decomposition as a QuantumCircuit.

        The circuit applies the recorded row operations (swaps and XORs) according to the 
        Gauss–Jordan elimination mod 2.
        """
        m: int = len(self.A)
        qc = QuantumCircuit(m, name="GJE")
        for op in self.operations:
            op_type, i, j = op
            if op_type == 'swap':
                qc.swap(i, j)
            elif op_type == 'xor':
                qc.cx(i, j)
        self.definition = qc
