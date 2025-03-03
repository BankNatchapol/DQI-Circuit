import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

# (Assuming gauss_jordan_operations_general is defined as in your code)
def gauss_jordan_operations_general(matrix):
    """
    Perform Gauss–Jordan elimination mod 2 on an m x n matrix
    and record row operations.

    Returns (operations, rref_matrix), where each operation is:
      - ('swap', i, j): swap row i with row j.
      - ('xor', i, j): replace row j with row j XOR row i.
    """
    m = len(matrix)
    num_cols = len(matrix[0])
    n = num_cols - 1
    operations = []
    mat = [row[:] for row in matrix]  # deep copy
    pivot_row = 0
    pivot_col = 0
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
    A custom gate that encapsulates the elimination (row operations) on the b‑register only.
    
    Given a binary coefficient matrix A and right-hand side vector b (as NumPy arrays),
    the gate computes the recorded Gauss–Jordan operations on the augmented matrix [A | b]
    and then applies those operations on a register representing b.
    
    Note: This gate includes measurements on the b‑register.
    """
    def __init__(self, A):
        """
        Args:
            A (np.ndarray): The binary coefficient matrix (shape m x n).
            b (np.ndarray): The binary right-hand side vector (shape m x 1 or (m,)).
        """
        # Ensure b is a 1D list of bits.
        
        A_list = A.tolist()
        m = len(A_list)
        # Record the row operations via Gauss–Jordan elimination mod2.
        ops, _ = gauss_jordan_operations_general(A_list)
        # Initialize the gate: acts on m qubits.
        super().__init__("GJE", m, [])
        self.operations = ops
        self.A = A

    def _define(self):
        m = len(self.A)
        qc = QuantumCircuit(m)
        # Apply the recorded row operations.
        for op in self.operations:
            op_type, i, j = op
            if op_type == 'swap':
                qc.swap(i, j)
            elif op_type == 'xor':
                qc.cx(i, j)
        self.definition = qc
