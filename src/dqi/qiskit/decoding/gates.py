import numpy as np
from typing import List, Tuple, Union
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from .algorithms import gauss_jordan_operations_general

Operation = Tuple[str, int, int]

class GJEGate(Gate):
    """
    Gauss–Jordan Elimination Gate (mod 2).

    Applies classical RREF row operations on a quantum register via:
      - swap(row_i, row_j) → qc.swap(i, j)
      - xor(row_i, row_j)  → qc.cx(i, j)

    Args:
        matrix: m×n binary matrix (as numpy array or list of lists) over GF(2).
    """
    def __init__(self, matrix: Union[np.ndarray, List[List[int]]]) -> None:
        # Convert input to nested Python lists for elimination
        if isinstance(matrix, np.ndarray):
            mat_list = matrix.tolist()
        else:
            mat_list = [list(row) for row in matrix]

        # Compute operations and ignore resulting RREF
        ops, _ = gauss_jordan_operations_general(mat_list)
        num_qubits = len(mat_list)

        super().__init__("GJE", num_qubits, [])
        self.operations: List[Operation] = ops
        # Using self.num_qubits and self.name from Gate base

    def _define(self) -> None:
        qc = QuantumCircuit(self.num_qubits, name=self.name)
        for op_type, src, tgt in self.operations:
            if op_type == "swap":
                qc.swap(src, tgt)
            elif op_type == "xor":
                qc.cx(src, tgt)
        self.definition = qc