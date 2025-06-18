"""Cirq implementation of Gauss-Jordan Elimination gate."""

import numpy as np
import cirq
from typing import List, Tuple, Union, Sequence
from src.dqi.cirq.decoding.algorithms import gauss_jordan_operations_general

Operation = Tuple[str, int, int]


class GJEGate(cirq.Gate):
    """
    Gauss–Jordan Elimination Gate (mod 2).

    Applies classical RREF row operations on a quantum register via:
      - swap(row_i, row_j) → circuit.SWAP(i, j)
      - xor(row_i, row_j)  → circuit.CNOT(i, j)

    Args:
        matrix: m×n binary matrix (as numpy array or list of lists) over GF(2).
    """
    
    def __init__(self, matrix: Union[np.ndarray, List[List[int]]]) -> None:
        super().__init__()
        
        # Convert input to nested Python lists for elimination
        if isinstance(matrix, np.ndarray):
            mat_list = matrix.tolist()
        else:
            mat_list = [list(row) for row in matrix]
        
        # Compute operations and ignore resulting RREF
        ops, _ = gauss_jordan_operations_general(mat_list)
        self.num_qubits = len(mat_list)
        self.operations: List[Operation] = ops
        self._name = "GJE"
    
    def _num_qubits_(self) -> int:
        return self.num_qubits
    
    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=[f"GJE_{i}" for i in range(self.num_qubits)]
        )
    
    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        """Decompose the GJE gate into SWAP and CNOT operations."""
        operations = []
        
        for op_type, src, tgt in self.operations:
            if op_type == "swap":
                operations.append(cirq.SWAP(qubits[src], qubits[tgt]))
            elif op_type == "xor":
                operations.append(cirq.CNOT(qubits[src], qubits[tgt]))
        
        return operations
    
    def __repr__(self) -> str:
        return f"GJEGate(num_qubits={self.num_qubits}, num_operations={len(self.operations)})" 