"""Cirq implementation of Dicke state preparation gates."""

import numpy as np
import cirq
from typing import Sequence, List


class SCS2Gate(cirq.Gate):
    """
    SCS gate for 2 qubits: CNOT-CRY-CNOT with theta=2*arccos(1/sqrt(n)).
    """
    
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
        self._name = "SCS2"
    
    def _num_qubits_(self) -> int:
        return 2
    
    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=["SCS2_0", "SCS2_1"])
    
    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        q0, q1 = qubits
        theta = 2 * np.arccos(1 / np.sqrt(self.n))
        
        return [
            cirq.CNOT(q1, q0),
            cirq.ControlledGate(cirq.ry(theta), num_controls=1).on(q0, q1),
            cirq.CNOT(q1, q0)
        ]


class SCS3Gate(cirq.Gate):
    """
    SCS gate for 3 qubits: CNOT-CCRY-CNOT with theta=2*arccos(sqrt(l/n)).
    """
    
    def __init__(self, l: int, n: int) -> None:
        super().__init__()
        self.l = l
        self.n = n
        self._name = "SCS3"
    
    def _num_qubits_(self) -> int:
        return 3
    
    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(wire_symbols=["SCS3_0", "SCS3_1", "SCS3_2"])
    
    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        q0, q1, q2 = qubits
        theta = 2 * np.arccos(np.sqrt(self.l / self.n))
        
        return [
            cirq.CNOT(q2, q0),
            cirq.ControlledGate(cirq.ry(theta), num_controls=2).on(q0, q1, q2),
            cirq.CNOT(q2, q0)
        ]


class SCSnkGate(cirq.Gate):
    """
    Composite SCS gate on k+1 qubits: one SCS2 then SCS3 for each extra qubit.
    """
    
    def __init__(self, n: int, k: int) -> None:
        super().__init__()
        self.n = n
        self.k = k
        self._name = f"SCS{n},{k}"
    
    def _num_qubits_(self) -> int:
        return self.k + 1
    
    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=[f"SCS_{i}" for i in range(self.k + 1)]
        )
    
    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        operations = []
        
        # Apply SCS2 gate
        operations.append(SCS2Gate(self.n).on(qubits[0], qubits[1]))
        
        # Apply SCS3 gates
        for l in range(2, self.k + 1):
            operations.append(SCS3Gate(l, self.n).on(qubits[0], qubits[l-1], qubits[l]))
        
        return operations


def _apply_stages(operations: List, qubits: List[cirq.Qid], n: int, k: int):
    """
    Apply two-stage SCSnk gates for U_{n,k} and Dicke preparation:
      1) For l from n down to k+1, apply SCSnk(n=l, k=k) on qubits [n-l : n-l+k+1]
      2) For l from k down to 2, apply SCSnk(n=l, k=l-1) on qubits [n-l : n]
    """
    # Stage 1: overlapping gates descending
    for l in range(n, k, -1):
        start = n - l
        gate_qubits = qubits[start:start + k + 1]
        operations.append(SCSnkGate(n=l, k=k).on(*gate_qubits))
    
    # Stage 2: refinement gates
    for l in range(k, 1, -1):
        start = n - l
        gate_qubits = qubits[start:n]
        operations.append(SCSnkGate(n=l, k=l-1).on(*gate_qubits))


class UnkGate(cirq.Gate):
    """
    Constructs the U_{n,k} state preparation gate by composing two-stage SCSnk sequence.
    """
    
    def __init__(self, n: int, k: int) -> None:
        super().__init__()
        self.n = n
        self.k = k
        self._name = f"U_{{{n},{k}}}"
    
    def _num_qubits_(self) -> int:
        return self.n
    
    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=[f"U_{i}" for i in range(self.n)]
        )
    
    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        operations = []
        _apply_stages(operations, list(qubits), self.n, self.k)
        return operations


class DickeStatePreparation(cirq.Gate):
    """
    Constructs the D_{n,k} Dicke state gate: initialize k qubits then apply U_{n,k} stages.
    """
    
    def __init__(self, n: int, k: int) -> None:
        super().__init__()
        self.n = n
        self.k = k
        self._name = f"D_{{{n},{k}}}"
    
    def _num_qubits_(self) -> int:
        return self.n
    
    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        return cirq.CircuitDiagramInfo(
            wire_symbols=[f"D_{i}" for i in range(self.n)]
        )
    
    def _decompose_(self, qubits: Sequence[cirq.Qid]) -> cirq.OP_TREE:
        operations = []
        
        # Initialize k qubits to |1>
        for i in range(self.k):
            operations.append(cirq.X(qubits[i]))
        
        # Apply U_{n,k} stages
        _apply_stages(operations, list(qubits), self.n, self.k)
        
        return operations 